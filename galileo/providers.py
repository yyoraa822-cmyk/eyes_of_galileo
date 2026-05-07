"""VLM / LLM provider — OpenAI-compatible chat completions with image support."""

from __future__ import annotations

import base64
import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from PIL import Image


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str | list[dict]
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    def to_api(self) -> dict:
        msg: dict[str, Any] = {"role": self.role}
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.name:
            msg["name"] = self.name
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        msg["content"] = self.content
        return msg


def image_to_data_url(img: Image.Image, fmt: str = "png",
                      max_size: int = 768) -> str:
    """Encode PIL Image as base64 data URL, resizing if needed."""
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format=fmt.upper())
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt};base64,{b64}"


def gif_to_data_url(gif_bytes: bytes) -> str:
    """Encode raw GIF bytes as a base64 data URL."""
    b64 = base64.b64encode(gif_bytes).decode()
    return f"data:image/gif;base64,{b64}"


def make_image_content(image_or_text, images=None):
    """Build multimodal content.

    Supports two call signatures:
      make_image_content(pil_image)           -> single image_url dict
      make_image_content(text, [pil_images])  -> list of text + image_url dicts

    Images can be PIL.Image or raw bytes (for animated GIFs).
    """
    if images is not None:
        parts: list[dict] = []
        if image_or_text:
            parts.append({"type": "text", "text": image_or_text})
        for img in images:
            if isinstance(img, bytes):
                url = gif_to_data_url(img)
            else:
                url = image_to_data_url(img)
            parts.append({
                "type": "image_url",
                "image_url": {"url": url, "detail": "high"},
            })
        return parts

    if isinstance(image_or_text, bytes):
        url = gif_to_data_url(image_or_text)
    else:
        url = image_to_data_url(image_or_text)
    return {
        "type": "image_url",
        "image_url": {"url": url, "detail": "high"},
    }


@dataclass
class VLMProvider:
    """OpenAI-compatible chat completions client with tool-calling support."""

    api_base: str
    api_key: str
    model: str = "gpt-4o"
    temperature: float = 0.2
    max_tokens: int = 4096
    max_retries: int = 5
    timeout: int = 180
    seed: int | None = None

    def _prefer_responses_api(self) -> bool:
        """Use /responses for GPT-5 family on OpenAI-compatible backends."""
        m = (self.model or "").lower()
        return m.startswith("gpt-5")

    def _convert_tools_for_responses(self, tools: list[dict] | None) -> list[dict] | None:
        if not tools:
            return None
        out: list[dict] = []
        for t in tools:
            if t.get("type") != "function":
                continue
            fn = t.get("function", {})
            out.append({
                "type": "function",
                "name": fn.get("name"),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {"type": "object"}),
            })
        return out or None

    def _content_to_responses(self, content: str | list[dict], *, role: str) -> list[dict]:
        if isinstance(content, str):
            ctype = "output_text" if role == "assistant" else "input_text"
            return [{"type": ctype, "text": content}]

        parts: list[dict] = []
        for p in content:
            ptype = p.get("type")
            if ptype == "text":
                ctype = "output_text" if role == "assistant" else "input_text"
                parts.append({"type": ctype, "text": p.get("text", "")})
            elif ptype == "image_url":
                img = p.get("image_url", {})
                url = img.get("url")
                if url:
                    parts.append({"type": "input_image", "image_url": url})
        return parts

    def _messages_to_responses_input(self, messages: list[Message]) -> list[dict]:
        items: list[dict] = []
        for m in messages:
            # Tool outputs from runner -> function_call_output item.
            if m.role == "tool":
                output = m.content if isinstance(m.content, str) else json.dumps(m.content)
                if m.tool_call_id:
                    items.append({
                        "type": "function_call_output",
                        "call_id": m.tool_call_id,
                        "output": output,
                    })
                else:
                    # Fallback: keep as assistant text if call_id is missing.
                    items.append({
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": output}],
                    })
                continue

            # Assistant function calls -> function_call item(s).
            if m.role == "assistant" and m.tool_calls:
                if m.content:
                    items.append({
                        "type": "message",
                        "role": "assistant",
                        "content": self._content_to_responses(m.content, role="assistant"),
                    })
                for tc in m.tool_calls:
                    fn = tc.get("function", {})
                    call_id = tc.get("id") or f"call_{len(items)}"
                    items.append({
                        "type": "function_call",
                        "call_id": call_id,
                        "name": fn.get("name", ""),
                        "arguments": fn.get("arguments", "{}"),
                    })
                continue

            # Regular system/user/assistant messages.
            if m.role in ("system", "user", "assistant"):
                items.append({
                    "type": "message",
                    "role": m.role,
                    "content": self._content_to_responses(m.content, role=m.role),
                })
        return items

    def _chat_responses(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        tool_choice: str | dict = "auto",
    ) -> Message:
        url = f"{self.api_base}/responses"
        body: dict[str, Any] = {
            "model": self.model,
            "input": self._messages_to_responses_input(messages),
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
        }
        if self.seed is not None:
            body["seed"] = int(self.seed)
        rtools = self._convert_tools_for_responses(tools)
        if rtools:
            body["tools"] = rtools
            body["tool_choice"] = tool_choice

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(
                    url, headers=self._headers(), json=body, timeout=self.timeout,
                )
                if resp.status_code in (429, 500, 502, 503, 504):
                    wait = min(2 ** attempt * 2, 60)
                    print(f"  [provider] {resp.status_code} on attempt {attempt}, "
                          f"retrying in {wait}s... (body: {resp.text[:200]})")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()

                text_parts: list[str] = []
                tool_calls: list[dict] = []
                for i, item in enumerate(data.get("output", [])):
                    itype = item.get("type")
                    if itype == "message" and item.get("role") == "assistant":
                        for c in item.get("content", []):
                            if c.get("type") in ("output_text", "text"):
                                t = c.get("text")
                                if t:
                                    text_parts.append(t)
                    elif itype == "function_call":
                        call_id = item.get("call_id") or item.get("id") or f"fc_{i}"
                        tool_calls.append({
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": item.get("name", ""),
                                "arguments": item.get("arguments", "{}"),
                            },
                        })

                return Message(
                    role="assistant",
                    content="\n".join(text_parts).strip(),
                    tool_calls=tool_calls or None,
                )
            except (requests.Timeout, requests.ConnectionError):
                if attempt == self.max_retries:
                    raise
                time.sleep(2 ** attempt)

        raise RuntimeError(f"Failed after {self.max_retries} retries")

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        tool_choice: str | dict = "auto",
    ) -> Message:
        """Send a chat completion request and return the assistant message."""
        if self._prefer_responses_api():
            return self._chat_responses(messages, tools=tools, tool_choice=tool_choice)

        url = f"{self.api_base}/chat/completions"
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_api() for m in messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.seed is not None:
            body["seed"] = int(self.seed)
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(
                    url, headers=self._headers(),
                    json=body, timeout=self.timeout,
                )
                if resp.status_code in (429, 500, 502, 503, 504):
                    wait = min(2 ** attempt * 2, 60)
                    print(f"  [provider] {resp.status_code} on attempt {attempt}, "
                          f"retrying in {wait}s... "
                          f"(body: {resp.text[:200]})")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]["message"]

                return Message(
                    role="assistant",
                    content=choice.get("content") or "",
                    tool_calls=choice.get("tool_calls"),
                )
            except (requests.Timeout, requests.ConnectionError) as e:
                if attempt == self.max_retries:
                    raise
                time.sleep(2 ** attempt)

        raise RuntimeError(f"Failed after {self.max_retries} retries")
