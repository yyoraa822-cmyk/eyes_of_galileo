"""Abstract base class for all counterfactual physics scenarios."""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


@dataclass
class ScenarioMeta:
    name: str
    slug: str
    description: str
    control_var: str
    control_label: str
    observable_label: str
    true_exponent: float
    law_template: str
    historical_instrument: str
    formula_type: str = "power_law"
    true_params: dict = field(default_factory=dict)


class Scenario(ABC):
    """Base class for a counterfactual physics scenario.

    Each scenario encodes a physical law of the form:
        observable = C * control^alpha        (power_law)
        observable = A * exp(rate * control)  (exponential)
        observable = A * log(control) + B     (logarithmic)
        observable = slope * control + B      (linear)
    """

    def __init__(self, alpha: float | None = None, seed: int = 42):
        self.alpha = alpha if alpha is not None else self.meta.true_exponent
        self.rng = np.random.default_rng(seed)
        self._g = 9.81

    @property
    @abstractmethod
    def meta(self) -> ScenarioMeta:
        ...

    @abstractmethod
    def simulate(self, control_value: float, dt: float = 0.05,
                 max_t: float = 10.0) -> dict[str, Any]:
        """Run the physics simulation. Returns a dict with at least 'times' key."""
        ...

    @abstractmethod
    def render_frames(self, sim_data: dict[str, Any],
                      figsize: tuple[float, float] = (6, 5),
                      dpi: int = 100,
                      clean: bool = False) -> list[Image.Image]:
        """Render a list of PIL Image frames for the simulation.

        Args:
            clean: If True, remove all numerical annotations and text labels
                from the rendered image. Used for vision-only mode where the
                agent must extract measurements from pixel geometry alone.
        """
        ...

    @abstractmethod
    def get_observable(self, control_value: float) -> float:
        """Return the ground-truth scalar observable for a given control value."""
        ...

    @property
    def default_controls(self) -> list[float]:
        """Suggested control values for the agent to try."""
        return [1.0, 2.0, 4.0, 8.0]

    def render_gif(self, sim_data: dict[str, Any],
                   out_path: str | Path | None = None,
                   fps: int = 10, **render_kw) -> bytes | Path:
        """Render simulation to an animated GIF."""
        frames = self.render_frames(sim_data, **render_kw)
        if not frames:
            raise ValueError("render_frames returned empty list")
        duration_ms = int(1000 / fps)
        buf = io.BytesIO()
        frames[0].save(buf, format="GIF", save_all=True,
                       append_images=frames[1:], duration=duration_ms,
                       loop=0, optimize=True)
        if out_path is not None:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(buf.getvalue())
            return out_path
        return buf.getvalue()

    def render_static(self, sim_data: dict[str, Any],
                      out_path: str | Path | None = None,
                      **render_kw) -> bytes | Path:
        """Render a single static image (first frame)."""
        frames = self.render_frames(sim_data, **render_kw)
        buf = io.BytesIO()
        frames[0].save(buf, format="PNG")
        if out_path is not None:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(buf.getvalue())
            return out_path
        return buf.getvalue()

    def evaluate(self, submitted_alpha: float) -> dict[str, float]:
        """Score the agent's submitted exponent (V1 backward compat)."""
        error = abs(submitted_alpha - self.alpha)
        denom = max(abs(self.alpha), 0.1)
        relative_error = error / denom
        score = max(0.0, 1.0 - relative_error)
        return {
            "submitted_alpha": submitted_alpha,
            "true_alpha": self.alpha,
            "abs_error": round(error, 6),
            "relative_error": round(relative_error, 6),
            "score": round(score, 6),
        }

    def evaluate_v2(self, submitted_form: str, submitted_params: dict) -> dict:
        """Score full discovery: formula type + parameters.

        Returns dict with form_correct, param_accuracy, and overall score.
        """
        true_form = self.meta.formula_type
        form_correct = 1.0 if submitted_form == true_form else 0.0

        param_accuracy = 0.0
        if true_form == "power_law":
            true_val = self.alpha
            sub_val = submitted_params.get("exponent", submitted_params.get("alpha", 0))
            denom = max(abs(true_val), 0.01)
            param_accuracy = max(0.0, 1.0 - abs(float(sub_val) - true_val) / denom)
        elif true_form == "exponential":
            true_rate = self.meta.true_params.get("rate", 0)
            sub_rate = float(submitted_params.get("rate", submitted_params.get("decay_rate", 0)))
            denom = max(abs(true_rate), 0.01)
            param_accuracy = max(0.0, 1.0 - abs(sub_rate - true_rate) / denom)
        elif true_form == "logarithmic":
            true_coeff = self.meta.true_params.get("coefficient", 0)
            sub_coeff = float(submitted_params.get("coefficient", submitted_params.get("a", 0)))
            denom = max(abs(true_coeff), 0.01)
            param_accuracy = max(0.0, 1.0 - abs(sub_coeff - true_coeff) / denom)
        elif true_form == "linear":
            true_slope = self.meta.true_params.get("slope", 0)
            sub_slope = float(submitted_params.get("slope", 0))
            denom = max(abs(true_slope), 0.01)
            param_accuracy = max(0.0, 1.0 - abs(sub_slope - true_slope) / denom)

        overall = form_correct * param_accuracy

        return {
            "form_correct": form_correct,
            "submitted_form": submitted_form,
            "true_form": true_form,
            "param_accuracy": round(param_accuracy, 6),
            "submitted_params": submitted_params,
            "true_params": self.meta.true_params or {"exponent": self.alpha},
            "overall": round(overall, 6),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(alpha={self.alpha})"


# ── Rendering helpers ────────────────────────────────────────────────────────

def fig_to_pil(fig: plt.Figure, dpi: int = 100) -> Image.Image:
    """Convert a matplotlib Figure to a PIL Image and close the figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def add_visual_noise(img: Image.Image, level: str = "none") -> Image.Image:
    """Degrade a PIL image to simulate non-ideal visual conditions.

    Levels:
      none   — no degradation
      mild   — light Gaussian noise (σ=10) + slight blur (r=0.5)
      medium — moderate noise (σ=25) + blur (r=1.5) + downsample 50%
      severe — heavy noise (σ=50) + blur (r=3.0) + downsample 25% + JPEG artifacts
    """
    if level == "none":
        return img

    from PIL import ImageFilter

    arr = np.array(img, dtype=np.float32)

    if level == "mild":
        noise = np.random.normal(0, 10, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

    elif level == "medium":
        noise = np.random.normal(0, 25, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
        w, h = img.size
        img = img.resize((w // 2, h // 2), Image.BILINEAR).resize((w, h), Image.BILINEAR)

    elif level == "severe":
        noise = np.random.normal(0, 50, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        img = img.filter(ImageFilter.GaussianBlur(radius=3.0))
        w, h = img.size
        img = img.resize((w // 4, h // 4), Image.BILINEAR).resize((w, h), Image.BILINEAR)
        jpeg_buf = io.BytesIO()
        img.save(jpeg_buf, format="JPEG", quality=15)
        jpeg_buf.seek(0)
        img = Image.open(jpeg_buf).convert("RGB")

    return img


def make_clean_fig(figsize=(6, 5), bg="white") -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize, facecolor=bg)
    ax.set_facecolor(bg)
    return fig, ax
