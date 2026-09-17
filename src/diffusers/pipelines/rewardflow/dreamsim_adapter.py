"""Explicit optional-dependency boundary for RewardSlider V1 DreamSim losses.

No proxy (including LPIPS) is used when DreamSim is unavailable.  The public
DreamSim package's standard PIL preprocessing is not proven to preserve image
gradients, so V1 intentionally fails fast instead of claiming a differentiable
implementation without a verified torch-native preprocessing path.
"""

from __future__ import annotations

from pathlib import Path

import torch


class DreamSimUnavailableError(RuntimeError):
    """Raised when the official DreamSim differentiable path is unavailable."""


class DreamSimAdapter:
    """Fail-fast adapter placeholder for an audited official DreamSim path."""

    def __init__(self, *, model_path: str | None = None, device: torch.device | str | None = None):
        try:
            import dreamsim  # noqa: F401
        except ImportError as error:
            raise DreamSimUnavailableError(
                "RewardSlider V1 requires the official `dreamsim` package for trajectory geometry; "
                "it is not installed. No LPIPS or other substitute was selected."
            ) from error
        if model_path is not None and not Path(model_path).exists():
            raise DreamSimUnavailableError(f"Requested DreamSim local model path does not exist: {model_path}")
        raise DreamSimUnavailableError(
            "The installed DreamSim API needs an audited torch-native differentiable preprocessing adapter before "
            "it can be used for RewardSlider V1. Refusing to use its PIL path and silently detach candidate images."
        )

    @classmethod
    def availability_reason(cls, *, model_path: str | None = None) -> str | None:
        try:
            cls(model_path=model_path)
        except DreamSimUnavailableError as error:
            return str(error)
        return None
