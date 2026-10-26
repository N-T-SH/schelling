"""Config for SchellingModel. Self-contained: it is copied to the Hub repo for trust_remote_code."""

from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig


class SchellingConfig(PretrainedConfig):
    model_type = "schelling"

    def __init__(
        self,
        backbone_config: dict[str, Any] | None = None,
        backbone_name: str | None = None,
        head_hidden: int = 256,
        temperature_noul: float = 1.0,
        temperature_choice: float = 1.0,
        temperature_score: float = 1.0,
        max_context_tokens: int = 512,
        max_option_tokens: int = 32,
        history_items: int = 20,
        **kwargs: Any,
    ) -> None:
        # The backbone's own config is embedded so loading never needs the network.
        self.backbone_config = backbone_config or {}
        self.backbone_name = backbone_name  # provenance only; weights live in this checkpoint
        self.head_hidden = head_hidden
        # Post-hoc calibration (fitted on validation in M4); 1.0 means uncalibrated.
        self.temperature_noul = temperature_noul
        self.temperature_choice = temperature_choice
        self.temperature_score = temperature_score
        self.max_context_tokens = max_context_tokens
        self.max_option_tokens = max_option_tokens
        self.history_items = history_items
        super().__init__(**kwargs)

    @property
    def backbone_hidden_size(self) -> int:
        return int(self.backbone_config["hidden_size"])
