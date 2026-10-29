"""Config for the two-tower model. Self-contained: copied to the Hub repo for trust_remote_code."""

from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig


class SchellingTwoTowerConfig(PretrainedConfig):
    model_type = "schelling_tt"

    def __init__(
        self,
        backbone_config: dict[str, Any] | None = None,
        backbone_name: str | None = None,
        dim: int = 256,
        person_heads: int = 4,
        dropout: float = 0.1,
        use_person: bool = True,
        temperature_noul: float = 1.0,
        temperature_choice: float = 1.0,
        temperature_score: float = 1.0,
        max_question_tokens: int = 192,
        max_option_tokens: int = 16,
        max_attr_tokens: int = 16,
        max_attributes: int = 32,
        **kwargs: Any,
    ) -> None:
        self.backbone_config = backbone_config or {}
        self.backbone_name = backbone_name  # provenance only
        self.dim = dim
        self.person_heads = person_heads
        self.dropout = dropout
        self.use_person = use_person  # ablation: False removes the person tower's effect
        self.temperature_noul = temperature_noul
        self.temperature_choice = temperature_choice
        self.temperature_score = temperature_score
        self.max_question_tokens = max_question_tokens
        self.max_option_tokens = max_option_tokens
        self.max_attr_tokens = max_attr_tokens
        self.max_attributes = max_attributes
        super().__init__(**kwargs)

    @property
    def backbone_hidden_size(self) -> int:
        return int(self.backbone_config["hidden_size"])
