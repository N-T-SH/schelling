"""SchellingModel: one shared text encoder feeding Noul, Choice and Score heads.

Relative imports only, and no dependency on the rest of the `schelling` package, because this
file and its siblings are copied into the Hub repo and loaded with trust_remote_code.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoConfig, AutoModel, PreTrainedModel

from .configuration_schelling import SchellingConfig
from .heads import (
    ChoiceHead,
    NoulHead,
    ScoreHead,
    choice_probs,
    mean_pool,
    noul_probs,
    score_conditionals,
    score_probs,
)

TASKS = ("noul", "choice", "score")


class SchellingModel(PreTrainedModel):
    config_class = SchellingConfig
    base_model_prefix = "schelling"

    def __init__(self, config: SchellingConfig) -> None:
        super().__init__(config)
        bc = dict(config.backbone_config)
        self.backbone = AutoModel.from_config(AutoConfig.for_model(**bc))
        dim = config.backbone_hidden_size
        self.noul_head = NoulHead(dim, config.head_hidden)
        self.choice_head = ChoiceHead(dim)
        self.score_head = ScoreHead(dim, config.head_hidden)
        self.post_init()

    @classmethod
    def from_backbone(cls, name_or_path: str, **config_kwargs: Any) -> SchellingModel:
        """Wrap a pretrained encoder (heads are freshly initialised)."""
        backbone = AutoModel.from_pretrained(name_or_path)
        config = SchellingConfig(
            backbone_config=backbone.config.to_dict(), backbone_name=name_or_path, **config_kwargs
        )
        model = cls(config)
        model.backbone.load_state_dict(backbone.state_dict())
        return model

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return mean_pool(hidden, attention_mask)

    def _encode_texts(
        self, ids: torch.Tensor, mask: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        """(B, K, T) option texts -> (B, K, H); padded options are not run through the encoder.

        Identical texts (the same Likert levels recur across hundreds of questions) are encoded
        once per call and the result is gathered back, which changes nothing numerically.
        """
        b, k, t = ids.shape
        flat_valid = valid.reshape(-1)
        out = torch.zeros(b * k, self.config.backbone_hidden_size, device=ids.device)
        if flat_valid.any():
            rows = torch.cat([ids.reshape(-1, t), mask.reshape(-1, t)], dim=1)[flat_valid]
            uniq, inverse = torch.unique(rows.cpu(), dim=0, return_inverse=True)
            uniq = uniq.to(ids.device)
            enc = self.encode(uniq[:, :t], uniq[:, t:])
            out[flat_valid] = enc[inverse.to(ids.device)].to(out.dtype)
        return out.reshape(b, k, -1)

    def score_survival(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        option_input_ids: torch.Tensor,
        option_attention_mask: torch.Tensor,
        option_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.encode(input_ids, attention_mask)
        levels = self._encode_texts(option_input_ids, option_attention_mask, option_mask)
        return self.score_head.survival(h, levels, option_mask, self.config.temperature_score)

    def raw(
        self,
        task: str,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        option_input_ids: torch.Tensor | None = None,
        option_attention_mask: torch.Tensor | None = None,
        option_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Pre-temperature logits and the option/level mask: (B,) for Noul, (B, K) for Choice,
        (B, L-1) threshold logits for Score. Temperature calibration refits on these."""
        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {task!r}")
        h = self.encode(input_ids, attention_mask)
        if task == "noul":
            return self.noul_head.logit(h), None
        if option_input_ids is None or option_attention_mask is None or option_mask is None:
            raise ValueError(
                f"{task} needs option_input_ids, option_attention_mask and option_mask"
            )
        opts = self._encode_texts(option_input_ids, option_attention_mask, option_mask)
        if task == "choice":
            return self.choice_head.logits(h, opts), option_mask
        return self.score_head.threshold_logits(h, opts), option_mask

    def forward(
        self,
        task: str,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        option_input_ids: torch.Tensor | None = None,
        option_attention_mask: torch.Tensor | None = None,
        option_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Probabilities over the answers of one task type, shape (B, 2), (B, K) or (B, L).

        Each row is one independent question, so an answer cannot depend on other questions.
        """
        logits, mask = self.raw(
            task, input_ids, attention_mask, option_input_ids, option_attention_mask, option_mask
        )
        if task == "noul":
            return noul_probs(logits, self.config.temperature_noul)
        assert mask is not None
        if task == "choice":
            return choice_probs(logits, mask, self.config.temperature_choice)
        return score_probs(score_conditionals(logits, mask, self.config.temperature_score))

    def score_conditional_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        option_input_ids: torch.Tensor,
        option_attention_mask: torch.Tensor,
        option_mask: torch.Tensor,
    ) -> torch.Tensor:
        """P(y > k | y >= k): what the CORN loss is defined on."""
        t, mask = self.raw(
            "score", input_ids, attention_mask, option_input_ids, option_attention_mask, option_mask
        )
        assert mask is not None
        return score_conditionals(t, mask, self.config.temperature_score)


SchellingConfig.register_for_auto_class("AutoConfig")
SchellingModel.register_for_auto_class("AutoModel")
