"""SchellingTwoTower: question tower, person tower and a prior-plus-residual output (decision 0004).

Relative imports only: these files are copied to the Hub repo for trust_remote_code.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoConfig, AutoModel, PreTrainedModel

from .configuration_schelling_tt import SchellingTwoTowerConfig
from .heads import choice_probs, mean_pool, noul_probs, score_conditionals, score_probs
from .heads_tt import TTOut, TwoTowerHeads

TASKS = ("noul", "choice", "score")
MAX_CACHE = 400_000


class SchellingTwoTower(PreTrainedModel):
    config_class = SchellingTwoTowerConfig
    base_model_prefix = "schelling_tt"

    def __init__(self, config: SchellingTwoTowerConfig) -> None:
        super().__init__(config)
        self.backbone = AutoModel.from_config(AutoConfig.for_model(**dict(config.backbone_config)))
        self.heads = TwoTowerHeads(
            config.backbone_hidden_size,
            config.dim,
            config.person_heads,
            config.dropout,
            config.use_person,
        )
        self.use_cache = True  # inference only; ignored while training or with gradients enabled
        self._cache: dict[bytes, torch.Tensor] = {}
        self.post_init()
        self.heads.reset()  # post_init re-initialises every Linear; restore default init, residual at 0

    @classmethod
    def from_backbone(cls, name_or_path: str, **config_kwargs: Any) -> SchellingTwoTower:
        backbone = AutoModel.from_pretrained(name_or_path)
        config = SchellingTwoTowerConfig(
            backbone_config=backbone.config.to_dict(), backbone_name=name_or_path, **config_kwargs
        )
        model = cls(config)
        model.backbone.load_state_dict(backbone.state_dict())
        return model

    def init_residual(self, std: float) -> None:
        self.heads.init_residual(std)

    # -- text -> vectors ----------------------------------------------------------------------

    def _encode(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state
        return mean_pool(hidden, mask)

    def encode_texts(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """(N, T) token ids -> (N, H). Identical texts are encoded once; with gradients off and in
        eval mode, vectors are also remembered across calls, keyed by their token sequence."""
        ids_np = ids.detach().cpu().numpy()
        lens = mask.detach().sum(1).cpu().numpy()
        keys = [ids_np[i, : lens[i]].tobytes() for i in range(len(ids_np))]
        first: dict[bytes, int] = {}
        for i, k in enumerate(keys):
            first.setdefault(k, i)
        use_cache = self.use_cache and not self.training and not torch.is_grad_enabled()
        todo = [k for k in first if not (use_cache and k in self._cache)]
        fresh: dict[bytes, torch.Tensor] = {}
        if todo:
            rows = torch.tensor([first[k] for k in todo], device=ids.device)
            enc = self._encode(ids[rows], mask[rows])
            fresh = {k: enc[j] for j, k in enumerate(todo)}
            if use_cache:
                if len(self._cache) + len(fresh) > MAX_CACHE:
                    self._cache.clear()
                self._cache.update(fresh)
        lookup = self._cache if use_cache else fresh
        return torch.stack([lookup[k] if k in lookup else fresh[k] for k in keys])

    def _encode_grouped(
        self, ids: torch.Tensor, mask: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        """(B, K, T) texts with a (B, K) validity mask -> (B, K, H); invalid slots stay zero."""
        b, k, t = ids.shape
        out = torch.zeros(b * k, self.config.backbone_hidden_size, device=ids.device)
        flat = valid.reshape(-1)
        if flat.any():
            out[flat] = self.encode_texts(ids.reshape(-1, t)[flat], mask.reshape(-1, t)[flat]).to(
                out.dtype
            )
        return out.reshape(b, k, -1)

    # -- combine ------------------------------------------------------------------------------

    def combine(
        self,
        task: str,
        u: torch.Tensor,
        attr_vecs: torch.Tensor,
        attr_mask: torch.Tensor,
        e: torch.Tensor | None = None,
        option_mask: torch.Tensor | None = None,
    ) -> TTOut:
        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {task!r}")
        return self.heads(task, u, attr_vecs, attr_mask, e, option_mask)

    def raw_parts(
        self,
        task: str,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        attr_input_ids: torch.Tensor,
        attr_attention_mask: torch.Tensor,
        attr_mask: torch.Tensor,
        option_input_ids: torch.Tensor | None = None,
        option_attention_mask: torch.Tensor | None = None,
        option_mask: torch.Tensor | None = None,
    ) -> TTOut:
        u = self.encode_texts(input_ids, attention_mask)
        attrs = self._encode_grouped(attr_input_ids, attr_attention_mask, attr_mask)
        e = None
        if task != "noul":
            if option_input_ids is None or option_attention_mask is None or option_mask is None:
                raise ValueError(f"{task} needs option tensors")
            e = self._encode_grouped(option_input_ids, option_attention_mask, option_mask)
        return self.combine(task, u, attrs, attr_mask, e, option_mask)

    def raw(self, task: str, **tensors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        out = self.raw_parts(task, **tensors)
        return out.logits, out.mask

    def probs_from(
        self, task: str, logits: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        if task == "noul":
            return noul_probs(logits, self.config.temperature_noul)
        assert mask is not None
        if task == "choice":
            return choice_probs(logits, mask, self.config.temperature_choice)
        return score_probs(score_conditionals(logits, mask, self.config.temperature_score))

    def forward(self, task: str, **tensors: torch.Tensor) -> torch.Tensor:
        logits, mask = self.raw(task, **tensors)
        return self.probs_from(task, logits, mask)

    def score_conditional_probs(self, **tensors: torch.Tensor) -> torch.Tensor:
        logits, mask = self.raw("score", **tensors)
        assert mask is not None
        return score_conditionals(logits, mask, self.config.temperature_score)


SchellingTwoTowerConfig.register_for_auto_class("AutoConfig")
SchellingTwoTower.register_for_auto_class("AutoModel")
