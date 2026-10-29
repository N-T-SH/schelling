"""Shared builders for the tiny random backbone used by model tests (no network)."""

from pathlib import Path

import torch
from transformers import BertTokenizer

from schelling.model import SchellingConfig, SchellingModel

VOCAB = Path(__file__).parent / "fixtures" / "tiny_vocab.txt"


def tiny_tokenizer() -> BertTokenizer:
    # transformers 5 takes the vocabulary as a mapping, not a file path.
    vocab = {tok: i for i, tok in enumerate(VOCAB.read_text().split())}
    return BertTokenizer(vocab=vocab, do_lower_case=True)


def tiny_model(seed: int = 0, **config_kwargs: object) -> SchellingModel:
    vocab_size = len(VOCAB.read_text().split())
    backbone = {
        "model_type": "bert",
        "vocab_size": vocab_size,
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "intermediate_size": 64,
        "max_position_embeddings": 512,
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
    }
    config = SchellingConfig(
        backbone_config=backbone,
        head_hidden=16,
        max_context_tokens=128,
        max_option_tokens=8,
        **config_kwargs,
    )
    torch.manual_seed(seed)
    return SchellingModel(config).eval()


def tiny_backbone_config() -> dict[str, object]:
    return {
        "model_type": "bert",
        "vocab_size": len(VOCAB.read_text().split()),
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "intermediate_size": 64,
        "max_position_embeddings": 512,
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
    }


def tiny_tt_model(seed: int = 0, residual_std: float = 0.0, **config_kwargs: object):  # type: ignore[no-untyped-def]
    """Tiny two-tower model. `residual_std` > 0 makes the residual branch non-zero (it starts at
    exactly zero by design), so tests can see person effects."""
    from schelling.model.configuration_schelling_tt import SchellingTwoTowerConfig
    from schelling.model.modeling_schelling_tt import SchellingTwoTower

    kwargs = {
        "dim": 16,
        "person_heads": 2,
        "max_question_tokens": 96,
        "max_option_tokens": 8,
        "max_attr_tokens": 8,
        **config_kwargs,
    }
    config = SchellingTwoTowerConfig(backbone_config=tiny_backbone_config(), **kwargs)
    torch.manual_seed(seed)
    model = SchellingTwoTower(config).eval()
    if residual_std > 0:
        model.init_residual(residual_std)
    return model
