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
