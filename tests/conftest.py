import pytest

from configs.model_config import default_config


@pytest.fixture
def tiny_cfg():
    """Small config so tests run in seconds on CPU."""
    return default_config(vocab_size=128, context_len=32, d_model=64, n_layers=2,
                          n_heads=4, ffn_hidden=176)
