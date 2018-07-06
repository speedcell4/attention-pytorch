import unittest

import torch
from hypothesis import given, strategies as st

from attentions import MultiHeadAttention
from testing import rand_mask


class TestMultiHeadAttention(unittest.TestCase):
    @given(
        batch=st.integers(1, 50),
        nom1=st.integers(1, 20),
        nom2=st.integers(1, 20),
        num_heads=st.integers(1, 5),
        model_features=st.integers(5, 20),
        key_features=st.integers(5, 100),
        value_features=st.integers(5, 100),
        use_mask=st.booleans(),
    )
    def test_shape(self, batch: int, nom1: int, nom2: int, num_heads: int, model_features: int,
                   key_features: int, value_features: int, use_mask: bool):
        out_features = model_features * num_heads
        attention = MultiHeadAttention(
            num_heads=num_heads, key_features=key_features,
            value_features=value_features, out_features=out_features)

        Q = torch.rand(batch, nom1, key_features)
        K = torch.rand(batch, nom2, key_features)
        V = torch.rand(batch, nom2, value_features)

        if use_mask:
            mask = rand_mask(batch, nom2, nom2, batch_first=True)
        else:
            mask = None

        answer = attention(Q, K, V, mask)
        self.assertIs(answer.dtype, torch.float32)
        self.assertEqual(answer.size(), (batch, nom1, out_features))
