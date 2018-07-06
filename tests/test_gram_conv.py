import unittest

import torch
from hypothesis import given, strategies as st

from conv import GramConv1d
from testing import rand_mask


class TestGramConv1d(unittest.TestCase):
    @given(
        batch=st.integers(1, 20),
        in_features=st.integers(5, 100),
        out_features=st.integers(5, 100),
        num_grams=st.sampled_from([1, 3, 5, 7, 9]),
        times=st.integers(1, 120),
        use_mask=st.booleans(),
    )
    def test_shape(self, batch: int, in_features: int, out_features: int, num_grams: int, times: int, use_mask: bool):
        convolution = GramConv1d(in_features, out_features, num_grams=num_grams)
        if use_mask:
            mask = rand_mask(batch=batch, times=times, total_length=times, batch_first=True)
        else:
            mask = None
        x = torch.rand(batch, in_features, times)
        self.assertEqual(convolution(x, mask=mask).size(), (batch, out_features, times))
