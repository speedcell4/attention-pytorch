import torch
from torch import nn

__all__ = [
    'Attention', 'FacetAttention', 'MultiHeadAttention',
]


class Attention(nn.Module):
    def forward(self, Q: torch.FloatTensor, K: torch.FloatTensor, V: torch.FloatTensor,
                mask: torch.ByteTensor = None) -> torch.FloatTensor:
        """

        Args:
            Q: (*batch, nom1, vector1)
            K: (*batch, nom2, vector1)
            V: (*batch, nom2, vector2)
            mask: (*batch, nom2)
        Returns:
            (*batch, nom1, vector2)
        """
        raise NotImplementedError


from attentions.multi_head import DotProductAttention, MultiHeadAttention
from attentions.facet import FacetAttention
