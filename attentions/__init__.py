import torch
from torch import nn

__all__ = [
    'Attention', 'expanded_masked_fill',
    'FacetAttention', 'MultiHeadAttention',
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


def expanded_masked_fill(tensor: torch.Tensor, mask: torch.ByteTensor,
                         filling_value: float = -float('inf')) -> torch.Tensor:
    """

    Args:
        tensor: (*batch, *nom1, nom2)
        mask: (*batch, nom2)
        filling_value: (,)

    Returns:
        (*batch, *nom1, nom2)
    """
    *batch, dim = mask.size()
    mask = mask.view(*batch, *(1,) * (tensor.dim() - mask.dim()), dim).expand_as(tensor)
    return tensor.masked_fill(mask, filling_value)


from attentions.multi_head import DotProductAttention, MultiHeadAttention
from attentions.facet import FacetAttention
