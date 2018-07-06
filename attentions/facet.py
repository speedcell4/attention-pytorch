import torch
from torch.nn import functional as F
from torch import nn

from attentions import Attention
from utils import expanded_masked_fill


class FacetAttention(Attention):
    def __init__(self, in_features: int, negative_slope: float = 0, bias: int = False) -> None:
        super(FacetAttention, self).__init__()

        self.in_features = in_features
        self.negative_slope = negative_slope

        self.fc = nn.Sequential(
            nn.Linear(in_features * self.num_facts, in_features * self.num_facts, bias=bias),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Linear(in_features * self.num_facts, 1, bias=bias),
        )

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
        *batch, nom1, vector1 = Q.size()
        *batch, nom2, vector2 = V.size()

        Q = Q.view(*batch, nom1, 1, vector1).expand(*batch, nom1, nom2, vector1)
        K = K.view(*batch, 1, nom2, vector1).expand(*batch, nom1, nom2, vector1)
        A = self.fc(self.facets(Q, K)).squeeze(-1)  # (*batch, nom1, nom2)
        if mask is not None:
            A = expanded_masked_fill(A, ~mask, filling_value=-float('inf'))
        return F.softmax(A, dim=-1) @ V

    num_facts: int = 4

    @classmethod
    def facets(cls, Q: torch.FloatTensor, K: torch.FloatTensor) -> torch.FloatTensor:
        return torch.cat([Q, K, torch.abs(Q - K), Q * K], dim=-1)
