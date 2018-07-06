from typing import Tuple

from more_itertools import islice_extended
import torch

__all__ = [
    'cartesian_view'
]


def cartesian_view(tensor1: torch.Tensor, tensor2: torch.Tensor,
                   start: int, end: int) -> Tuple[torch.Tensor, torch.Tensor]:
    assert tensor1.dim() == tensor2.dim()

    start1, start2 = tensor1.size()[:start], tensor2.size()[:start]
    end1, end2 = tensor1.size()[end:], tensor2.size()[end:]
    mid1, mid2, mid_common = [], [], []

    for dim1, dim2 in islice_extended(zip(tensor1.size(), tensor2.size()), start, end):
        if dim1 != dim2:
            mid1.extend((dim1, 1))
            mid2.extend((1, dim2))
            mid_common.extend((dim1, dim2))
        else:
            mid1.append(dim1)
            mid2.append(dim2)
            mid_common.append(dim1)

    tensor1 = tensor1.view(*start1, *mid1, *end1).expand(*start1, *mid_common, *end1)
    tensor2 = tensor2.view(*start2, *mid2, *end2).expand(*start2, *mid_common, *end2)
    return tensor1, tensor2
