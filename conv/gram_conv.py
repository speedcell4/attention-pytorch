import torch
from torch import nn

from houttuynia.nn.init import keras_conv_


class GramConv1d(nn.Sequential):
    """
    Args:
        inputs: (batch, in_features, times)
        mask: (batch, times)
    Returns:
        (batch, out_features, times)
    """

    def __init__(self, in_features: int, out_features: int, num_grams: int,
                 negative_slop: float = 0, bias: bool = False) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.num_grams = num_grams
        self.negative_slop = negative_slop
        self.bias = bias

        _hidden = max(in_features, out_features)
        super(GramConv1d, self).__init__(
            nn.Conv1d(in_features, _hidden, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.LeakyReLU(negative_slope=negative_slop, inplace=True),
            nn.Conv1d(_hidden, _hidden, kernel_size=num_grams, stride=1, padding=num_grams // 2, bias=bias),
            nn.LeakyReLU(negative_slope=negative_slop, inplace=True),
            nn.Conv1d(_hidden, out_features, kernel_size=1, stride=1, padding=0, bias=bias),
        )

        self.reset_parameters()

    def reset_parameters(self):
        keras_conv_(self[0])
        keras_conv_(self[2])
        keras_conv_(self[4])

    def forward(self, inputs: torch.FloatTensor, mask: torch.ByteTensor = None) -> torch.FloatTensor:
        if mask is not None:
            mask = ~mask.unsqueeze(1)
            inputs = inputs.masked_fill_(mask, 0)
        return super(GramConv1d, self).forward(inputs)
