import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_backbone import BaseBackbone


def init_conv_or_linear(layer: nn.Module) -> None:
    """Initialize trainable projections following the original PANNs CNN6."""
    nn.init.xavier_uniform_(layer.weight)
    if getattr(layer, "bias", None) is not None:
        nn.init.zeros_(layer.bias)


def init_bn(layer: nn.modules.batchnorm._BatchNorm) -> None:
    nn.init.ones_(layer.weight)
    nn.init.zeros_(layer.bias)


class ECAAttention(nn.Module):
    """Efficient Channel Attention with only k trainable parameters."""

    def __init__(
        self,
        channels: int,
        gamma: float = 2.0,
        bias: float = 1.0,
        kernel_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if kernel_size is None:
            kernel_size = int(abs((math.log2(channels) + bias) / gamma))
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            kernel_size = max(kernel_size, 3)
        elif kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        # Keep this attribute name unchanged: it is part of the checkpoint keys.
        self.channel_conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        init_conv_or_linear(self.channel_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.adaptive_avg_pool2d(x, output_size=1)
        weights = weights.squeeze(-1).transpose(1, 2)
        weights = self.channel_conv(weights)
        weights = torch.sigmoid(weights.transpose(1, 2).unsqueeze(-1))
        return x * weights


def panns_local_pool(x: torch.Tensor, pool_size: Tuple[int, int]) -> torch.Tensor:
    """Average pooling used after every convolutional block in PANNs CNN6."""
    return F.avg_pool2d(x, kernel_size=pool_size, stride=pool_size)


def panns_global_pool(x: torch.Tensor) -> torch.Tensor:
    """PANNs clip pooling: frequency mean followed by temporal max + mean."""
    x = torch.mean(x, dim=3)
    temporal_max = torch.max(x, dim=2).values
    temporal_mean = torch.mean(x, dim=2)
    return temporal_max + temporal_mean


class StemBlock5x5ECA(nn.Module):
    """A small full-convolution stem that avoids a single-channel DW bottleneck."""

    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            1, out_channels, kernel_size=5, stride=1, padding=2, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.eca = ECAAttention(out_channels)

        init_conv_or_linear(self.conv)
        init_bn(self.bn)

    def forward(
        self, x: torch.Tensor, pool_size: Tuple[int, int] = (2, 2)
    ) -> torch.Tensor:
        x = F.silu(self.bn(self.conv(x)), inplace=True)
        x = self.eca(x)
        return panns_local_pool(x, pool_size)


class DepthwiseECAConvBlock5x5(nn.Module):
    """5x5 depthwise + 1x1 pointwise block with lightweight ECA attention."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=in_channels,
            bias=False,
        )
        self.depthwise_bn = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, bias=False
        )
        self.pointwise_bn = nn.BatchNorm2d(out_channels)
        self.eca = ECAAttention(out_channels)

        init_conv_or_linear(self.depthwise)
        init_conv_or_linear(self.pointwise)
        init_bn(self.depthwise_bn)
        init_bn(self.pointwise_bn)

    def forward(
        self, x: torch.Tensor, pool_size: Tuple[int, int] = (2, 2)
    ) -> torch.Tensor:
        x = F.silu(self.depthwise_bn(self.depthwise(x)), inplace=True)
        x = F.silu(self.pointwise_bn(self.pointwise(x)), inplace=True)
        x = self.eca(x)
        return panns_local_pool(x, pool_size)


class PANNS_Cnn6_DW_ECA(BaseBackbone):
    """Parameter-efficient PANNs CNN6 using depthwise convolution and ECA."""

    def __init__(
        self,
        classes_num: int = 4,
        channels: Sequence[int] = (32, 64, 128, 256),
        head_channels: int = 128,
        block_dropout: float = 0.2,
        head_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if len(channels) != 4 or any(channel <= 0 for channel in channels):
            raise ValueError("channels must contain four positive integers")
        if classes_num <= 0 or head_channels <= 0:
            raise ValueError("classes_num and head_channels must be positive")
        if not 0.0 <= block_dropout < 1.0 or not 0.0 <= head_dropout < 1.0:
            raise ValueError("dropout probabilities must be in [0, 1)")

        c1, c2, c3, c4 = channels
        self.model_name = "panns_cnn6_dw_eca"
        self.block_dropout = block_dropout
        self.head_dropout = head_dropout

        self.conv_block1 = StemBlock5x5ECA(out_channels=c1)
        self.conv_block2 = DepthwiseECAConvBlock5x5(c1, c2)
        self.conv_block3 = DepthwiseECAConvBlock5x5(c2, c3)
        self.conv_block4 = DepthwiseECAConvBlock5x5(c3, c4)

        self.fc1 = nn.Linear(c4, head_channels, bias=True)
        self.fc_audioset = nn.Linear(head_channels, classes_num, bias=True)
        init_conv_or_linear(self.fc1)
        init_conv_or_linear(self.fc_audioset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in (
            self.conv_block1,
            self.conv_block2,
            self.conv_block3,
            self.conv_block4,
        ):
            x = block(x, pool_size=(2, 2))
            x = F.dropout(x, p=self.block_dropout, training=self.training)

        x = panns_global_pool(x)
        x = F.dropout(x, p=self.head_dropout, training=self.training)
        x = F.silu(self.fc1(x), inplace=True)
        x = F.dropout(x, p=self.head_dropout, training=self.training)
        return self.fc_audioset(x)
