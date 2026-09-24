from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_backbone import BaseBackbone
from .panns_cnn6_dw_eca import ECAAttention, init_bn, init_conv_or_linear


class DepthwiseECABlock(nn.Module):
    """Depthwise-separable time-frequency convolution followed by ECA."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        eca_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_channels,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.eca = ECAAttention(out_channels, kernel_size=eca_kernel_size)

        init_conv_or_linear(self.depthwise)
        init_conv_or_linear(self.pointwise)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(
        self, x: torch.Tensor, pool_size: Optional[Tuple[int, int]] = (2, 2)
    ) -> torch.Tensor:
        x = F.silu(self.bn1(self.depthwise(x)), inplace=True)
        x = F.silu(self.bn2(self.pointwise(x)), inplace=True)
        x = self.eca(x)
        if pool_size is not None:
            x = F.avg_pool2d(x, kernel_size=pool_size, stride=pool_size)
        return x


class TinyPANNS_ECA(BaseBackbone):
    """Accuracy-oriented lightweight PANNs backbone with depthwise ECA blocks."""

    def __init__(
        self,
        classes_num: int = 4,
        channels: Sequence[int] = (32, 64, 128, 256, 384, 512),
        head_channels: int = 256,
        depthwise_kernel_size: int = 3,
        eca_kernel_size: int = 3,
        block_dropouts: Sequence[float] = (0.10, 0.10, 0.15, 0.15),
        head_dropout: float = 0.20,
    ) -> None:
        super().__init__()
        if len(channels) != 6 or any(channel <= 0 for channel in channels):
            raise ValueError("channels must contain six positive integers")
        if len(block_dropouts) != 4:
            raise ValueError("block_dropouts must contain four values")
        if any(not 0.0 <= dropout < 1.0 for dropout in block_dropouts):
            raise ValueError("block dropout probabilities must be in [0, 1)")
        if not 0.0 <= head_dropout < 1.0:
            raise ValueError("head_dropout must be in [0, 1)")
        if classes_num <= 0 or head_channels <= 0:
            raise ValueError("classes_num and head_channels must be positive")

        stem_channels, c1, c2, c3, c4, c5 = channels
        self.model_name = "tiny_panns_eca"
        self.block_dropouts = tuple(block_dropouts)
        self.head_dropout = head_dropout

        self.stem_conv = nn.Conv2d(
            1, stem_channels, kernel_size=5, padding=2, bias=False
        )
        self.stem_bn = nn.BatchNorm2d(stem_channels)
        self.stem_activation = nn.SiLU(inplace=True)

        block_args = {
            "kernel_size": depthwise_kernel_size,
            "eca_kernel_size": eca_kernel_size,
        }
        self.block1 = DepthwiseECABlock(stem_channels, c1, **block_args)
        self.block2 = DepthwiseECABlock(c1, c2, **block_args)
        self.block3 = DepthwiseECABlock(c2, c3, **block_args)
        self.block4 = DepthwiseECABlock(c3, c4, **block_args)
        self.block5 = DepthwiseECABlock(c4, c5, **block_args)

        self.fc1 = nn.Linear(c5, head_channels)
        self.fc_audioset = nn.Linear(head_channels, classes_num)

        init_conv_or_linear(self.stem_conv)
        init_bn(self.stem_bn)
        init_conv_or_linear(self.fc1)
        init_conv_or_linear(self.fc_audioset)

    @staticmethod
    def _panns_global_pool(x: torch.Tensor) -> torch.Tensor:
        x = torch.mean(x, dim=3)
        return torch.max(x, dim=2).values + torch.mean(x, dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem_activation(self.stem_bn(self.stem_conv(x)))

        for block, dropout in zip(
            (self.block1, self.block2, self.block3, self.block4),
            self.block_dropouts,
        ):
            x = block(x, pool_size=(2, 2))
            x = F.dropout(x, p=dropout, training=self.training)

        # Preserve temporal detail at the final, highest-level feature stage.
        x = self.block5(x, pool_size=None)
        x = self._panns_global_pool(x)

        x = F.dropout(x, p=self.head_dropout, training=self.training)
        x = F.silu(self.fc1(x), inplace=True)
        x = F.dropout(x, p=self.head_dropout, training=self.training)
        return self.fc_audioset(x)
