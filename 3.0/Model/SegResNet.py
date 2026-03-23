import torch.nn as nn
import torch
from typing import List, Tuple
from torch import Tensor
import torch.nn.functional as F
from monai.networks.blocks import ResBlock
from monai.networks.blocks.segresnet_block import get_conv_layer, get_upsample_layer
from monai.networks.layers import get_norm_layer, get_act_layer, Dropout
from monai.utils import UpsampleMode


class SegResNetBackbone(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        init_filters: int = 8,
        in_channels: int = 1,
        out_channels: int = 2,
        dropout_prob: float | None = None,
        act: tuple | str = ("RELU", {"inplace": True}),
        norm: tuple | str = ("GROUP", {"num_groups": 8}),
        norm_name: str = "",
        num_groups: int = 8,
        use_conv_final: bool = True,
        blocks_down: tuple = (1, 2, 2, 4),
        blocks_up: tuple = (1, 1, 1),
        upsample_mode: UpsampleMode | str = UpsampleMode.NONTRAINABLE,
    ):
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("`spatial_dims` can only be 2 or 3.")

        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.in_channels = in_channels
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.dropout_prob = dropout_prob
        self.act = act
        self.act_mod = get_act_layer(act)
        if norm_name:
            if norm_name.lower() != "group":
                raise ValueError(f"Deprecating option 'norm_name={norm_name}', please use 'norm' instead.")
            norm = ("group", {"num_groups": num_groups})
        self.norm = norm
        self.upsample_mode = UpsampleMode(upsample_mode)
        self.use_conv_final = use_conv_final
        self.convInit = get_conv_layer(spatial_dims, in_channels, init_filters)
        self.down_layers = self._make_down_layers()
        self.up_layers, self.up_samples = self._make_up_layers()
        self.conv_final = self._make_final_conv(out_channels, in_channels=self.init_filters)
        self.conv_penultimate = self._make_final_conv(out_channels, in_channels=self.init_filters * 2)

        if dropout_prob is not None:
            self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)

    def _make_down_layers(self):
        down_layers = nn.ModuleList()
        blocks_down, spatial_dims, filters, norm = (
            self.blocks_down, self.spatial_dims, self.init_filters, self.norm)

        for i, item in enumerate(blocks_down):
            layer_in_channels = filters * 2 ** i
            pre_conv = (
                get_conv_layer(spatial_dims=spatial_dims, in_channels=layer_in_channels // 2,
                               out_channels=layer_in_channels, stride=2, bias=False)
                if i > 0 else nn.Identity()
            )
            down_layer = nn.Sequential(
                pre_conv,
                *[ResBlock(spatial_dims=spatial_dims, in_channels=layer_in_channels,
                           norm=norm, act=self.act) for _ in range(item)]
            )
            down_layers.append(down_layer)
        return down_layers

    def _make_up_layers(self):
        up_layers, up_samples = nn.ModuleList(), nn.ModuleList()
        upsample_mode, blocks_up, spatial_dims, filters, norm = (
            self.upsample_mode, self.blocks_up, self.spatial_dims, self.init_filters, self.norm)

        n_up = len(blocks_up)
        for i in range(n_up):
            sample_in_channels = filters * 2 ** (n_up - i)
            up_layers.append(
                nn.Sequential(*[
                    ResBlock(spatial_dims=spatial_dims, in_channels=sample_in_channels // 2,
                             norm=norm, act=self.act)
                    for _ in range(blocks_up[i])
                ])
            )
            up_samples.append(
                nn.Sequential(*[
                    get_conv_layer(spatial_dims=spatial_dims, in_channels=sample_in_channels,
                                   out_channels=sample_in_channels // 2, kernel_size=1, bias=False),
                    get_upsample_layer(spatial_dims=spatial_dims, in_channels=sample_in_channels // 2,
                                       upsample_mode=upsample_mode, scale_factor=2),
                ])
            )
        return up_layers, up_samples

    def _make_final_conv(self, out_channels: int, in_channels: int):
        return nn.Sequential(
            get_conv_layer(spatial_dims=self.spatial_dims, in_channels=in_channels,
                           out_channels=out_channels, kernel_size=1, bias=True),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.convInit(x)
        if self.dropout_prob is not None:
            x = self.dropout(x)
        down_x = []
        for down in self.down_layers:
            x = down(x)
            down_x.append(x)
        return x, down_x

    def decode(self, x: Tensor, down_x: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        feature_maps = []
        for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
            x = up(x) + down_x[i + 1]
            x = upl(x)
            feature_maps.append(x)

        if self.use_conv_final:
            processed_feature_maps = [
                self.conv_penultimate(feature_maps[-2]),
                self.conv_final(feature_maps[-1])
            ]
        else:
            processed_feature_maps = feature_maps[-2:]

        return feature_maps, processed_feature_maps

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x, down_x = self.encode(x)
        down_x.reverse()
        _, processed_feature_maps = self.decode(x, down_x)
        return processed_feature_maps[-1], processed_feature_maps[-2]  # final, penultimate


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])
