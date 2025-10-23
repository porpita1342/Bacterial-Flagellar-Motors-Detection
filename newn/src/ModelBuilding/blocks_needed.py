from monai.networks.blocks import Convolution, UpSample
from enum import Enum
from typing import Union

class StrEnum(str, Enum):
    """
    Enum subclass that converts its value to a string.

    .. code-block:: python

        from monai.utils import StrEnum

        class Example(StrEnum):
            MODE_A = "A"
            MODE_B = "B"

        assert (list(Example) == ["A", "B"])
        assert Example.MODE_A == "A"
        assert str(Example.MODE_A) == "A"
        assert monai.utils.look_up_option("A", Example) == "A"
    """

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value
class UpsampleMode(StrEnum):
    """
    See also: :py:class:`monai.networks.blocks.UpSample`
    """
    DECONV = "deconv"
    DECONVGROUP = "deconvgroup"
    NONTRAINABLE = "nontrainable"  # e.g. using torch.nn.Upsample
    PIXELSHUFFLE = "pixelshuffle"


class InterpolateMode(StrEnum):
    """
    See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
    """

    NEAREST = "nearest"
    NEAREST_EXACT = "nearest-exact"
    LINEAR = "linear"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    TRILINEAR = "trilinear"
    AREA = "area"
def get_conv_layer(
    spatial_dims: int, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, bias: bool = False
):

    return Convolution(
        spatial_dims,
        in_channels,
        out_channels,
        strides=stride,
        kernel_size=kernel_size,
        bias=bias,
        conv_only=True,
    )


def get_upsample_layer(
    spatial_dims: int, in_channels: int, upsample_mode: Union[UpsampleMode, str] = "nontrainable", scale_factor: int = 2
):
    return UpSample(
        dimensions=spatial_dims,
        in_channels=in_channels,
        out_channels=in_channels,
        scale_factor=scale_factor,
        mode=upsample_mode,
        interp_mode=InterpolateMode.LINEAR,
        align_corners=False,
    )