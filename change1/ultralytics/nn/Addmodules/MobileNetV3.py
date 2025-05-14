"""A from-scratch implementation of MobileNetV3 paper ( for educational purposes ).
Paper
    Searching for MobileNetV3 - https://arxiv.org/abs/1905.02244v5
author : shubham.aiengineer@gmail.com
"""

import torch
from torch import nn
from torchsummary import summary

__all__ = ['MobileNetV3']

class SqueezeExitationBlock(nn.Module):
    def __init__(self, in_channels: int):
        """Constructor for SqueezeExitationBlock.
        Args:
            in_channels (int): Number of input channels.
        """
        super().__init__()

        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Linear(
            in_channels, in_channels // 4
        )  # divide by 4 is mentioned in the paper, 5.3. Large squeeze-and-excite
        self.act1 = nn.ReLU()
        self.linear2 = nn.Linear(in_channels // 4, in_channels)
        self.act2 = nn.Hardsigmoid()

    def forward(self, x):
        """Forward pass for SqueezeExitationBlock."""

        identity = x

        x = self.pool1(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act2(x)

        x = identity * x[:, :, None, None]

        return x


class ConvNormActivationBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: list,
            stride: int = 1,
            padding: int = 0,
            groups: int = 1,
            bias: bool = False,
            activation: torch.nn = nn.Hardswish,
    ):
        """Constructs a block containing a convolution, batch normalization and activation layer
        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            kernel_size (list): size of the convolutional kernel
            stride (int, optional): stride of the convolutional kernel. Defaults to 1.
            padding (int, optional): padding of the convolutional kernel. Defaults to 0.
            groups (int, optional): number of groups for depthwise seperable convolution. Defaults to 1.
            bias (bool, optional): whether to use bias. Defaults to False.
            activation (torch.nn, optional): activation function. Defaults to nn.Hardswish.
        """
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = activation()

    def forward(self, x):
        """Perform forward pass."""

        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)

        return x


class InverseResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            expansion_size: int = 6,
            stride: int = 1,
            squeeze_exitation: bool = True,
            activation: nn.Module = nn.Hardswish,
    ):

        """Constructs a inverse residual block
        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            kernel_size (int): size of the convolutional kernel
            expansion_size (int, optional): size of the expansion factor. Defaults to 6.
            stride (int, optional): stride of the convolutional kernel. Defaults to 1.
            squeeze_exitation (bool, optional): whether to add squeeze and exitation block or not. Defaults to True.
            activation (nn.Module, optional): activation function. Defaults to nn.Hardswish.
        """

        super().__init__()

        self.residual = in_channels == out_channels and stride == 1
        self.squeeze_exitation = squeeze_exitation

        self.conv1 = (
            ConvNormActivationBlock(
                in_channels, expansion_size, (1, 1), activation=activation
            )
            if in_channels != expansion_size
            else nn.Identity()
        )  # If it's not the first layer, then we need to add a 1x1 convolutional layer to expand the number of channels
        self.depthwise_conv = ConvNormActivationBlock(
            expansion_size,
            expansion_size,
            (kernel_size, kernel_size),
            stride=stride,
            padding=kernel_size // 2,
            groups=expansion_size,
            activation=activation,
        )
        if self.squeeze_exitation:
            self.se = SqueezeExitationBlock(expansion_size)

        self.conv2 = nn.Conv2d(
            expansion_size, out_channels, (1, 1), bias=False
        )  # bias is false because we are using batch normalization, which already has bias
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """Perform forward pass."""

        identity = x

        x = self.conv1(x)
        x = self.depthwise_conv(x)

        if self.squeeze_exitation:
            x = self.se(x)

        x = self.conv2(x)
        x = self.norm(x)

        if self.residual:
            x = x + identity

        return x


class MobileNetV3(nn.Module):
    def __init__(
            self,
            n_classes: int = 1000,
            input_channel: int = 3,
            config: str = "large",
            dropout: float = 0.8,
    ):
        """Constructs MobileNetV3 architecture
        Args:
        `n_classes`: An integer count of output neuron in last layer, default 1000
        `input_channel`: An integer value input channels in first conv layer, default is 3.
        `config`: A string value indicating the configuration of MobileNetV3, either `large` or `small`, default is `large`.
        `dropout` [0, 1] : A float parameter for dropout in last layer, between 0 and 1, default is 0.8.
        """

        super().__init__()

        # The configuration of MobileNetv3.
        # input channels, kernel size, expension size, output channels, squeeze exitation, activation, stride
        RE = nn.ReLU
        HS = nn.Hardswish
        configs_dict = {
            "small": (
                (16, 3, 16, 16, True, RE, 2),
                (16, 3, 72, 24, False, RE, 2),
                (24, 3, 88, 24, False, RE, 1),
                (24, 5, 96, 40, True, HS, 2),
                (40, 5, 240, 40, True, HS, 1),
                (40, 5, 240, 40, True, HS, 1),
                (40, 5, 120, 48, True, HS, 1),
                (48, 5, 144, 48, True, HS, 1),
                (48, 5, 288, 96, True, HS, 2),
                (96, 5, 576, 96, True, HS, 1),
                (96, 5, 576, 96, True, HS, 1),
            ),
            "large": (
                (16, 3, 16, 16, False, RE, 1),
                (16, 3, 64, 24, False, RE, 2),
                (24, 3, 72, 24, False, RE, 1),
                (24, 5, 72, 40, True, RE, 2),
                (40, 5, 120, 40, True, RE, 1),
                (40, 5, 120, 40, True, RE, 1),
                (40, 3, 240, 80, False, HS, 2),
                (80, 3, 200, 80, False, HS, 1),
                (80, 3, 184, 80, False, HS, 1),
                (80, 3, 184, 80, False, HS, 1),
                (80, 3, 480, 112, True, HS, 1),
                (112, 3, 672, 112, True, HS, 1),
                (112, 5, 672, 160, True, HS, 2),
                (160, 5, 960, 160, True, HS, 1),
                (160, 5, 960, 160, True, HS, 1),
            ),
        }

        self.model = nn.Sequential(
            ConvNormActivationBlock(
                input_channel, 16, (3, 3), stride=2, padding=1, activation=nn.Hardswish
            ),
        )

        for (
                in_channels,
                kernel_size,
                expansion_size,
                out_channels,
                squeeze_exitation,
                activation,
                stride,
        ) in configs_dict[config]:
            self.model.append(
                InverseResidualBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    expansion_size=expansion_size,
                    stride=stride,
                    squeeze_exitation=squeeze_exitation,
                    activation=activation,
                )
            )

        hidden_channels = 576 if config == "small" else 960
        _out_channel = 1024 if config == "small" else 1280

        self.model.append(
            ConvNormActivationBlock(
                out_channels,
                hidden_channels,
                (1, 1),
                bias=False,
                activation=nn.Hardswish,
            )
        )
        if config == 'small':
            self.index = [16, 24, 48, 576]
        else:
            self.index = [24, 40, 112, 960]
        self.width_list = [i.size(1) for i in self.forward(torch.randn(1, 3, 640, 640))]

    def forward(self, x):
        """Perform forward pass."""
        results = [None, None, None, None]

        for model in self.model:
            x = model(x)
            if x.size(1) in self.index:
                position = self.index.index(x.size(1))  # Find the position in the index list
                results[position] = x
            # results.append(x)
        return results


if __name__ == "__main__":
    # Generating Sample image
    image_size = (1, 3, 640, 640)
    image = torch.rand(*image_size)

    # Model
    mobilenet_v3 = MobileNetV3(config="large")

    # summary(
    #     mobilenet_v3,
    #     input_data=image,
    #     col_names=["input_size", "output_size", "num_params"],
    #     device="cpu",
    #     depth=2,
    # )

    out = mobilenet_v3(image)
    print(out)