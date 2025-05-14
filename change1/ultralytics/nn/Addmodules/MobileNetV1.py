"""A from-scratch implementation of original MobileNet paper ( for educational purposes ).
Paper
    MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications - https://arxiv.org/abs/1704.04861
author : shubham.aiengineer@gmail.com
"""
import torch
from torch import nn

__all__ = ['MobileNetV1']

class DepthwiseSepConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            use_relu6: bool = True,
    ):
        """Constructs Depthwise seperable with pointwise convolution with relu and batchnorm respectively.
        Args:
            in_channels (int): input channels for depthwise convolution
            out_channels (int): output channels for pointwise convolution
            stride (int, optional): stride paramemeter for depthwise convolution. Defaults to 1.
            use_relu6 (bool, optional): whether to use standard ReLU or ReLU6 for depthwise separable convolution block. Defaults to True.
        """

        super().__init__()

        # Depthwise conv
        self.depthwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            (3, 3),
            stride=stride,
            padding=1,
            groups=in_channels,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.relu1 = nn.ReLU6() if use_relu6 else nn.ReLU()

        # Pointwise conv
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, (1, 1))
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu2 = nn.ReLU6() if use_relu6 else nn.ReLU()

    def forward(self, x):
        """Perform forward pass."""

        x = self.depthwise_conv(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pointwise_conv(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x


class MobileNetV1(nn.Module):
    def __init__(
            self,
            input_channel: int = 3,
            depth_multiplier: float = 1.0,
            use_relu6: bool = True,
    ):
        """Constructs MobileNetV1 architecture
        Args:
            n_classes (int, optional): count of output neuron in last layer. Defaults to 1000.
            input_channel (int, optional): input channels in first conv layer. Defaults to 3.
            depth_multiplier (float, optional): network width multiplier ( width scaling ). Suggested Values - 0.25, 0.5, 0.75, 1.. Defaults to 1.0.
            use_relu6 (bool, optional): whether to use standard ReLU or ReLU6 for depthwise separable convolution block. Defaults to True.
        """

        super().__init__()

        # The configuration of MobileNetV1
        # input channels, output channels, stride
        config = (
            (32, 64, 1),
            (64, 128, 2),
            (128, 128, 1),
            (128, 256, 2),
            (256, 256, 1),
            (256, 512, 2),
            (512, 512, 1),
            (512, 512, 1),
            (512, 512, 1),
            (512, 512, 1),
            (512, 512, 1),
            (512, 1024, 2),
            (1024, 1024, 1),
        )

        self.model = nn.Sequential(
            nn.Conv2d(
                input_channel, int(32 * depth_multiplier), (3, 3), stride=2, padding=1
            )
        )

        # Adding depthwise block in the model from the config
        for in_channels, out_channels, stride in config:
            self.model.append(
                DepthwiseSepConvBlock(
                    int(in_channels * depth_multiplier),  # input channels
                    int(out_channels * depth_multiplier),  # output channels
                    stride,
                    use_relu6=use_relu6,
                )
            )
        self.index = [128, 256, 512, 1024]
        self.width_list = [i.size(1) for i in self.forward(torch.randn(1, 3, 640, 640))]

    def forward(self, x):
        """Perform forward pass."""
        results = [None, None, None, None]

        for model in self.model:
            x = model(x)
            if x.size(1) in self.index:
                position = self.index.index(x.size(1))  # Find the position in the index list
                results[position] = x
        return results


if __name__ == "__main__":
    # Generating Sample image
    image_size = (1, 3, 224, 224)
    image = torch.rand(*image_size)

    # Model
    mobilenet_v1 = MobileNetV1(depth_multiplier=1)

    out = mobilenet_v1(image)
    print(out)