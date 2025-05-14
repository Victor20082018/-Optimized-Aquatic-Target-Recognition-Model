import torch
import torch.nn as nn

__all__ = ['CSPHet']


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class HetConv(nn.Module):

    def __init__(self, input_channels, output_channels, stride=1, p=4):
        """
        Initialize the HetConv class.
        :param input_channels: the number of input channels
        :param output_channels: the number of output channels
        :param stride: convolution stride
        :param p: the value of P used in HetConv
        """
        super(HetConv, self).__init__()
        self.p = p
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.filters = nn.ModuleList()
        self.convolution_1x1_index = []
        # Compute the indices of input channels fed to 1x1 convolutional kernels in all filters.
        # These indices of input channels are also the indices of the 1x1 convolutional kernels in the filters.
        # This is only executed when the HetConv class is created,
        # and the execution time is not included during inference.
        for i in range(self.p):
            self.convolution_1x1_index.append(self.compute_convolution_1x1_index(i))
        # Build HetConv filters.
        for i in range(self.p):
            self.filters.append(self.build_HetConv_filters(stride, p))

    def compute_convolution_1x1_index(self, i):
        """
        Compute the indices of input channels fed to 1x1 convolutional kernels in the i-th branch of filters (i=0, 1, 2,…, P-1). The i-th branch of filters consists of the {i, i+P, i+2P,…, i+N-P}-th filters.
        :param i: the i-th branch of filters in HetConv
        :return: return the required indices of input channels
        """
        index = [j for j in range(0, self.input_channels)]
        # Remove the indices of input channels fed to 3x3 convolutional kernels in the i-th branch of filters.
        while i < self.input_channels:
            index.remove(i)
            i += self.p
        return index

    def build_HetConv_filters(self, stride, p):
        """
        Build N/P filters in HetConv.
        :param stride: convolution stride
        :param p: the value of P used in HetConv
        :return: return N/P HetConv filters
        """
        temp_filters = nn.ModuleList()
        # nn.Conv2d arguments: nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding)
        temp_filters.append(nn.Conv2d(self.input_channels // p, self.output_channels // p, 3, stride, 1, bias=False))
        temp_filters.append(
            nn.Conv2d(self.input_channels - self.input_channels // p, self.output_channels // p, 1, stride, 0,
                      bias=False))
        return temp_filters

    def forward(self, input_data):
        """
        Define how HetConv processes the input images or input feature maps.
        :param input_data: input images or input feature maps
        :return: return output feature maps
        """
        output_feature_maps = []
        # Loop P times to get output feature maps. The number of output feature maps = the batch size.
        for i in range(0, self.p):
            # M/P HetConv filter kernels perform the 3x3 convolution and output to N/P output channels.
            output_feature_3x3 = self.filters[i][0](input_data[:, i::self.p, :, :])
            # (M-M/P) HetConv filter kernels perform the 1x1 convolution and output to N/P output channels.
            output_feature_1x1 = self.filters[i][1](input_data[:, self.convolution_1x1_index[i], :, :])

            # Obtain N/P output feature map channels.
            output_feature_map = output_feature_1x1 + output_feature_3x3

            # Append N/P output feature map channels.
            output_feature_maps.append(output_feature_map)

        # Get the batch size, number of output channels (N/P), height and width of output feature map.
        N, C, H, W = output_feature_maps[0].size()
        # Change the value of C to the number of output feature map channels (N).
        C = self.p * C
        # Arrange the output feature map channels to make them fit into the shifted manner.
        return torch.cat(output_feature_maps, 1).view(N, self.p, C // self.p, H, W).permute(0, 2, 1, 3,
                                                                                            4).contiguous().view(N, C,
                                                                                                                 H, W)


class CSPHet_Bottleneck(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.DualPConv = nn.Sequential(HetConv(dim, dim), HetConv(dim, dim))

    def forward(self, x):
        return self.DualPConv(x)


class CSPHet(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(CSPHet_Bottleneck(self.c) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


if __name__ == "__main__":
    # Generating Sample image
    image_size = (1, 64, 224, 224)
    image = torch.rand(*image_size)

    # Model
    model = CSPHet(64, 128)

    out = model(image)
    print(out.size())