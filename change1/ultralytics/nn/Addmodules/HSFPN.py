import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CA', 'multiply', 'Add']


class Add(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, ch=256):
        super().__init__()

    def forward(self, x):
        input1, input2 = x[0], x[1]
        x = input1 + input2
        return x


class multiply(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x[0] * x[1]
        return x


class CA(nn.Module):
    def __init__(self, in_planes, ratio=4, flag=True):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.flag = flag
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, x):
        avg_out = self.conv2(self.relu(self.conv1(self.avg_pool(x))))
        max_out = self.conv2(self.relu(self.conv1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out) * x if self.flag else self.sigmoid(out)
        return out


class FeatureSelectionModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(FeatureSelectionModule, self).__init__()
        self.conv_atten = nn.Conv2d(in_chan, in_chan, kernel_size=1)
        self.group_norm1 = nn.GroupNorm(32, in_chan)
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=1)
        self.group_norm2 = nn.GroupNorm(32, out_chan)
        nn.init.xavier_uniform_(self.conv_atten.weight)
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        atten = self.sigmoid(self.group_norm1(self.conv_atten(F.avg_pool2d(x, x.size()[2:]))))
        feat = torch.mul(x, atten)
        x = x + feat
        feat = self.group_norm2(self.conv(x))
        return feat


if __name__ == "__main__":
    # Generating Sample image
    image_size = (1, 64, 240, 240)
    image = torch.rand(*image_size)

    # Model
    mobilenet_v3 = FeatureSelectionModule(64, 64)

    out = mobilenet_v3(image)
    print(out.size())