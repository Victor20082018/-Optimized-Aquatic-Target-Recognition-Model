import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SDI']

class SDI(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.convs = nn.ModuleList(
            [nn.Conv2d(c, channel[0], kernel_size=3, stride=1, padding=1) for c in channel])

    def forward(self, xs):
        ans = torch.ones_like(xs[0])
        target_size = xs[0].shape[-2:]

        for i, x in enumerate(xs):
            if x.shape[-1] > target_size[0]:
                x = F.adaptive_avg_pool2d(x, (target_size[0], target_size[1]))
            elif x.shape[-1] < target_size[0]:
                x = F.interpolate(x, size=(target_size[0], target_size[1]),
                                      mode='bilinear', align_corners=True)

            ans = ans * self.convs[i](x)

        return ans