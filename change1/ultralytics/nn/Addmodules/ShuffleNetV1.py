# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import Any, List, Optional

import torch
from torch import Tensor
from torch import nn

__all__ = [
    "shufflenet_v1_x0_5", "shufflenet_v1_x1_0", "shufflenet_v1_x1_5", "shufflenet_v1_x2_0",
]


class ShuffleNetV1(nn.Module):

    def __init__(
            self,
            repeats_times: List[int],
            stages_out_channels: List[int],
            groups: int = 8,
            num_classes: int = 1000,
    ) -> None:
        super(ShuffleNetV1, self).__init__()
        in_channels = stages_out_channels[0]

        self.first_conv = nn.Sequential(
            nn.Conv2d(3, in_channels, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
        )
        self.maxpool = nn.MaxPool2d((3, 3), (2, 2), (1, 1))

        features = []
        for state_repeats_times_index in range(len(repeats_times)):
            out_channels = stages_out_channels[state_repeats_times_index + 1]

            for i in range(repeats_times[state_repeats_times_index]):
                stride = 2 if i == 0 else 1
                first_group = state_repeats_times_index == 0 and i == 0
                features.append(
                    ShuffleNetV1Unit(
                        in_channels,
                        out_channels,
                        stride,
                        groups,
                        first_group,
                    )
                )
                in_channels = out_channels
        self.features = nn.Sequential(*features)

        self.globalpool = nn.AvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(stages_out_channels[-1], num_classes, bias=False),
        )

        # Initialize neural network weights
        self._initialize_weights()
        self.index = stages_out_channels[-3:]
        self.width_list = [i.size(1) for i in self.forward(torch.randn(1, 3, 640, 640))]

    def forward(self, x):
        x = self.first_conv(x)
        x = self.maxpool(x)
        results = [None, None, None, None]
        for index, model in enumerate(self.features):
            x = model(x)
            # results.append(x)
            if index == 0:
                results[index] = x
            if x.size(1) in self.index:
                position = self.index.index(x.size(1))  # Find the position in the index list
                results[position + 1] = x
        return results

    def _initialize_weights(self) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(module.weight, 0, 0.01)
                else:
                    nn.init.normal_(module.weight, 0, 1.0 / module.weight.shape[1])
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0001)
                nn.init.constant_(module.running_mean, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0001)
                nn.init.constant_(module.running_mean, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


class ShuffleNetV1Unit(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
            groups: int,
            first_groups: bool = False,
    ) -> None:
        super(ShuffleNetV1Unit, self).__init__()
        self.stride = stride
        self.groups = groups
        self.first_groups = first_groups
        hidden_channels = out_channels // 4

        if stride == 2:
            out_channels -= in_channels
            self.branch_proj = nn.AvgPool2d((3, 3), (2, 2), (1, 1))

        self.branch_main_1 = nn.Sequential(
            # pw
            nn.Conv2d(in_channels, hidden_channels, (1, 1), (1, 1), (0, 0), groups=1 if first_groups else groups,
                      bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(True),
            # dw
            nn.Conv2d(hidden_channels, hidden_channels, (3, 3), (stride, stride), (1, 1), groups=hidden_channels,
                      bias=False),
            nn.BatchNorm2d(hidden_channels),
        )
        self.branch_main_2 = nn.Sequential(
            # pw-linear
            nn.Conv2d(hidden_channels, out_channels, (1, 1), (1, 1), (0, 0), groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.relu = nn.ReLU(True)

    def channel_shuffle(self, x):
        batch_size, channels, height, width = x.data.size()
        assert channels % self.groups == 0
        group_channels = channels // self.groups

        out = x.reshape(batch_size, group_channels, self.groups, height, width)
        out = out.permute(0, 2, 1, 3, 4)
        out = out.reshape(batch_size, channels, height, width)

        return out

    def forward(self, x):
        identify = x

        out = self.branch_main_1(x)
        out = self.channel_shuffle(out)
        out = self.branch_main_2(out)

        if self.stride == 2:
            branch_proj = self.branch_proj(x)
            out = self.relu(out)
            out = torch.cat([branch_proj, out], 1)
            return out
        else:
            out = torch.add(out, identify)
            out = self.relu(out)
            return out


def shufflenet_v1_x0_5(**kwargs: Any) -> ShuffleNetV1:
    model = ShuffleNetV1([4, 8, 4], [16, 192, 384, 768], 8, **kwargs)

    return model


def shufflenet_v1_x1_0(**kwargs: Any) -> ShuffleNetV1:
    model = ShuffleNetV1([4, 8, 4], [24, 384, 768, 1536], 8, **kwargs)

    return model


def shufflenet_v1_x1_5(**kwargs: Any) -> ShuffleNetV1:
    model = ShuffleNetV1([4, 8, 4], [24, 576, 1152, 2304], 8, **kwargs)

    return model


def shufflenet_v1_x2_0(**kwargs: Any) -> ShuffleNetV1:
    model = ShuffleNetV1([4, 8, 4], [48, 768, 1536, 3072], 8, **kwargs)

    return model


if __name__ == "__main__":
    # Generating Sample image
    image_size = (1, 3, 640, 640)
    image = torch.rand(*image_size)

    # Model
    model = shufflenet_v1_x0_5()

    out = model(image)
    print(out)