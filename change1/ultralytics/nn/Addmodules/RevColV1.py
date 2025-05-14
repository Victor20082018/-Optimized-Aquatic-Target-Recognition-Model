# --------------------------------------------------------
# Reversible Column Networks
# Copyright (c) 2022 Megvii Inc.
# Licensed under The Apache License 2.0 [see LICENSE for details]
# Written by Yuxuan Cai
# --------------------------------------------------------
from typing import Tuple, Any, List
from timm.models.layers import trunc_normal_
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

__all__ = ['revcol_tiny', 'revcol_small', 'revcol_base', 'revcol_large', 'revcol_xlarge']

class UpSampleConvnext(nn.Module):
    def __init__(self, ratio, inchannel, outchannel):
        super().__init__()
        self.ratio = ratio
        self.channel_reschedule = nn.Sequential(
            # LayerNorm(inchannel, eps=1e-6, data_format="channels_last"),
            nn.Linear(inchannel, outchannel),
            LayerNorm(outchannel, eps=1e-6, data_format="channels_last"))
        self.upsample = nn.Upsample(scale_factor=2 ** ratio, mode='nearest')

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.channel_reschedule(x)
        x = x = x.permute(0, 3, 1, 2)

        return self.upsample(x)


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first", elementwise_affine=True):
        super().__init__()
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            if self.elementwise_affine:
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNextBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, in_channel, hidden_dim, out_channel, kernel_size=3, layer_scale_init_value=1e-6, drop_path=0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                                groups=in_channel)  # depthwise conv
        self.norm = nn.LayerNorm(in_channel, eps=1e-6)
        self.pwconv1 = nn.Linear(in_channel, hidden_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, out_channel)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((out_channel)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        # print(f"x min: {x.min()}, x max: {x.max()}, input min: {input.min()}, input max: {input.max()}, x mean: {x.mean()}, x var: {x.var()}, ratio: {torch.sum(x>8)/x.numel()}")
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class Decoder(nn.Module):
    def __init__(self, depth=[2, 2, 2, 2], dim=[112, 72, 40, 24], block_type=None, kernel_size=3) -> None:
        super().__init__()
        self.depth = depth
        self.dim = dim
        self.block_type = block_type
        self._build_decode_layer(dim, depth, kernel_size)
        self.projback = nn.Sequential(
            nn.Conv2d(
                in_channels=dim[-1],
                out_channels=4 ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(4),
        )

    def _build_decode_layer(self, dim, depth, kernel_size):
        normal_layers = nn.ModuleList()
        upsample_layers = nn.ModuleList()
        proj_layers = nn.ModuleList()

        norm_layer = LayerNorm

        for i in range(1, len(dim)):
            module = [self.block_type(dim[i], dim[i], dim[i], kernel_size) for _ in range(depth[i])]
            normal_layers.append(nn.Sequential(*module))
            upsample_layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            proj_layers.append(nn.Sequential(
                nn.Conv2d(dim[i - 1], dim[i], 1, 1),
                norm_layer(dim[i]),
                nn.GELU()
            ))
        self.normal_layers = normal_layers
        self.upsample_layers = upsample_layers
        self.proj_layers = proj_layers

    def _forward_stage(self, stage, x):
        x = self.proj_layers[stage](x)
        x = self.upsample_layers[stage](x)
        return self.normal_layers[stage](x)

    def forward(self, c3):
        x = self._forward_stage(0, c3)  # 14
        x = self._forward_stage(1, x)  # 28
        x = self._forward_stage(2, x)  # 56
        x = self.projback(x)
        return x


class SimDecoder(nn.Module):
    def __init__(self, in_channel, encoder_stride) -> None:
        super().__init__()
        self.projback = nn.Sequential(
            LayerNorm(in_channel),
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=encoder_stride ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(encoder_stride),
        )

    def forward(self, c3):
        return self.projback(c3)


def get_gpu_states(fwd_gpu_devices) -> Tuple[List[int], List[torch.Tensor]]:
    # This will not error out if "arg" is a CPU tensor or a non-tensor type because
    # the conditionals short-circuit.
    fwd_gpu_states = []
    for device in fwd_gpu_devices:
        with torch.cuda.device(device):
            fwd_gpu_states.append(torch.cuda.get_rng_state())

    return fwd_gpu_states


def get_gpu_device(*args):
    fwd_gpu_devices = list(set(arg.get_device() for arg in args
                               if isinstance(arg, torch.Tensor) and arg.is_cuda))
    return fwd_gpu_devices


def set_device_states(fwd_cpu_state, devices, states) -> None:
    torch.set_rng_state(fwd_cpu_state)
    for device, state in zip(devices, states):
        with torch.cuda.device(device):
            torch.cuda.set_rng_state(state)


def detach_and_grad(inputs: Tuple[Any, ...]) -> Tuple[torch.Tensor, ...]:
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            if not isinstance(inp, torch.Tensor):
                out.append(inp)
                continue

            x = inp.detach()
            x.requires_grad = True
            out.append(x)
        return tuple(out)
    else:
        raise RuntimeError(
            "Only tuple of tensors is supported. Got Unsupported input type: ", type(inputs).__name__)


def get_cpu_and_gpu_states(gpu_devices):
    return torch.get_rng_state(), get_gpu_states(gpu_devices)


class ReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
        l0, l1, l2, l3 = run_functions
        alpha0, alpha1, alpha2, alpha3 = alpha
        ctx.run_functions = run_functions
        ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 5
        [x, c0, c1, c2, c3] = args
        if type(c0) == int:
            ctx.first_col = True
        else:
            ctx.first_col = False
        with torch.no_grad():
            gpu_devices = get_gpu_device(*args)
            ctx.gpu_devices = gpu_devices
            ctx.cpu_states_0, ctx.gpu_states_0 = get_cpu_and_gpu_states(gpu_devices)
            c0 = l0(x, c1) + c0 * alpha0
            ctx.cpu_states_1, ctx.gpu_states_1 = get_cpu_and_gpu_states(gpu_devices)
            c1 = l1(c0, c2) + c1 * alpha1
            ctx.cpu_states_2, ctx.gpu_states_2 = get_cpu_and_gpu_states(gpu_devices)
            c2 = l2(c1, c3) + c2 * alpha2
            ctx.cpu_states_3, ctx.gpu_states_3 = get_cpu_and_gpu_states(gpu_devices)
            c3 = l3(c2, None) + c3 * alpha3
        ctx.save_for_backward(x, c0, c1, c2, c3)
        return x, c0, c1, c2, c3

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, c0, c1, c2, c3 = ctx.saved_tensors
        l0, l1, l2, l3 = ctx.run_functions
        alpha0, alpha1, alpha2, alpha3 = ctx.alpha
        gx_right, g0_right, g1_right, g2_right, g3_right = grad_outputs
        (x, c0, c1, c2, c3) = detach_and_grad((x, c0, c1, c2, c3))

        with torch.enable_grad(), \
                torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
                torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):

            g3_up = g3_right
            g3_left = g3_up * alpha3  ##shortcut
            set_device_states(ctx.cpu_states_3, ctx.gpu_devices, ctx.gpu_states_3)
            oup3 = l3(c2, None)
            torch.autograd.backward(oup3, g3_up, retain_graph=True)
            with torch.no_grad():
                c3_left = (1 / alpha3) * (c3 - oup3)  ## feature reverse
            g2_up = g2_right + c2.grad
            g2_left = g2_up * alpha2  ##shortcut

            (c3_left,) = detach_and_grad((c3_left,))
            set_device_states(ctx.cpu_states_2, ctx.gpu_devices, ctx.gpu_states_2)
            oup2 = l2(c1, c3_left)
            torch.autograd.backward(oup2, g2_up, retain_graph=True)
            c3_left.requires_grad = False
            cout3 = c3_left * alpha3  ##alpha3 update
            torch.autograd.backward(cout3, g3_up)

            with torch.no_grad():
                c2_left = (1 / alpha2) * (c2 - oup2)  ## feature reverse
            g3_left = g3_left + c3_left.grad if c3_left.grad is not None else g3_left
            g1_up = g1_right + c1.grad
            g1_left = g1_up * alpha1  ##shortcut

            (c2_left,) = detach_and_grad((c2_left,))
            set_device_states(ctx.cpu_states_1, ctx.gpu_devices, ctx.gpu_states_1)
            oup1 = l1(c0, c2_left)
            torch.autograd.backward(oup1, g1_up, retain_graph=True)
            c2_left.requires_grad = False
            cout2 = c2_left * alpha2  ##alpha2 update
            torch.autograd.backward(cout2, g2_up)

            with torch.no_grad():
                c1_left = (1 / alpha1) * (c1 - oup1)  ## feature reverse
            g0_up = g0_right + c0.grad
            g0_left = g0_up * alpha0  ##shortcut
            g2_left = g2_left + c2_left.grad if c2_left.grad is not None else g2_left  ## Fusion

            (c1_left,) = detach_and_grad((c1_left,))
            set_device_states(ctx.cpu_states_0, ctx.gpu_devices, ctx.gpu_states_0)
            oup0 = l0(x, c1_left)
            torch.autograd.backward(oup0, g0_up, retain_graph=True)
            c1_left.requires_grad = False
            cout1 = c1_left * alpha1  ##alpha1 update
            torch.autograd.backward(cout1, g1_up)

            with torch.no_grad():
                c0_left = (1 / alpha0) * (c0 - oup0)  ## feature reverse
            gx_up = x.grad  ## Fusion
            g1_left = g1_left + c1_left.grad if c1_left.grad is not None else g1_left  ## Fusion
            c0_left.requires_grad = False
            cout0 = c0_left * alpha0  ##alpha0 update
            torch.autograd.backward(cout0, g0_up)

        if ctx.first_col:
            return None, None, gx_up, None, None, None, None
        else:
            return None, None, gx_up, g0_left, g1_left, g2_left, g3_left


class Fusion(nn.Module):
    def __init__(self, level, channels, first_col) -> None:
        super().__init__()

        self.level = level
        self.first_col = first_col
        self.down = nn.Sequential(
            nn.Conv2d(channels[level - 1], channels[level], kernel_size=2, stride=2),
            LayerNorm(channels[level], eps=1e-6, data_format="channels_first"),
        ) if level in [1, 2, 3] else nn.Identity()
        if not first_col:
            self.up = UpSampleConvnext(1, channels[level + 1], channels[level]) if level in [0, 1, 2] else nn.Identity()

    def forward(self, *args):

        c_down, c_up = args

        if self.first_col:
            x = self.down(c_down)
            return x

        if self.level == 3:
            x = self.down(c_down)
        else:
            x = self.up(c_up) + self.down(c_down)
        return x


class Level(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, first_col, dp_rate=0.0) -> None:
        super().__init__()
        countlayer = sum(layers[:level])
        expansion = 4
        self.fusion = Fusion(level, channels, first_col)
        modules = [ConvNextBlock(channels[level], expansion * channels[level], channels[level], kernel_size=kernel_size,
                                 layer_scale_init_value=1e-6, drop_path=dp_rate[countlayer + i]) for i in
                   range(layers[level])]
        self.blocks = nn.Sequential(*modules)

    def forward(self, *args):
        x = self.fusion(*args)
        x = self.blocks(x)
        return x


class SubNet(nn.Module):
    def __init__(self, channels, layers, kernel_size, first_col, dp_rates, save_memory) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        self.save_memory = save_memory
        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[0], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None

        self.level0 = Level(0, channels, layers, kernel_size, first_col, dp_rates)

        self.level1 = Level(1, channels, layers, kernel_size, first_col, dp_rates)

        self.level2 = Level(2, channels, layers, kernel_size, first_col, dp_rates)

        self.level3 = Level(3, channels, layers, kernel_size, first_col, dp_rates)

    def _forward_nonreverse(self, *args):
        x, c0, c1, c2, c3 = args

        c0 = (self.alpha0) * c0 + self.level0(x, c1)
        c1 = (self.alpha1) * c1 + self.level1(c0, c2)
        c2 = (self.alpha2) * c2 + self.level2(c1, c3)
        c3 = (self.alpha3) * c3 + self.level3(c2, None)
        return c0, c1, c2, c3

    def _forward_reverse(self, *args):

        local_funs = [self.level0, self.level1, self.level2, self.level3]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3]
        _, c0, c1, c2, c3 = ReverseFunction.apply(
            local_funs, alpha, *args)

        return c0, c1, c2, c3

    def forward(self, *args):

        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        self._clamp_abs(self.alpha3.data, 1e-3)

        if self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign


class Classifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_channels, eps=1e-6),  # final norm layer
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class FullNet(nn.Module):
    def __init__(self, channels=[32, 64, 96, 128], layers=[2, 3, 6, 3], num_subnet=5, kernel_size=3, drop_path=0.0,
                 save_memory=True, inter_supv=True) -> None:
        super().__init__()
        self.num_subnet = num_subnet
        self.inter_supv = inter_supv
        self.channels = channels
        self.layers = layers

        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=4, stride=4),
            LayerNorm(channels[0], eps=1e-6, data_format="channels_first")
        )

        dp_rate = [x.item() for x in torch.linspace(0, drop_path, sum(layers))]
        for i in range(num_subnet):
            first_col = True if i == 0 else False
            self.add_module(f'subnet{str(i)}', SubNet(
                channels, layers, kernel_size, first_col, dp_rates=dp_rate, save_memory=save_memory))

        self.apply(self._init_weights)
        self.width_list = [i.size(1) for i in self.forward(torch.randn(1, 3, 640, 640))]

    def forward(self, x):

        c0, c1, c2, c3 = 0, 0, 0, 0
        x = self.stem(x)
        for i in range(self.num_subnet):
            c0, c1, c2, c3 = getattr(self, f'subnet{str(i)}')(x, c0, c1, c2, c3)
        return [c0, c1, c2, c3]

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            trunc_normal_(module.weight, std=.02)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            nn.init.constant_(module.bias, 0)


##-------------------------------------- Tiny -----------------------------------------

def revcol_tiny(save_memory=True, inter_supv=True, drop_path=0.1, kernel_size=3):
    channels = [64, 128, 256, 512]
    layers = [2, 2, 4, 2]
    num_subnet = 4
    return FullNet(channels, layers, num_subnet, drop_path=drop_path, save_memory=save_memory, inter_supv=inter_supv,
                   kernel_size=kernel_size)


##-------------------------------------- Small -----------------------------------------

def revcol_small(save_memory=True, inter_supv=True, drop_path=0.3, kernel_size=3):
    channels = [64, 128, 256, 512]
    layers = [2, 2, 4, 2]
    num_subnet = 8
    return FullNet(channels, layers, num_subnet, drop_path=drop_path, save_memory=save_memory, inter_supv=inter_supv,
                   kernel_size=kernel_size)


##-------------------------------------- Base -----------------------------------------

def revcol_base(save_memory=True, inter_supv=True, drop_path=0.4, kernel_size=3, head_init_scale=None):
    channels = [72, 144, 288, 576]
    layers = [1, 1, 3, 2]
    num_subnet = 16
    return FullNet(channels, layers, num_subnet, drop_path=drop_path, save_memory=save_memory, inter_supv=inter_supv,
                   kernel_size=kernel_size)


##-------------------------------------- Large -----------------------------------------

def revcol_large(save_memory=True, inter_supv=True, drop_path=0.5, kernel_size=3, head_init_scale=None):
    channels = [128, 256, 512, 1024]
    layers = [1, 2, 6, 2]
    num_subnet = 8
    return FullNet(channels, layers, num_subnet, drop_path=drop_path, save_memory=save_memory, inter_supv=inter_supv,
                   kernel_size=kernel_size)


##--------------------------------------Extra-Large -----------------------------------------
def revcol_xlarge(save_memory=True, inter_supv=True, drop_path=0.5, kernel_size=3, head_init_scale=None):
    channels = [224, 448, 896, 1792]
    layers = [1, 2, 6, 2]
    num_subnet = 8
    return FullNet(channels, layers, num_subnet, drop_path=drop_path, save_memory=save_memory, inter_supv=inter_supv,
                   kernel_size=kernel_size)

# model = revcol_xlarge(True)
# # 示例输入
# input = torch.randn(64, 3, 224, 224)
# output = model(input)
#
# print(len(output))#torch.Size([3, 64, 224, 224])