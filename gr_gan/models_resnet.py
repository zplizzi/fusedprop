# This code derived from https://github.com/TAMU-VITA/AutoGAN
import torch
import math
from torch import nn
from torch.nn import functional as F
from gr_gan import layers

sn = torch.nn.utils.spectral_norm


def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return F.avg_pool2d(x, 2)


class DisBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 conv,
                 hidden_channels=None,
                 kernel_size=3,
                 padding=1,
                 activation=F.relu,
                 downsample=False):
        super().__init__()
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels

        self.c1 = sn(
            conv(in_channels,
                 hidden_channels,
                 kernel_size=kernel_size,
                 padding=padding))
        self.c2 = sn(
            conv(hidden_channels,
                 out_channels,
                 kernel_size=kernel_size,
                 padding=padding))
        if self.learnable_sc:
            self.c_sc = sn(
                conv(in_channels, out_channels, kernel_size=1, padding=0))
            torch.nn.init.xavier_uniform_(self.c_sc.weight)
            torch.nn.init.zeros_(self.c_sc.bias)

        torch.nn.init.xavier_uniform_(self.c1.weight, gain=math.sqrt(2))
        torch.nn.init.xavier_uniform_(self.c2.weight, gain=math.sqrt(2))
        torch.nn.init.zeros_(self.c1.bias)
        torch.nn.init.zeros_(self.c2.bias)

    def residual(self, x):
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class OptimizedDisBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 conv,
                 kernel_size=3,
                 padding=1,
                 activation=F.relu):
        super().__init__()
        self.activation = activation
        self.c1 = sn(
            conv(in_channels,
                 out_channels,
                 kernel_size=kernel_size,
                 padding=padding))
        self.c2 = sn(
            conv(out_channels,
                 out_channels,
                 kernel_size=kernel_size,
                 padding=padding))
        self.c_sc = sn(
            conv(in_channels, out_channels, kernel_size=1, padding=0))

        torch.nn.init.xavier_uniform_(self.c1.weight, gain=math.sqrt(2))
        torch.nn.init.xavier_uniform_(self.c2.weight, gain=math.sqrt(2))
        torch.nn.init.xavier_uniform_(self.c_sc.weight)
        torch.nn.init.zeros_(self.c1.bias)
        torch.nn.init.zeros_(self.c2.bias)
        torch.nn.init.zeros_(self.c_sc.bias)

    def residual(self, x):
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = _downsample(h)
        return h

    def shortcut(self, x):
        return self.c_sc(_downsample(x))

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


def _upsample(x):
    return torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")


def upsample_conv(x, conv):
    return conv(_upsample(x))


class GenBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=None,
                 kernel_size=3,
                 padding=1,
                 activation=F.relu,
                 upsample=False):
        super().__init__()
        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = in_channels != out_channels or upsample
        hidden_channels = out_channels if hidden_channels is None else hidden_channels

        self.c1 = nn.Conv2d(in_channels,
                            hidden_channels,
                            kernel_size=kernel_size,
                            padding=padding)
        self.c2 = nn.Conv2d(hidden_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            padding=padding)
        self.b1 = nn.BatchNorm2d(in_channels)
        self.b2 = nn.BatchNorm2d(hidden_channels)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels,
                                  out_channels,
                                  kernel_size=1,
                                  padding=0)
            torch.nn.init.xavier_uniform_(self.c_sc.weight)
            torch.nn.init.zeros_(self.c_sc.bias)

        torch.nn.init.xavier_uniform_(self.c1.weight, gain=math.sqrt(2))
        torch.nn.init.xavier_uniform_(self.c2.weight, gain=math.sqrt(2))
        torch.nn.init.zeros_(self.c1.bias)
        torch.nn.init.zeros_(self.c2.bias)

    def residual(self, x):
        h = x
        h = self.b1(h)
        h = self.activation(h)
        h = upsample_conv(h, self.c1) if self.upsample else self.c1(h)
        h = self.b2(h)
        h = self.activation(h)
        h = self.c2(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = upsample_conv(x, self.c_sc) if self.upsample else self.c_sc(x)
            return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class ResNetGenerator(nn.Module):
    def __init__(self,
                 args,
                 ch=256,
                 dim_z=128,
                 bottom_width=4,
                 activation=F.relu):
        super().__init__()
        self.bottom_width = bottom_width
        self.activation = activation
        self.dim_z = dim_z

        self.l1 = nn.Linear(dim_z, (bottom_width**2) * ch)
        self.block2 = GenBlock(ch, ch, activation=activation, upsample=True)
        self.block3 = GenBlock(ch, ch, activation=activation, upsample=True)
        self.block4 = GenBlock(ch, ch, activation=activation, upsample=True)
        self.b5 = nn.BatchNorm2d(ch)
        self.c5 = nn.Conv2d(ch, 3, kernel_size=3, stride=1, padding=1)

        torch.nn.init.xavier_uniform_(self.l1.weight)
        torch.nn.init.xavier_uniform_(self.c5.weight)
        torch.nn.init.zeros_(self.l1.bias)
        torch.nn.init.zeros_(self.c5.bias)

    def forward(self, z):
        h = z
        h = self.l1(h)
        h = h.reshape((h.shape[0], -1, self.bottom_width, self.bottom_width))
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.b5(h)
        h = self.activation(h)
        h = torch.tanh(self.c5(h))
        return h


class ResNetDiscriminator(nn.Module):
    def __init__(self, args, ch=128, activation=F.relu):
        super().__init__()

        # Haven't implemented this disabled
        assert args.spectral_norm is True

        if args.lambda_inv:
            linear = layers.GradScalingLinear
            conv = layers.GradScalingConv2d
        else:
            linear = nn.Linear
            conv = nn.Conv2d

        self.activation = activation
        self.block1 = OptimizedDisBlock(3, ch, conv)
        self.block2 = DisBlock(ch,
                               ch,
                               conv,
                               activation=activation,
                               downsample=True)
        self.block3 = DisBlock(ch,
                               ch,
                               conv,
                               activation=activation,
                               downsample=False)
        self.block4 = DisBlock(ch,
                               ch,
                               conv,
                               activation=activation,
                               downsample=False)
        self.l5 = sn(linear(ch, 1, bias=False))

        torch.nn.init.xavier_uniform_(self.l5.weight)

    def forward(self, x):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)
        # Global average pooling
        h = h.sum(dim=(2, 3))
        output = self.l5(h)
        return output
