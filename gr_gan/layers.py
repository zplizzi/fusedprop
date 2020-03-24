import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from torch.utils.cpp_extension import load
dir_path = os.path.dirname(os.path.realpath(__file__))
conv2d_backward = load(name='conv2d_backward',
                       sources=[dir_path + '/conv2d_backward.cpp'],
                       verbose=True)

# This is the scaling factor applied to gradients in G
global scale1
# This is the scaling factor applied to gradients in D
global scale2


class GradScaling(torch.autograd.Function):
    """Pass input unmodified in the forward pass, but scale in backward."""
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        global scale1
        return grad_output * scale1.view(-1, 1, 1, 1)


grad_scaling = GradScaling.apply


class GradScalingLinear(nn.Linear):
    def forward(self, input):
        class _GradScalingLinear(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input, weight, bias):
                ctx.save_for_backward(input, weight, bias)
                return F.linear(input, weight, bias)

            @staticmethod
            def backward(ctx, grad_output):
                global scale2
                scaled_grad_output = grad_output * scale2.view(-1, 1)
                input, weight, bias = ctx.saved_tensors
                return (grad_output.matmul(weight),
                        scaled_grad_output.t().matmul(input),
                        scaled_grad_output.sum(0)
                        if bias is not None else None)

        return _GradScalingLinear.apply(input, self.weight, self.bias)


class GradScalingConv2d(nn.Conv2d):
    def forward(self, input):
        class _GradScalingConv2d(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input, weight, bias):
                ctx.save_for_backward(input, weight, bias)
                return F.conv2d(input, weight, bias, self.stride, self.padding,
                                self.dilation, self.groups)

            @staticmethod
            def backward(ctx, grad_output):
                global scale2
                input, weight, bias = ctx.saved_tensors

                cargs = input, grad_output, scale2.view(
                    -1, 1, 1, 1
                ), weight, self.padding, self.stride, self.dilation, self.groups
                if grad_output.is_cuda:
                    return conv2d_backward.cuda(
                        *cargs, torch.backends.cudnn.benchmark,
                        torch.backends.cudnn.deterministic)
                else:
                    return conv2d_backward.cpu(*cargs)

        assert self.padding_mode != 'circular'
        return _GradScalingConv2d.apply(input, self.weight, self.bias)


class GradScalingBatchNorm2d(nn.BatchNorm2d):
    # nchunks=2 when running D([real,fake]) in one batch, which seems unavoidable for now.
    def __init__(self, num_features, nchunks=1):
        super().__init__(num_features)
        self.nchunks = nchunks

    def forward(self, inputs):
        class _Affine(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input, weight, bias):
                ctx.save_for_backward(input, weight, bias)
                return input * weight.view(1, -1, 1, 1) + bias.view(
                    1, -1, 1, 1)

            @staticmethod
            def backward(ctx, grad_output):
                global scale2
                scaled_grad_output = grad_output * scale2.view(-1, 1, 1, 1)
                input, weight, bias = ctx.saved_tensors
                return (grad_output * weight.view(1, -1, 1, 1),
                        (scaled_grad_output * input).sum(
                            (0, 2, 3)), scaled_grad_output.sum((0, 2, 3)))

        outputs = []
        for input in inputs.chunk(self.nchunks):
            self.num_batches_tracked = self.num_batches_tracked + 1
            output = F.batch_norm(input, self.running_mean, self.running_var,
                                  None, None, self.training, self.momentum,
                                  self.eps)
            outputs.append(_Affine.apply(output, self.weight, self.bias))
        return torch.cat(outputs)
