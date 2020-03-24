#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

// modified from https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/mkldnn/Conv.cpp
std::tuple<at::Tensor,at::Tensor,at::Tensor> cpu(
    const at::Tensor& input, const at::Tensor& grad_output_t, const at::Tensor& grad_scale, const at::Tensor& weight,
    at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups)
{
  at::Tensor grad_output = grad_output_t.contiguous();
  at::Tensor scaled_grad_output = grad_output.mul(grad_scale);
  at::Tensor grad_input, grad_weight, grad_bias;

  grad_input                       = at::mkldnn_convolution_backward_input  ( input.sizes(),        grad_output, weight, padding, stride, dilation, groups, true);
  std::tie(grad_weight, grad_bias) = at::mkldnn_convolution_backward_weights(weight.sizes(), scaled_grad_output,  input, padding, stride, dilation, groups, true);

  return std::tuple<at::Tensor,at::Tensor,at::Tensor>{grad_input, grad_weight, grad_bias};
}

// modified from https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cudnn/Conv.cpp
std::tuple<at::Tensor,at::Tensor,at::Tensor> cuda(
    const at::Tensor& input, const at::Tensor& grad_output_t, const at::Tensor& grad_scale, const at::Tensor& weight,
    at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic)
{
  at::Tensor grad_output = grad_output_t.contiguous();
  at::Tensor scaled_grad_output = grad_output.mul(grad_scale);
  at::Tensor grad_input, grad_weight, grad_bias;

  grad_input  = at::cudnn_convolution_backward_input ( input.sizes(),        grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
  grad_weight = at::cudnn_convolution_backward_weight(weight.sizes(), scaled_grad_output,  input, padding, stride, dilation, groups, benchmark, deterministic);
  grad_bias   = scaled_grad_output.sum({0,2,3});

  return std::tuple<at::Tensor,at::Tensor,at::Tensor>{grad_input, grad_weight, grad_bias};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def( "cpu",  &cpu, "conv2d backward (cpu)");
  m.def("cuda", &cuda, "conv2d backward (cuda)");
}
