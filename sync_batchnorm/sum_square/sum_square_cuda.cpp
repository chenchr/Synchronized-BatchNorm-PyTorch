#include <torch/torch.h>

#include <vector>

// CUDA forward declarations

std::vector<at::Tensor> sum_square_cuda_forward(
    at::Tensor fea);

std::vector<at::Tensor> sum_square_cuda_backward(
    at::Tensor grad_sum,
    at::Tensor grad_ssum,
    at::Tensor fea);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> sum_square_forward(
    at::Tensor fea) {
    CHECK_INPUT(fea);
    
    return sum_square_cuda_forward(fea);
}

std::vector<at::Tensor> sum_square_backward(
    at::Tensor grad_sum,
    at::Tensor grad_ssum,
    at::Tensor fea) {
    CHECK_INPUT(grad_sum);
    CHECK_INPUT(grad_ssum);
    CHECK_INPUT(fea);

    return sum_square_cuda_backward(
        grad_sum,
        grad_ssum,
        fea);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sum_square_forward, "sum_square forward (CUDA)");
  m.def("backward", &sum_square_backward, "sum_square backward (CUDA)");
}
