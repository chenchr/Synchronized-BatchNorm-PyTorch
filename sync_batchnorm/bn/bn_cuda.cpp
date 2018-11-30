#include <torch/torch.h>

#include <vector>

// CUDA forward declarations

std::vector<at::Tensor> bn_cuda_forward(
    at::Tensor input,
    at::Tensor mean,
    at::Tensor inv_std,
    at::Tensor gamma,
    at::Tensor beta);

std::vector<at::Tensor> bn_cuda_backward(
    at::Tensor grad_out,
    at::Tensor input,
    at::Tensor mean,
    at::Tensor inv_std,
    at::Tensor gamma,
    at::Tensor beta,
    bool train);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> bn_forward(
    at::Tensor input,
    at::Tensor mean,
    at::Tensor inv_std,
    at::Tensor gamma,
    at::Tensor beta){
    
    return bn_cuda_forward(input, mean, inv_std, gamma, beta);
}

std::vector<at::Tensor> bn_backward(
    at::Tensor grad_out,
    at::Tensor input,
    at::Tensor mean,
    at::Tensor inv_std,
    at::Tensor gamma,
    at::Tensor beta,
    bool train){

    return bn_cuda_backward(grad_out, input, mean, inv_std, gamma, beta, train);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &bn_forward, "bn forward (CUDA)");
  m.def("backward", &bn_backward, "bn backward (CUDA)");
}
