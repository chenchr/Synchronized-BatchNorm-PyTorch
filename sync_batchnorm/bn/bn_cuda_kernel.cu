#include <ATen/ATen.h>

#include <THC.h>
#include <THCGeneral.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>

#define EPSILON 1e-6
#define WARP_SIZE 32
#define BLOCK_SIZE 512

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)
template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

namespace {

template <typename scalar_t>
__global__ void bn_cuda_forward_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ mean,
    const scalar_t* __restrict__ inv_std,
    const scalar_t* __restrict__ gamma,
    const scalar_t* __restrict__ beta,
    const size_t b,
    const size_t h){
    int c = gridDim.x;
    int c_idx = blockIdx.x;
    scalar_t temp_mean = mean[c_idx];
    scalar_t temp_inv_std = inv_std[c_idx];
    scalar_t temp_gamma = gamma[c_idx];
    scalar_t temp_beta = beta[c_idx];
    int mch = c * h;
    int mcidh = c_idx * h;
    for(int i=0; i<b; ++i){
        for(int j=threadIdx.x; j<h; j+=BLOCK_SIZE){
            output[i*mch + mcidh + j] = \
            (input[i*mch + mcidh + j] - temp_mean) * temp_inv_std * temp_gamma + temp_beta;
        }
    }

}

template <typename scalar_t>
__global__ void bn_cuda_backward_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ grad_in,
    scalar_t* __restrict__ grad_mean,
    scalar_t* __restrict__ grad_inv_std,
    scalar_t* __restrict__ grad_gamma,
    scalar_t* __restrict__ grad_beta,
    const scalar_t* __restrict__ mean,
    const scalar_t* __restrict__ inv_std,
    const scalar_t* __restrict__ gamma,
    const scalar_t* __restrict__ beta,
    const size_t b,
    const size_t h,
    bool train){
    __shared__ scalar_t values[BLOCK_SIZE];
    __shared__ scalar_t valuess[BLOCK_SIZE];
    //__syncthreads();
    int c_idx = blockIdx.x;
    int c = gridDim.x;
    int mch = c * h;
    int mcidh = c_idx * h;
    scalar_t temp_s = 0, temp_ss = 0, temp;
    for(int i=0; i<b; ++i){
        for(int j=threadIdx.x; j<h; j+=BLOCK_SIZE){
            temp = grad_out[i*mch + mcidh + j];
            temp_s += temp;
            temp_ss += temp * input[i*mch + mcidh + j];
        }
    }
    values[threadIdx.x] = temp_s;
    valuess[threadIdx.x] = temp_ss;
    __syncthreads();
    // sum over warp
    __shared__ scalar_t values_sum[WARP_SIZE];
    __shared__ scalar_t values_ssum[WARP_SIZE];
    if(threadIdx.x % WARP_SIZE == 0){
        int temp_index = threadIdx.x / WARP_SIZE;
        temp_s = 0;
        temp_ss = 0;
        int temp_max = threadIdx.x + WARP_SIZE;
        for(int i=threadIdx.x; i<temp_max; ++i){
            temp_s += values[i];
            temp_ss += valuess[i];
        }
        values_sum[temp_index] = temp_s;
        values_ssum[temp_index] = temp_ss;
    }
    __syncthreads();
    // sum over sum
    if(threadIdx.x == 0){
        int max = (BLOCK_SIZE-1) / WARP_SIZE + 1;
        temp_s = 0;
        temp_ss = 0;
        for(int i=0; i<max; ++i){
            temp_s += values_sum[i];
            temp_ss += values_ssum[i];
        }
        values_sum[0] = temp_s;
        values_ssum[0] = temp_ss;
    }
    // the reduce operation is done above, next is to assign value
    if(threadIdx.x == 0){
        if(train){
            grad_mean[c_idx] = - gamma[c_idx] * inv_std[c_idx] * values_sum[0];
            grad_inv_std[c_idx] =  gamma[c_idx] * (values_ssum[0] - mean[c_idx] * values_sum[0]);
        }
        grad_gamma[c_idx] = inv_std[c_idx] * (values_ssum[0] - mean[c_idx] * values_sum[0]);
        grad_beta[c_idx] = values_sum[0];
    }
    scalar_t scale = gamma[c_idx] * inv_std[c_idx];
    for(int i=0; i<b; ++i){
        for(int j=threadIdx.x; j<h; j+=BLOCK_SIZE){
            grad_in[i*mch + mcidh + j] = grad_out[i*mch + mcidh + j] * scale;
        }
    }

}
} // namespace

std::vector<at::Tensor> bn_cuda_forward(
    at::Tensor input,
    at::Tensor mean,
    at::Tensor inv_std,
    at::Tensor gamma,
    at::Tensor beta){
    //cudaDeviceSynchronize();
    //THCudaCheck(cudaGetLastError());

    const auto b = input.size(0);
    const auto c = input.size(1);
    const auto h = input.size(2);

    auto output = at::zeros_like(input);

    const int threads = BLOCK_SIZE;
    const int blocks = c;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "bn_forward_cuda", ([&] {
    bn_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        input.data<scalar_t>(),
        output.data<scalar_t>(),
        mean.data<scalar_t>(),
        inv_std.data<scalar_t>(),
        gamma.data<scalar_t>(),
        beta.data<scalar_t>(),
        b, h);
    }));
    //cudaDeviceSynchronize();
    //THCudaCheck(cudaGetLastError());

  return {output};
}

std::vector<at::Tensor> bn_cuda_backward(
    at::Tensor grad_out,
    at::Tensor input,
    at::Tensor mean,
    at::Tensor inv_std,
    at::Tensor gamma,
    at::Tensor beta,
    bool train){
    //cudaDeviceSynchronize();
    //THCudaCheck(cudaGetLastError());

    const auto b = grad_out.size(0);
    const auto c = grad_out.size(1);
    const auto h = grad_out.size(2);

    auto grad_in = at::zeros_like(grad_out);
    auto grad_mean = at::zeros_like(mean);
    auto grad_inv_std = at::zeros_like(inv_std);
    auto grad_gamma = at::zeros_like(gamma);
    auto grad_beta = at::zeros_like(beta);

    const int threads = BLOCK_SIZE;
    const int blocks = c;

    //cudaDeviceSynchronize();
    //THCudaCheck(cudaGetLastError());
    AT_DISPATCH_FLOATING_TYPES(grad_out.type(), "bn_backward_cuda", ([&] {
    bn_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        grad_out.data<scalar_t>(),
        input.data<scalar_t>(),
        grad_in.data<scalar_t>(),
        grad_mean.data<scalar_t>(),
        grad_inv_std.data<scalar_t>(),
        grad_gamma.data<scalar_t>(),
        grad_beta.data<scalar_t>(),
        mean.data<scalar_t>(),
        inv_std.data<scalar_t>(),
        gamma.data<scalar_t>(),
        beta.data<scalar_t>(),
        b,
        h,
        train);
    }));
    //cudaDeviceSynchronize();
    //THCudaCheck(cudaGetLastError());

    return {grad_in, grad_mean, grad_inv_std, grad_gamma, grad_beta};
}


