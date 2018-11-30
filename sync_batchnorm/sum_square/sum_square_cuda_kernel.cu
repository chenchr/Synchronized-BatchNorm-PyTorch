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
__global__ void sum_square_cuda_forward_kernel(
    const scalar_t* __restrict__ fea,
    scalar_t* __restrict__ sum,
    scalar_t* __restrict__ ssum,
    const size_t b,
    const size_t h){
    __shared__ scalar_t values[BLOCK_SIZE];
    __shared__ scalar_t valuess[BLOCK_SIZE];
    values[threadIdx.x] = 0;
    valuess[threadIdx.x] = 0;
    //__syncthreads();
    int c_idx = blockIdx.x;
    int c = gridDim.x;
    int mch = c * h;
    int mcidh = c_idx * h;
    scalar_t temp;
    scalar_t temp_values = 0, temp_valuess = 0;
    //int tidx = thre
    for(int i=0; i<b; ++i){
        for(int j=threadIdx.x; j<h; j+=BLOCK_SIZE){
            temp = fea[i*mch + mcidh + j];
            temp_values += temp;
            temp_valuess += temp * temp;
        }
    }
    values[threadIdx.x] = temp_values;
    valuess[threadIdx.x] = temp_valuess;
    __syncthreads();
    // sum over warp
    __shared__ scalar_t values_sum[WARP_SIZE];
    __shared__ scalar_t values_ssum[WARP_SIZE];
    if(threadIdx.x % WARP_SIZE == 0){
        int temp_index = threadIdx.x / WARP_SIZE;
        int temp_max = threadIdx.x + WARP_SIZE;
        temp_values = 0;
        temp_valuess = 0;
        for(int i=threadIdx.x; i<temp_max; ++i){
            temp_values += values[i];
            temp_valuess += valuess[i];
        }
        values_sum[temp_index] = temp_values;
        values_ssum[temp_index] = temp_valuess;
    }
    __syncthreads();
    // sum over sum
    if(threadIdx.x == 0){
        int max = (BLOCK_SIZE-1) / WARP_SIZE + 1;
        temp_values = 0;
        temp_valuess = 0;
        for(int i=0; i<max; ++i){
            temp_values += values_sum[i];
            temp_valuess += values_ssum[i];
        }
        sum[c_idx] = temp_values;
        ssum[c_idx] = temp_valuess;
    }

}

template <typename scalar_t>
__global__ void sum_square_cuda_backward_kernel(
    const scalar_t* __restrict__ grad_sum,
    const scalar_t* __restrict__ grad_ssum,
    const scalar_t* __restrict__ fea,
    scalar_t* __restrict__ grad_in,
    const size_t b,
    const size_t h){
    int mch = gridDim.x * h;
    int mxh = blockIdx.x * h;
    scalar_t temp_sum = grad_sum[blockIdx.x];
    scalar_t temp_ssum = 2 * grad_ssum[blockIdx.x];
    for(int i=0; i<b; ++i){
        for(int j=threadIdx.x; j<h; j+=BLOCK_SIZE){
            grad_in[i*mch + mxh + j] = temp_sum + temp_ssum * fea[i*mch + mxh + j];
        }
    }
}
} // namespace

std::vector<at::Tensor> sum_square_cuda_forward(
    at::Tensor fea){
    //cudaDeviceSynchronize();
    //THCudaCheck(cudaGetLastError());

    const auto b = fea.size(0);
    const auto c = fea.size(1);
    const auto h = fea.size(2);

    auto sum = at::zeros({c}, fea.options());
    auto ssum = at::zeros({c}, fea.options());

    const int threads = BLOCK_SIZE;
    const int blocks = c;

    AT_DISPATCH_FLOATING_TYPES(fea.type(), "sum_square_forward_cuda", ([&] {
    sum_square_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        fea.data<scalar_t>(),
        sum.data<scalar_t>(),
        ssum.data<scalar_t>(),
        b, h);
    }));
    //cudaDeviceSynchronize();
    //THCudaCheck(cudaGetLastError());

  return {sum, ssum};
}

std::vector<at::Tensor> sum_square_cuda_backward(
    at::Tensor grad_sum,
    at::Tensor grad_ssum,
    at::Tensor fea) {
    //cudaDeviceSynchronize();
    //THCudaCheck(cudaGetLastError());

    const auto b = fea.size(0);
    const auto c = fea.size(1);
    const auto h = fea.size(2);
    auto grad_in = at::zeros_like(fea);

    const int threads = BLOCK_SIZE;
    const int blocks = c;

    AT_DISPATCH_FLOATING_TYPES(fea.type(), "sum_square_forward_cuda", ([&] {
    sum_square_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        grad_sum.data<scalar_t>(),
        grad_ssum.data<scalar_t>(),
        fea.data<scalar_t>(),
        grad_in.data<scalar_t>(),
        b,
        h);
    }));
    //cudaDeviceSynchronize();
    //THCudaCheck(cudaGetLastError());

    return {grad_in};
}


