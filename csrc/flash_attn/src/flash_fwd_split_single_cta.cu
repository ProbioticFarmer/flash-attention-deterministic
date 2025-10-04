#include "flash.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

namespace FLASH_NAMESPACE {
    template <typename scalar_t>
    __device__ inline scalar_t to_scalar(float v) {
        return static_cast<scalar_t>(v);
    }

    template <>
    __device__ inline at::Half to_scalar<at::Half>(float v) {
        return __float2half(v);
    }

    template <>
    __device__ inline at::BFloat16 to_scalar<at::BFloat16>(float v) {
        return __float2bfloat16(v);
    }

    template <typename scalar_t>
    __global__ void splitkv_single_cta_kernel(
        scalar_t* __restrict__ out, const float* __restrict__ out_accum,
        float* __restrict__ softmax_lse, const float* __restrict__ softmax_lse_accum,
        int num_splits, int batch, int num_heads, int seqlen_q, int head_size, int head_size_accum,
        int64_t out_stride_b, int64_t out_stride_h, int64_t out_stride_m,
        int64_t accum_stride_split, int64_t accum_stride_b, int64_t accum_stride_h, int64_t accum_stride_m, int64_t accum_stride_d,
        int64_t lse_stride_split, int64_t lse_stride_b, int64_t lse_stride_h, int64_t lse_stride_m
    ) {
        const int total_rows = batch * num_heads * seqlen_q;
        int row = threadIdx.x;
        for (; row < total_rows; row += blockDim.x) {
            int tmp = row;
            int m = tmp % seqlen_q; // query index
            tmp /= seqlen_q;
            int h = tmp % num_heads; // head index
            int b = tmp / num_heads; // batch index

            const float* lse_accum_ptr = softmax_lse_accum + b * lse_stride_b + h * lse_stride_h + m * lse_stride_m; // increment will traverse split index

            // compute max value across splits in softmax_lse_accum
            float max_val = -CUDART_INF_F;
            for (int s = 0; s < num_splits; ++s) {
                max_val = fmaxf(max_val, lse_accum_ptr[s * lse_stride_split]);
            }

            scalar_t* out_ptr = out + b * out_stride_b + h * out_stride_h + m * out_stride_m;
            if (max_val == -CUDART_INF_F) {
                softmax_lse[b * out_stride_b + h * out_stride_h + m * out_stride_m] = max_val;
                for (int d = 0; d < head_size; ++d) {
                    out_ptr[d] = to_scalar<scalar_t>(0.f);
                }
                continue;
            }

            // reduce total sum across splits from softmax_lse_accum to store in softmax_lse
            float sum_exp = 0.f;
            for (int s = 0; s < num_splits; ++s) {
                sum_exp += expf(lse_accum_ptr[s * lse_stride_split] - max_val); // += partial_sum_split / exp(max_val)
            }
            const float total_lse = logf(sum_exp) + max_val; // = log(sum of partial sums)
            const float inv_sum = 1.f / sum_exp;
            softmax_lse[b * out_stride_b + h * out_stride_h + m * out_stride_m] = total_lse;

            const float* accum_ptr = out_accum + b * accum_stride_b + h * accum_stride_h + m * accum_stride_m;

            // compute partial output value across splits from out_accum
            // compute partial weight across splits from partial sum and total sum
            // reduce total output value in out
            for (int d = 0; d < head_size; ++d) {
                float acc = 0.f;
                for (int s = 0; s < num_splits; ++s) {
                    const float weight = expf(lse_accum_ptr[s * lse_stride_split] - max_val) * inv_sum; // partial_sum_split / sum of partial sums
                    const float value = accum_ptr[s * accum_stride_split + d * accum_stride_d];
                    acc += weight * value;
                }
                out_ptr[d] = to_scalar<scalar_t>(acc);
            }
        }
    }

    void run_splitkv_single_cta_combine(
        const at::Tensor& out_accum, // float [num_splits, batch, num_heads, seqlen_q, head_dim_round]
        const at::Tensor& softmax_lse_accum, // [num_splits, batch, num_heads, seqlen_q]
        at::Tensor& out, // attn @ V [batch, num_heads, seqlen_q, head_dim]
        at::Tensor& softmax_lse // [batch, num_heads, seqlen_q]
    ) {
        TORCH_CHECK(out_accum.is_cuda(), "out_accum must be on CUDA");
        TORCH_CHECK(softmax_lse_accum.is_cuda(), "softmax_lse_accum must be on CUDA");
        TORCH_CHECK(out.is_cuda(), "out must be on CUDA");
        TORCH_CHECK(softmax_lse.is_cuda(), "softmax_lse must be on CUDA");

        TORCH_CHECK(out_accum.dim() == 5, "out_accum must have shape [num_splits, batch, head, seqlen_q, head_dim_round]");
        TORCH_CHECK(softmax_lse_accum.dim() == 4, "softmax_lse_accum must have shape [num_splits, batch, heads, seqlen_q]");
        
        const int num_splits = out_accum.size(0);
        const int batch = out_accum.size(1);
        const int num_heads = out_accum.size(2);
        const int seqlen_q = out_accum.size(3);
        const int head_size_accum = out_accum.size(4);
        const int head_size = out.size(-1);

        if (num_splits == 0 || batch == 0 || num_heads == 0 || seqlen_q == 0) {
            return;
        }

        constexpr int threads = 256;
        const dim3 block(threads);
        const dim3 grid(1);

        auto stream = at::cuda::getCurrentCUDAStream();

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kHalf, at::kBFloat16, out.scalar_type(),
            "splitkv_single_cta_combine", [&] {
                splitkv_single_cta_kernel<scalar_t><<<grid, block, 0, stream>>>(
                    out.data_ptr<scalar_t>(), out_accum.data_ptr<float>(),
                    softmax_lse.data_ptr<float>(), softmax_lse_accum.data_ptr<float>(),
                    num_splits, batch, num_heads, seqlen_q, head_size, head_size_accum,
                    out.stride(0), out.stride(1), out.stride(2),
                    out_accum.stride(0), out_accum.stride(1), out_accum.stride(2), out_accum.stride(3), out_accum.stride(4),
                    softmax_lse_accum.stride(0), softmax_lse_accum.stride(1), softmax_lse_accum.stride(2), softmax_lse_accum.stride(3)
                );
            }
        );

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
} // namespace FLASH_NAMESPACE
