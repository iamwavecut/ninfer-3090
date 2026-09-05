// ninfer::ops::detail - public-op composition for group-16 NVFP4 causal prompt attention.
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include "ops/kv_cache/append/launch.h"
// Upstream's prompt kernel is warp-specialized (setmaxnreg) and compiled in the sm_120a-only
// non-RDC unit. sm_86/sm_89 cannot run it, so the SM8X build carries a portable, plainly
// tiled kernel with the same tile shape as the FP8 and K8V4 prompt paths.
#if defined(NINFER_SM8X_COMPAT)
#include "core/device.h"
#include "ops/common/math.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_nvfp4_sm8x.cuh"
#include <cstdint>
#else
#include "ops/softmax_attention/dense/causal_cache/prompt_nvfp4_non_rdc_launch.h"
#endif

namespace ninfer::ops::detail {
#if defined(NINFER_SM8X_COMPAT)
namespace {

template <typename Geometry, typename CacheView, typename Metadata>
void causal_attention_prompt_nvfp4_attention_launch_for(const Tensor& q, const Tensor& positions,
                                                        float scale, const CacheView& cache,
                                                        Metadata metadata, Tensor& out,
                                                        cudaStream_t stream) {
    static const cudaError_t attr = cudaFuncSetAttribute(
        causal_attention_prompt_nvfp4_kernel<Geometry, Metadata>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kCausalPromptNvfp4SmemBytes);
    CUDA_CHECK(attr);

    const auto tokens = static_cast<std::int32_t>(q.ne[2]);
    const dim3 grid(static_cast<unsigned>(div_up(tokens, kCausalPromptNvfp4Br)),
                    static_cast<unsigned>(Geometry::QHeads), 1u);
    causal_attention_prompt_nvfp4_kernel<Geometry, Metadata>
        <<<grid, kCausalPromptNvfp4Threads, kCausalPromptNvfp4SmemBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const std::uint8_t*>(cache.k_pages.data),
            static_cast<const std::uint8_t*>(cache.v_pages.data),
            static_cast<const std::uint8_t*>(cache.k_scale_pages.data),
            static_cast<const std::uint8_t*>(cache.v_scale_pages.data), metadata,
            static_cast<const std::int32_t*>(positions.data), scale,
            static_cast<__nv_bfloat16*>(out.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

template <typename CacheView, typename Metadata>
void causal_attention_prompt_nvfp4_attention_dispatch(const Tensor& q, const Tensor& positions,
                                                      float scale, const CacheView& cache,
                                                      Metadata metadata, Tensor& out,
                                                      cudaStream_t stream) {
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        causal_attention_prompt_nvfp4_attention_launch_for<CausalD256H24Kv4>(
            q, positions, scale, cache, metadata, out, stream);
        return;
    }
    causal_attention_prompt_nvfp4_attention_launch_for<CausalD256H16Kv2>(q, positions, scale, cache,
                                                                         metadata, out, stream);
}

} // namespace

void causal_attention_prompt_nvfp4_attention_launch(const Tensor& q, const Tensor& positions,
                                                    float scale, const PagedKVLayerView& cache,
                                                    Tensor& out, cudaStream_t stream) {
    const PagedKVDirectMetadata metadata{static_cast<const std::int32_t*>(cache.block_table.data)};
    causal_attention_prompt_nvfp4_attention_dispatch(q, positions, scale, cache, metadata, out,
                                                     stream);
}

void causal_attention_prompt_nvfp4_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                          const Tensor& positions, const Tensor& valid_columns,
                                          const Tensor& table_rows, float scale,
                                          PagedKVBatchLayerView cache, Tensor& out,
                                          cudaStream_t stream) {
    kv_cache_append_batch_launch(k, v, positions, valid_columns, table_rows, cache, stream);
    const auto launch = [&]<bool Masked>() {
        const PagedKVBatchMetadata<Masked> metadata{
            .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride = cache.block_tables.ne[0],
        };
        causal_attention_prompt_nvfp4_attention_dispatch(q, positions, scale, cache, metadata, out,
                                                         stream);
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

#else

void causal_attention_prompt_nvfp4_attention_launch(const Tensor& q, const Tensor& positions,
                                                    float scale, const PagedKVLayerView& cache,
                                                    Tensor& out, cudaStream_t stream) {
    causal_attention_prompt_nvfp4_kernel_launch(q, positions, scale, cache, out, stream);
}

void causal_attention_prompt_nvfp4_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                          const Tensor& positions, const Tensor& valid_columns,
                                          const Tensor& table_rows, float scale,
                                          PagedKVBatchLayerView cache, Tensor& out,
                                          cudaStream_t stream) {
    kv_cache_append_batch_launch(k, v, positions, valid_columns, table_rows, cache, stream);
    causal_attention_prompt_nvfp4_batch_kernel_launch(q, positions, valid_columns, table_rows,
                                                      scale, cache, out, stream);
}

#endif
} // namespace ninfer::ops::detail
