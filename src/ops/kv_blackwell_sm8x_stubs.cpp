// SM8X stubs for the NVFP4 and K8V4 KV-cache kernels, which use the Blackwell-only
// mma .kind::f8f6f4 encoding. Those storages are rejected at startup under
// NINFER_SM8X_COMPAT, so none of these can be reached on a correctly configured engine.

#include "ops/kv_cache/append/launch.h"
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

[[noreturn]] void reject_blackwell_kv() {
    throw std::runtime_error("NVFP4 and K8V4 KV caches require Blackwell mma support (sm_120a)");
}

} // namespace

void kv_cache_append_nvfp4_launch(const Tensor&, const Tensor&, const Tensor&, PagedKVLayerView,
                                  cudaStream_t) {
    reject_blackwell_kv();
}

void kv_cache_append_nvfp4_batch_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                        const Tensor&, PagedKVBatchLayerView, cudaStream_t) {
    reject_blackwell_kv();
}

void kv_cache_append_k8v4_launch(const Tensor&, const Tensor&, const Tensor&, PagedKVLayerView,
                                 cudaStream_t) {
    reject_blackwell_kv();
}

void kv_cache_append_k8v4_batch_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                       const Tensor&, PagedKVBatchLayerView, cudaStream_t) {
    reject_blackwell_kv();
}

void causal_attention_small_t_nvfp4_launch(const Tensor&, const Tensor&, const Tensor&,
                                           const Tensor&, const Tensor&, const Tensor&, float,
                                           PagedKVBatchLayerView, CausalAttentionExecutionEnvelope,
                                           std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&,
                                           Tensor&, cudaStream_t) {
    reject_blackwell_kv();
}

void causal_attention_cached_small_t_nvfp4_launch(const Tensor&, const Tensor&, float,
                                                  const PagedKVLayerView&,
                                                  CausalAttentionExecutionEnvelope, Tensor&,
                                                  Tensor&, Tensor&, Tensor&, cudaStream_t) {
    reject_blackwell_kv();
}

void causal_attention_small_t_k8v4_launch(const Tensor&, const Tensor&, const Tensor&,
                                          const Tensor&, const Tensor&, const Tensor&, float,
                                          PagedKVBatchLayerView, CausalAttentionExecutionEnvelope,
                                          std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&,
                                          Tensor&, cudaStream_t) {
    reject_blackwell_kv();
}

void causal_attention_cached_small_t_k8v4_launch(const Tensor&, const Tensor&, float,
                                                 const PagedKVLayerView&,
                                                 CausalAttentionExecutionEnvelope, Tensor&,
                                                 Tensor&, Tensor&, Tensor&, cudaStream_t) {
    reject_blackwell_kv();
}

void causal_attention_prompt_nvfp4_launch(const Tensor&, const Tensor&, const Tensor&,
                                          const Tensor&, const Tensor&, const Tensor&, float,
                                          PagedKVBatchLayerView, Tensor&, cudaStream_t) {
    reject_blackwell_kv();
}

void causal_attention_prompt_nvfp4_attention_launch(const Tensor&, const Tensor&, float,
                                                    const PagedKVLayerView&, Tensor&,
                                                    cudaStream_t) {
    reject_blackwell_kv();
}

void causal_attention_prompt_k8v4_launch(const Tensor&, const Tensor&, const Tensor&,
                                         const Tensor&, const Tensor&, const Tensor&, float,
                                         PagedKVBatchLayerView, Tensor&, cudaStream_t) {
    reject_blackwell_kv();
}

void causal_attention_prompt_k8v4_attention_launch(const Tensor&, const Tensor&, float,
                                                   const PagedKVLayerView&, Tensor&,
                                                   cudaStream_t) {
    reject_blackwell_kv();
}

} // namespace ninfer::ops::detail
