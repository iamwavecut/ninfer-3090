#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.

#include "core/arena.h"
#include "core/device.h"
#include "core/evictable_weight_pool.h"
#include <ninfer/targets/qwen3_6/vision.h>
#include "targets/qwen3_6/impl/runtime/vision_context.h"
#include "targets/qwen3_6/impl/runtime/vision_overlay.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
namespace overlay_detail {

using OverlayClock = std::chrono::steady_clock;

inline std::size_t staging_align(std::size_t bytes) {
    constexpr std::size_t kAlign = 256;
    return (bytes + kAlign - 1) / kAlign * kAlign;
}

inline Tensor rebase(Tensor tensor, std::ptrdiff_t delta) {
    tensor.data = static_cast<std::byte*>(tensor.data) + delta;
    return tensor;
}

inline Weight rebase(Weight weight, std::ptrdiff_t delta) {
    const auto shift = [delta](const void* pointer) -> const void* {
        return pointer == nullptr
                   ? nullptr
                   : static_cast<const void*>(static_cast<const std::byte*>(pointer) + delta);
    };
    weight.payload = shift(weight.payload);
    weight.qdata   = shift(weight.qdata);
    weight.qhigh   = shift(weight.qhigh);
    weight.scales  = shift(weight.scales);
    return weight;
}

inline qwen3_6::VisionLayerWeights rebase_layer(const qwen3_6::VisionLayerWeights& source,
                                                std::ptrdiff_t delta) {
    qwen3_6::VisionLayerWeights out = source;
    out.qkv                         = rebase(source.qkv, delta);
    out.qkv_bias                    = rebase(source.qkv_bias, delta);
    out.output                      = rebase(source.output, delta);
    out.output_bias                 = rebase(source.output_bias, delta);
    out.fc1                         = rebase(source.fc1, delta);
    out.fc1_bias                    = rebase(source.fc1_bias, delta);
    out.fc2                         = rebase(source.fc2, delta);
    out.fc2_bias                    = rebase(source.fc2_bias, delta);
    out.norm1_weight                = rebase(source.norm1_weight, delta);
    out.norm1_bias                  = rebase(source.norm1_bias, delta);
    out.norm2_weight                = rebase(source.norm2_weight, delta);
    out.norm2_bias                  = rebase(source.norm2_bias, delta);
    return out;
}

} // namespace overlay_detail

// Workspace plan of one window: the encode scratch of the planned merged-token budget followed by
// the item handoff, both inside the borrowed extent.
inline VisionWorkspacePlan window_workspace_plan(const VisionWorkspacePlan& planned) {
    return VisionContext::plan_workspace(planned.max_merged_tokens, planned.encode_peak_bytes);
}

// Device bytes one window borrows: the streamed weight staging plus the window workspace.
inline std::size_t window_bytes(const qwen3_6::VisionOverlayLayout& layout,
                                const VisionWorkspacePlan& planned) {
    return overlay_detail::staging_align(layout.staging_bytes) +
           window_workspace_plan(planned).capacity_bytes;
}

// Streams the vision tower from the pinned block through borrowed device staging: a fixed prelude
// region (patch/position embedding), a fixed merger region, and two layer slots refilled on the
// transfer stream one layer ahead of compute. Synchronization is event-based; the host never
// blocks between layers.
class VisionWeightStream {
public:
    VisionWeightStream(DeviceContext& device, const qwen3_6::VisionOverlayAssets& assets,
                       std::byte* staging)
        : device_(device), assets_(assets) {
        using overlay_detail::staging_align;
        const qwen3_6::VisionOverlayLayout& layout = assets.layout;
        prelude_                                   = staging;
        merger_                                    = prelude_ + staging_align(layout.prelude.bytes);
        slot_[0]                                   = merger_ + staging_align(layout.merger.bytes);
        slot_[1]                                   = slot_[0] + staging_align(layout.slot_bytes);
        for (cudaEvent_t& event : uploaded_) {
            CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaEventCreateWithFlags(&prelude_event_, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&merger_event_, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&compute_fence_, cudaEventDisableTiming));

        const std::byte* block = assets.pinned_block.data();
        CUDA_CHECK(cudaMemcpyAsync(prelude_, block + layout.prelude.offset, layout.prelude.bytes,
                                   cudaMemcpyHostToDevice, copy_stream()));
        CUDA_CHECK(cudaEventRecord(prelude_event_, copy_stream()));
        CUDA_CHECK(cudaMemcpyAsync(merger_, block + layout.merger.offset, layout.merger.bytes,
                                   cudaMemcpyHostToDevice, copy_stream()));
        CUDA_CHECK(cudaEventRecord(merger_event_, copy_stream()));
        upload_bytes_ = layout.prelude.bytes + layout.merger.bytes;
        reset(device_.stream);
    }

    ~VisionWeightStream() {
        // The window drains both streams before closing; the events are idle here.
        for (cudaEvent_t event : uploaded_) { (void)cudaEventDestroy(event); }
        (void)cudaEventDestroy(prelude_event_);
        (void)cudaEventDestroy(merger_event_);
        (void)cudaEventDestroy(compute_fence_);
    }

    VisionWeightStream(const VisionWeightStream&)            = delete;
    VisionWeightStream& operator=(const VisionWeightStream&) = delete;

    // Rebased view of the host weights: every layer's tensors point at the slot that holds the
    // layer once arrive(layer) admits it.
    [[nodiscard]] qwen3_6::VisionWeights window_weights(const qwen3_6::VisionWeights& host) const {
        using overlay_detail::rebase;
        using overlay_detail::rebase_layer;
        const qwen3_6::VisionOverlayLayout& layout = assets_.layout;
        const std::byte* block                     = assets_.pinned_block.data();
        const auto delta_to = [block](const std::byte* target, std::size_t source_offset) {
            return target - (block + source_offset);
        };

        qwen3_6::VisionWeights out         = host;
        const std::ptrdiff_t prelude_delta = delta_to(prelude_, layout.prelude.offset);
        out.common.patch_embedding         = rebase(host.common.patch_embedding, prelude_delta);
        out.common.patch_embedding_bias = rebase(host.common.patch_embedding_bias, prelude_delta);
        out.common.position_embedding   = rebase(host.common.position_embedding, prelude_delta);
        for (std::size_t layer = 0; layer < host.common.layers.size(); ++layer) {
            const std::ptrdiff_t delta = delta_to(slot_[layer % 2], layout.layers[layer].offset);
            out.common.layers[layer]   = rebase_layer(host.common.layers[layer], delta);
        }
        const std::ptrdiff_t merger_delta = delta_to(merger_, layout.merger.offset);
        out.common.merger_fc1             = rebase(host.common.merger_fc1, merger_delta);
        out.common.merger_fc1_bias        = rebase(host.common.merger_fc1_bias, merger_delta);
        out.common.merger_norm_weight     = rebase(host.common.merger_norm_weight, merger_delta);
        out.common.merger_norm_bias       = rebase(host.common.merger_norm_bias, merger_delta);
        out.merger_fc2                    = rebase(host.merger_fc2, merger_delta);
        out.merger_fc2_bias               = rebase(host.merger_fc2_bias, merger_delta);
        return out;
    }

    // Prepare an encode pass: uploads of layers 0 and 1 are issued after everything already
    // submitted on the compute stream.
    void reset(cudaStream_t compute) {
        CUDA_CHECK(cudaEventRecord(compute_fence_, compute));
        CUDA_CHECK(cudaStreamWaitEvent(copy_stream(), compute_fence_, 0));
        next_upload_ = 0;
        upload_next_layer();
        upload_next_layer();
    }

    void prelude_ready(cudaStream_t compute) {
        CUDA_CHECK(cudaStreamWaitEvent(compute, prelude_event_, 0));
    }

    void merger_ready(cudaStream_t compute) {
        CUDA_CHECK(cudaStreamWaitEvent(compute, merger_event_, 0));
    }

    // Called at the top of the encoder loop for `layer`: gates compute on the slot upload, then
    // refills the slot the previous layer just vacated. Every op of layers below `layer` is
    // already issued, so the fence orders the refill after them.
    void arrive(std::uint32_t layer, cudaStream_t compute) {
        CUDA_CHECK(cudaStreamWaitEvent(compute, uploaded_[layer % 2], 0));
        if (next_upload_ == layer + 1 &&
            next_upload_ < static_cast<std::uint32_t>(VisionScheduleConfig::layers)) {
            CUDA_CHECK(cudaEventRecord(compute_fence_, compute));
            CUDA_CHECK(cudaStreamWaitEvent(copy_stream(), compute_fence_, 0));
            upload_next_layer();
        }
    }

    [[nodiscard]] std::size_t uploaded_bytes() const noexcept { return upload_bytes_; }

private:
    [[nodiscard]] cudaStream_t copy_stream() const noexcept { return device_.transfer_stream; }

    void upload_next_layer() {
        if (next_upload_ >= static_cast<std::uint32_t>(VisionScheduleConfig::layers)) { return; }
        const std::uint32_t layer                  = next_upload_++;
        const qwen3_6::VisionOverlayLayout& layout = assets_.layout;
        CUDA_CHECK(cudaMemcpyAsync(slot_[layer % 2],
                                   assets_.pinned_block.data() + layout.layers[layer].offset,
                                   layout.layers[layer].bytes, cudaMemcpyHostToDevice,
                                   copy_stream()));
        CUDA_CHECK(cudaEventRecord(uploaded_[layer % 2], copy_stream()));
        upload_bytes_ += layout.layers[layer].bytes;
    }

    DeviceContext& device_;
    const qwen3_6::VisionOverlayAssets& assets_;
    std::byte* prelude_        = nullptr;
    std::byte* merger_         = nullptr;
    std::byte* slot_[2]        = {nullptr, nullptr};
    cudaEvent_t uploaded_[2]   = {nullptr, nullptr};
    cudaEvent_t prelude_event_ = nullptr;
    cudaEvent_t merger_event_  = nullptr;
    cudaEvent_t compute_fence_ = nullptr;
    std::uint32_t next_upload_ = 0;
    std::size_t upload_bytes_  = 0;
};

// Encodes one vision item per window: borrow the window extent from the weight tail, stream the
// tower through it, land the merged embeddings in the session's pinned slot, restore the tail.
// Runs inside the prefill unit of the single worker; the broker rejects nested windows.
class VisionOverlaySession {
public:
    VisionOverlaySession(DeviceContext& device, VisionResidencyBroker& broker,
                         const qwen3_6::VisionOverlayAssets& assets,
                         const VisionWorkspacePlan& planned, PinnedResultPool::Handle result)
        : device_(device), broker_(broker), assets_(assets),
          window_plan_(window_workspace_plan(planned)), result_(std::move(result)) {
        if (window_bytes(assets_.layout, planned) > broker_.window_capacity_bytes()) {
            throw std::logic_error("vision overlay window plan exceeds the pool window capacity");
        }
        if (result_.bytes().size() < window_plan_.handoff_capacity_bytes) {
            throw std::logic_error("pinned vision result slot is smaller than the item handoff");
        }
    }

    // Returns the pinned BF16 [out_hidden, merged] embeddings of the item; valid until the next
    // encode_item call.
    [[nodiscard]] std::span<const std::byte>
    encode_item(const VisionItemView& item, const qwen3_6::VisionItemControl& control) {
        using overlay_detail::OverlayClock;
        using overlay_detail::staging_align;
        const std::size_t staging = staging_align(assets_.layout.staging_bytes);
        // Size the window for this item, not for the planned maximum: fewer chunks to remap and
        // fewer bytes to restore. The planned window bounds it.
        const VisionWorkspacePlan item_plan = VisionContext::plan_workspace(
            control.merged_count,
            VisionContext::workspace_bytes(control.patch_count, control.merged_count));
        if (item_plan.capacity_bytes > window_plan_.capacity_bytes) {
            throw std::invalid_argument("vision item exceeds the overlay window budget");
        }
        const std::size_t need  = staging + item_plan.capacity_bytes;
        const auto window_start = OverlayClock::now();
        EvictableWeightPool::Transaction transaction = broker_.acquire(need);
        auto* const base       = static_cast<std::byte*>(transaction.leased().data);
        const DeviceSpan backing{base + staging, transaction.leased().bytes - staging};
        std::size_t result_bytes = 0;
        {
            VisionWeightStream stream(device_, assets_, base);
            const qwen3_6::VisionWeights view = stream.window_weights(assets_.host.weights);
            const VisionContext context(device_, view);
            Tensor output = VisionContext::bind_output(backing, item_plan, control.merged_count);
            context.encode(item, output, backing, item_plan, &stream);
            result_bytes = output.bytes();
            if (result_bytes > result_.bytes().size()) {
                throw std::logic_error("vision item embeddings exceed the pinned result slot");
            }
            CUDA_CHECK(cudaMemcpyAsync(result_.bytes().data(), output.data, result_bytes,
                                       cudaMemcpyDeviceToHost, device_.stream));
            CUDA_CHECK(cudaStreamSynchronize(device_.stream));
            CUDA_CHECK(cudaStreamSynchronize(device_.transfer_stream));
            stats_.staged_bytes += stream.uploaded_bytes();
        }
        transaction.close();
        if (broker_.poisoned()) {
            throw std::runtime_error(
                "vision overlay restore failed; the text weights are no longer trustworthy");
        }
        const EvictableWeightPool::TransactionStats& costs = transaction.stats();
        stats_.window_seconds +=
            std::chrono::duration<double>(OverlayClock::now() - window_start).count();
        stats_.evict_seconds += costs.evict_seconds;
        stats_.restore_seconds += costs.restore_seconds;
        stats_.evicted_bytes += costs.mapped_bytes;
        stats_.windows += 1;
        return {result_.bytes().data(), result_bytes};
    }

    [[nodiscard]] const VisionOverlayWindowStats& stats() const noexcept { return stats_; }

private:
    DeviceContext& device_;
    VisionResidencyBroker& broker_;
    const qwen3_6::VisionOverlayAssets& assets_;
    VisionWorkspacePlan window_plan_;
    PinnedResultPool::Handle result_;
    VisionOverlayWindowStats stats_;
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
