#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.

#include "core/arena.h"
#include "core/device.h"
#include "core/evictable_weight_pool.h"
#include <ninfer/targets/qwen3_6/vision.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {

struct VisionOverlayWindowStats {
    double window_seconds     = 0.0;
    double evict_seconds      = 0.0;
    double restore_seconds    = 0.0;
    std::size_t evicted_bytes = 0;
    std::size_t staged_bytes  = 0;
    std::uint32_t windows     = 0;
};

// Program-owned arbiter of the single overlay window a device can hold. Every window is served
// from the evict-ranked weight tail, which makes it exclusive by construction: the text weights
// it borrows are unmapped until the window closes.
class VisionResidencyBroker {
public:
    VisionResidencyBroker(DeviceContext& device, EvictableWeightPool& pool) noexcept
        : device_(device), pool_(pool) {}

    [[nodiscard]] EvictableWeightPool::Transaction acquire(std::size_t bytes) {
        if (pool_.transaction_open()) {
            throw std::logic_error("a vision overlay window is already open");
        }
        return pool_.evict(bytes, device_.stream);
    }

    [[nodiscard]] bool poisoned() const noexcept { return pool_.poisoned(); }
    [[nodiscard]] std::size_t window_capacity_bytes() const noexcept {
        return pool_.window_capacity_bytes();
    }

private:
    DeviceContext& device_;
    EvictableWeightPool& pool_;
};

// Pinned slots that receive the embeddings of one item each; a session holds one slot for its
// lifetime, so at most max_concurrency slots are ever needed and no window allocates host memory.
class PinnedResultPool {
public:
    class Handle {
    public:
        Handle() noexcept = default;
        ~Handle() { release(); }
        Handle(const Handle&)            = delete;
        Handle& operator=(const Handle&) = delete;
        Handle(Handle&& other) noexcept
            : pool_(other.pool_), index_(other.index_), bytes_(other.bytes_) {
            other.pool_ = nullptr;
        }
        Handle& operator=(Handle&& other) noexcept {
            if (this != &other) {
                release();
                pool_       = other.pool_;
                index_      = other.index_;
                bytes_      = other.bytes_;
                other.pool_ = nullptr;
            }
            return *this;
        }

        [[nodiscard]] std::span<std::byte> bytes() const noexcept { return bytes_; }

    private:
        friend class PinnedResultPool;
        Handle(PinnedResultPool& pool, std::size_t index, std::span<std::byte> bytes) noexcept
            : pool_(&pool), index_(index), bytes_(bytes) {}
        void release() noexcept {
            if (pool_ != nullptr) {
                pool_->free_.push_back(index_);
                pool_ = nullptr;
            }
        }

        PinnedResultPool* pool_ = nullptr;
        std::size_t index_      = 0;
        std::span<std::byte> bytes_;
    };

    PinnedResultPool(std::size_t slots, std::size_t slot_bytes) {
        if (slots == 0 || slot_bytes == 0) {
            throw std::invalid_argument("pinned vision result pool needs at least one slot");
        }
        buffers_.reserve(slots);
        free_.reserve(slots);
        for (std::size_t index = 0; index < slots; ++index) {
            buffers_.push_back(std::make_unique<PinnedHostBuffer>(slot_bytes));
            free_.push_back(index);
        }
    }

    [[nodiscard]] Handle acquire() {
        if (free_.empty()) {
            throw std::logic_error("pinned vision result pool has no free slot");
        }
        const std::size_t index = free_.back();
        free_.pop_back();
        PinnedHostBuffer& buffer = *buffers_[index];
        return Handle(*this, index,
                      std::span<std::byte>(static_cast<std::byte*>(buffer.data()), buffer.size()));
    }

    [[nodiscard]] std::size_t slot_bytes() const noexcept {
        return buffers_.empty() ? 0 : buffers_.front()->size();
    }

private:
    std::vector<std::unique_ptr<PinnedHostBuffer>> buffers_;
    std::vector<std::size_t> free_;
};

class VisionWeightStream;
class VisionOverlaySession;

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
