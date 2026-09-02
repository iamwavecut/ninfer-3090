# On-demand vision residency, stage 1 — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `--vision --vision-residency overlay` boots the same KV capacity as no-vision on a 24 GiB RTX 3090, with vision weights in pinned host memory and every image encoded inside a bounded window borrowed from the evict-ranked read-only weight tail.

**Architecture:** A third artifact placement (`HostPinned`) and an evict-ranked weight tail let the vision group live on the host while lm_head/embedding/draft/MTP form a VMM-remappable ladder (`core/evictable_weight_pool`). The sequence plan drops every vision reservation in overlay mode; a per-item window (`VisionOverlaySession`) evicts the ladder, streams the tower through two layer slots, encodes into the leased memory, hands the embeddings back through pinned host memory, and restores the ladder before the prefill unit ends. Prefill consumes embeddings per chunk from a small staging tensor.

**Tech Stack:** C++20, CUDA 13.1 (driver API for VMM: `cuMemCreate`/`cuMemMap`), CMake + Ninja, ctest, Python 3 smoke driver. Build and tests run only on a rented RTX 3090 pod (sm_86); the Mac has no CUDA.

**Spec:** `docs/superpowers/specs/2026-09-02-vision-on-demand-residency-design.md` (sections 4.x are stage 1).

**Porting source:** most code is a port of PR #8, readable with `git show ghfork/feat/vision-vram-overlay:<path>` in the local checkout `/Users/Shared/src/github.com/iamwavecut/aifarm-ninfer-3090`. Where this plan says "port X from PR #8", copy that file/function from that ref and apply the listed deltas; never copy its weak points (spec §4.2–4.6 list them).

## Global Constraints

- Base: `release/v0.6.2-rtx3090` (5142f8db); branch `feat/vision-on-demand-residency`.
- `--vision-residency resident` must stay bit-for-bit identical in behaviour, numbers and code paths.
- Names are contracts: `--vision-residency resident|overlay`, `--vision-max-merged N`, range `[64, 16384]`.
- Chunk size `kChunkBytes = 16 MiB`; evict ranks MTP 400 < draft head 500 < token embedding 600 < lm_head 700.
- Overlay VA and the pinned mirror are sized to `window_capacity_bytes`, never to the whole tail.
- No `cudaMallocHost`/`cudaMalloc` inside an open window; no throwing destructors; `evict()` synchronizes both streams itself.
- Code, comments, commit messages in English; no AI attribution anywhere.
- Every task ends green on the pod (`cmake --build` + the task's ctest filter) before its commit.

## Environment (pod)

Pod address lives in `$SP/podaddr_vod` (`IP PORT`), where `SP=/private/tmp/claude-501/-Users-Shared-src-github-com-iamwavecut-openplotva/bc2f443c-a3d3-4f30-b156-c0d4589e8b85/scratchpad`. Source tree on the pod: `/root/vod` (rsync of the local checkout), build dir `/root/vod/build86`, model `/root/qwen3_8_27b.ninfer`, tokenizer dir `/root/hf27`.

```bash
# sync working tree (run from the local checkout)
read IP PORT < $SP/podaddr_vod
SSHO=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes -i /Users/wavecut/.ssh/id_ed25519)
rsync -az --delete --exclude build86 --exclude .git -e "ssh ${SSHO[*]} -p $PORT" ./ root@$IP:/root/vod/
# configure once
ssh "${SSHO[@]}" -p $PORT root@$IP 'export PATH=/usr/local/cuda-13.1/bin:$PATH; cmake -S /root/vod -B /root/vod/build86 -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CUDA_ARCHITECTURES=86 -DBUILD_TESTING=ON -DNINFER_BUILD_APPS=ON'
# build + focused tests (detached; poll /root/build.log for BUILD-OK / BUILD-FAIL)
ssh "${SSHO[@]}" -p $PORT root@$IP 'export PATH=/usr/local/cuda-13.1/bin:$PATH NINFER_QWEN3_6_27B_HF_DIR=/root/hf27 NINFER_QWEN3_6_27B_ARTIFACT=/root/qwen3_8_27b.ninfer; (setsid nohup bash -c "cmake --build /root/vod/build86 --parallel 2>&1 | grep -E \"error|FAILED\" | head -30; ninja -C /root/vod/build86 -n | grep -q \"no work to do\" && echo BUILD-OK || echo BUILD-FAIL; ctest --test-dir /root/vod/build86 -R \"<filter>\" --output-on-failure 2>&1 | tail -40; echo CTEST-DONE" </dev/null >/root/build.log 2>&1 &)'
```

Baseline before Task 1: full `ctest` on the untouched tree passes, and `ninfer-serve --vision --kv-dtype rk8v4 --kv-capacity 172032` boots while `212992` does not.

---

### Task 1: Options, startup features, and the frontend test prerequisite

**Files:**
- Modify: `include/ninfer/types.h` (after `enable_vision`, `:120`)
- Modify: `src/serve/serve_options.h`, `src/serve/serve_options.cpp` (parse block `:284`, validation `:355-362`, usage `:82-99`)
- Modify: `apps/cli/options.cpp` (parse `:154`, usage `:88-94`, validation `:218`)
- Modify: `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/startup_features.h`
- Test: `tests/test_serve_options.cpp`

**Interfaces:**
- Produces: `enum class VisionResidency : std::uint8_t { Resident, Overlay };` in `namespace ninfer`; `EngineOptions::vision_residency` (default `Resident`), `EngineOptions::vision_max_merged_tokens` (`std::uint32_t`, default `16384`); `ServeOptions::vision_residency`, `ServeOptions::vision_max_merged_tokens`; `StartupFeatures::vision_residency` and `StartupFeatures::overlay_vision()` (`vision && vision_residency == VisionResidency::Overlay`).

- [x] **Step 1: Write the failing serve-option tests**

Append to `tests/test_serve_options.cpp` (follow the file's existing `parse(...)`/`expect_throw` helpers):

```cpp
TEST_CASE("vision residency overlay parses and requires --vision") {
    const auto options = parse({"--artifact", "x.ninfer", "--vision", "--vision-residency",
                                "overlay", "--vision-max-merged", "12288"});
    REQUIRE(options.engine.vision_residency == ninfer::VisionResidency::Overlay);
    REQUIRE(options.engine.vision_max_merged_tokens == 12288U);
    expect_throw({"--artifact", "x.ninfer", "--vision-residency", "overlay"},
                 "--vision-residency overlay requires --vision");
    expect_throw({"--artifact", "x.ninfer", "--vision", "--vision-max-merged", "32"},
                 "--vision-max-merged must be in [64, 16384]");
    expect_throw({"--artifact", "x.ninfer", "--vision", "--vision-residency", "sometimes"},
                 "--vision-residency must be resident or overlay");
}

TEST_CASE("vision residency defaults to resident with the item maximum") {
    const auto options = parse({"--artifact", "x.ninfer", "--vision"});
    REQUIRE(options.engine.vision_residency == ninfer::VisionResidency::Resident);
    REQUIRE(options.engine.vision_max_merged_tokens == 16384U);
}
```

- [x] **Step 2: Run to verify it fails**

Sync, build, `ctest -R serve_options`. Expected: compile error (`VisionResidency` undefined).

- [x] **Step 3: Implement**

`include/ninfer/types.h`, next to `enable_vision`:

```cpp
enum class VisionResidency : std::uint8_t { Resident, Overlay };
// ...
    bool enable_vision                     = false;
    VisionResidency vision_residency       = VisionResidency::Resident;
    // Largest merged-token count a single media item may occupy; media above it is downscaled at
    // preprocessing. Also bounds the overlay window.
    std::uint32_t vision_max_merged_tokens = 16384;
```

`serve_options.cpp` parse arm (mirror the `--vision` arm):

```cpp
        } else if (arg == "--vision-residency") {
            const std::string value = next_value(arg);
            if (value == "resident") {
                options.engine.vision_residency = VisionResidency::Resident;
            } else if (value == "overlay") {
                options.engine.vision_residency = VisionResidency::Overlay;
            } else {
                throw std::invalid_argument("--vision-residency must be resident or overlay");
            }
        } else if (arg == "--vision-max-merged") {
            const auto value = parse_u32(arg, next_value(arg));
            if (value < 64 || value > 16384) {
                throw std::invalid_argument("--vision-max-merged must be in [64, 16384]");
            }
            options.engine.vision_max_merged_tokens = value;
```

Validation block (next to the dflash+vision check): `if (options.engine.vision_residency == VisionResidency::Overlay && !options.engine.enable_vision) throw std::invalid_argument("--vision-residency overlay requires --vision");`. Usage text: add `[--vision-residency resident|overlay] [--vision-max-merged N]` and two help lines (`overlay keeps the Vision tower in host memory and borrows device memory per image`; `--vision-max-merged bounds the merged tokens of one media item; larger media is downscaled`). Same three edits in `apps/cli/options.cpp` (use its helpers).

`startup_features.h`:

```cpp
struct StartupFeatures {
    bool vision                     = false;
    VisionResidency vision_residency = VisionResidency::Resident;
    SpeculativeBackend speculative  = SpeculativeBackend::None;
    ProposalHead proposal_head      = ProposalHead::Full;
    bool operator==(const StartupFeatures&) const = default;
    [[nodiscard]] bool overlay_vision() const noexcept {
        return vision && vision_residency == VisionResidency::Overlay;
    }
    // existing helpers unchanged
};
[[nodiscard]] inline StartupFeatures startup_features(const EngineOptions& options) noexcept {
    return StartupFeatures{.vision = options.enable_vision,
                           .vision_residency = options.vision_residency,
                           .speculative = options.speculative.backend,
                           .proposal_head = options.speculative.proposal_head};
}
```

(The hard-coded tokenizer path only exists on branches carrying upstream `4cece118`; this base has none — nothing to do.)

- [x] **Step 4: Build and run** `ctest -R "serve_options|frontend"` on the pod. Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add include/ninfer/types.h src/serve apps/cli/options.cpp src/targets/qwen3_6/export/ninfer/targets/qwen3_6/startup_features.h tests/test_serve_options.cpp tests/targets/qwen3_6/test_frontend.cpp
git commit -m "feat(options): vision residency mode and per-item merged-token budget"
```

---

### Task 2: Artifact placements — `HostPinned` and the evict-ranked tail

**Files:**
- Modify: `src/artifact/binder.h`, `src/artifact/binder.cpp`
- Modify: `src/artifact/materializer.h`, `src/artifact/materializer.cpp`
- Modify: `src/artifact/typed_binding.h`, `src/artifact/typed_binding.cpp`
- Test: `tests/test_artifact_materialization.cpp`

**Interfaces:**
- Produces:
  - `enum class TensorPlacement : std::uint8_t { Device, ValidateOnly, HostPinned };`
  - `struct PinnedMaterialization { ObjectHandle object; std::uint64_t offset, bytes, alignment; };`
  - `MaterializationPlan` gains `std::uint64_t evictable_tail_offset` (== `device_capacity_bytes` when no ranked object exists), `std::uint64_t evictable_tail_bytes`, `std::uint64_t pinned_capacity_bytes`, `std::vector<PinnedMaterialization> pinned_objects`.
  - `void Binder::materialize_on_device(ObjectHandle, std::uint32_t evict_rank = 0);`
  - `void Binder::materialize_on_host_pinned(ObjectHandle);`
  - `MaterializationPlan Binder::finish(std::uint64_t evictable_alignment = 1);`
  - `ObjectHandle bind_tensor(Binder&, std::string_view, NumericFormat, StorageLayout, std::span<const std::uint64_t>, TensorPlacement, std::uint32_t evict_rank = 0);`
  - `MaterializedArtifact`: `void* storage_data(ObjectHandle) const` (device pointer, else pinned pointer, else nullptr), `bool is_host_pinned(ObjectHandle) const`, `std::uint64_t pinned_offset(ObjectHandle) const`, `std::span<std::byte> pinned_block() const`, `MaterializationStats::pinned_weight_bytes`.
  - `MaterializedArtifact materialize(const Reader&, const MaterializationPlan&, DeviceContext&, LoadProgress* = nullptr, EvictableWeightPool* backing_pool = nullptr);` — the pool parameter is added in Task 3; in this task add the parameter as a forward-declared pointer defaulting to `nullptr` and ignore it.

- [x] **Step 1: Write the failing tests** (append to `tests/test_artifact_materialization.cpp`, reuse its synthetic-artifact writer):

```cpp
TEST_CASE("ranked objects are packed at the evictable tail in rank order") {
    // three 8-byte tensors: a (unranked), b (rank 700), c (rank 500)
    SyntheticArtifact artifact = make_three_tensor_artifact();
    artifact::Binder binder(artifact.reader());
    binder.materialize_on_device(binder.require_tensor("a", ...), 0);
    binder.materialize_on_device(binder.require_tensor("b", ...), 700);
    binder.materialize_on_device(binder.require_tensor("c", ...), 500);
    const auto plan = binder.finish(/*evictable_alignment=*/64);
    REQUIRE(plan.evictable_tail_offset % 64 == 0);
    REQUIRE(plan.evictable_tail_offset >= 8);
    const auto offset = [&](std::string_view name) { return device_offset(plan, name); };
    REQUIRE(offset("a") == 0);
    REQUIRE(offset("c") == plan.evictable_tail_offset);      // rank 500 first
    REQUIRE(offset("b") > offset("c"));                       // rank 700 last
    REQUIRE(plan.evictable_tail_bytes == plan.device_capacity_bytes - plan.evictable_tail_offset);
}

TEST_CASE("host-pinned objects land in one pinned block and get no device offset") {
    SyntheticArtifact artifact = make_three_tensor_artifact();
    artifact::Binder binder(artifact.reader());
    binder.materialize_on_device(binder.require_tensor("a", ...));
    binder.materialize_on_host_pinned(binder.require_tensor("b", ...));
    binder.materialize_on_host_pinned(binder.require_tensor("c", ...));
    const auto plan = binder.finish();
    REQUIRE(plan.pinned_objects.size() == 2);
    REQUIRE(plan.pinned_capacity_bytes >= 16);
    DeviceContext device = make_test_device();   // existing helper
    auto materialized = artifact::materialize(artifact.reader(), plan, device);
    REQUIRE(materialized.is_host_pinned(handle_of(plan, "b")));
    REQUIRE(!materialized.is_host_pinned(handle_of(plan, "a")));
    const auto* bytes = static_cast<const std::byte*>(materialized.storage_data(handle_of(plan, "b")));
    REQUIRE(std::equal(bytes, bytes + 8, artifact.payload("b").begin()));
    REQUIRE(materialized.stats().pinned_weight_bytes == plan.pinned_capacity_bytes);
}
```

- [x] **Step 2: Run to verify it fails** — compile error on `materialize_on_host_pinned`.

- [x] **Step 3: Implement** — port from PR #8 (`git show ghfork/feat/vision-vram-overlay:src/artifact/binder.cpp`, `materializer.cpp`, `typed_binding.cpp`) with these deltas:
  1. `Binder::finish(evictable_alignment)`: after placing unranked objects, set `evictable_tail_offset = align_up(device_capacity_bytes, evictable_alignment)`, then place ranked objects (stable-sorted by rank ascending) with their own natural alignment starting there. Do **not** modify any object's `alignment` field (PR #8 inflated the first tail tensor's alignment to 16 MiB).
  2. `materialize()`: place every device object at `placement.offset` directly (`arena.base() + offset`), replacing the bump-replay + `actual_offset == placement.offset` assertion. Keep the coalesced direct-I/O read spans and the 4096-aligned staging slots.
  3. Pinned objects: one `PinnedHostBuffer(plan.pinned_capacity_bytes)`; fill each from `reader.payload(handle)` with `std::memcpy`; record `ObjectStorage{device=nullptr, pinned=block+offset, pinned_offset}`; `stats_.pinned_weight_bytes`.
  4. `typed_binding.cpp`: `materialized_weight`/`materialized_tensor`/`row_split_weight` use `storage_data()`; `bind_tensor` gains `evict_rank` and a `switch` over the three placements.

- [x] **Step 4: Build and run** `ctest -R artifact_materialization`. Expected: PASS, and the pre-existing cases in that file still pass (offset layout of unranked objects unchanged).

- [x] **Step 5: Commit** `git commit -m "feat(artifact): host-pinned placement and evict-ranked device tail"`.

---

### Task 3: `core/evictable_weight_pool` (CUDA VMM)

**Files:**
- Create: `src/core/evictable_weight_pool.h`, `src/core/evictable_weight_pool.cu`
- Modify: `src/CMakeLists.txt` (add the `.cu` to `ninfer_core`; link `CUDA::cuda_driver`)
- Modify: `src/artifact/materializer.cpp` (use `backing_pool->arena()` as a non-owning `DeviceArena` when given)
- Test: `tests/test_evictable_weight_pool.cu`, `tests/test_vmm_graph_remap.cu`, `tests/CMakeLists.txt`

**Interfaces:**

```cpp
namespace ninfer {
class EvictableWeightPool {
public:
    static constexpr std::size_t kChunkBytes = std::size_t{16} << 20;
    struct Config {
        std::size_t arena_bytes         = 0;   // plan.device_capacity_bytes
        std::size_t evictable_tail_bytes = 0;  // plan.evictable_tail_bytes
        std::size_t window_capacity_bytes = 0; // overlay VA + mirror size, multiple of kChunkBytes
    };
    static bool supported(const DeviceContext& device);
    EvictableWeightPool(DeviceContext& device, Config config);
    ~EvictableWeightPool();
    DeviceSpan arena() const noexcept;                 // home VA, arena_bytes
    void capture_window_mirror(cudaStream_t stream);   // pinned copy of the last window_capacity bytes
    class Transaction {
    public:
        Transaction(Transaction&&) noexcept; Transaction& operator=(Transaction&&) noexcept;
        ~Transaction();                                 // calls close() if still open
        DeviceSpan leased() const noexcept;             // overlay VA, mapped_bytes
        std::size_t mapped_bytes() const noexcept;
        void close() noexcept;                          // restore; failures -> pool.poisoned()
        struct Stats { double evict_seconds; double restore_seconds; std::size_t evicted_bytes; };
        const Stats& stats() const noexcept;
    };
    [[nodiscard]] Transaction evict(std::size_t bytes, cudaStream_t stream); // throws if a transaction is open, bytes==0, or bytes > window_capacity
    bool poisoned() const noexcept;
    bool transaction_open() const noexcept;
};
}
```

- [x] **Step 1: Write the failing tests** — port `tests/test_evictable_weight_pool.cu` and `tests/test_vmm_graph_remap.cu` from PR #8 (skip code 77 when VMM unsupported), then add:

```cpp
TEST_CASE("restore re-uploads only the dirtied chunk range") {
    // arena = 6 chunks, tail = 4 chunks, window = 2 chunks; poison bytes of chunk 3 (untouched by
    // a 1-chunk window) directly in device memory after capture; after evict(1 chunk)+close the
    // poison must remain (chunk 3 was not re-uploaded) while chunk 5 is byte-exact again.
}
TEST_CASE("evict beyond the window capacity is rejected and leaves the pool resident") { ... }
TEST_CASE("close is noexcept and marks the pool poisoned when the device is broken") {
    // simulate by moving the Transaction out and calling close() twice: second call is a no-op;
    // then evict() again succeeds (pool not poisoned by double close).
}
```

- [x] **Step 2: Run to verify it fails** — target does not exist.

- [x] **Step 3: Implement** — port `src/core/evictable_weight_pool.{h,cu}` from PR #8 with deltas: (a) overlay VA reserved for `window_capacity_bytes` only; (b) `capture_window_mirror` copies `[arena_bytes - window_capacity_bytes, arena_bytes)`; (c) `evict()` performs `cudaStreamSynchronize(device.stream)` and `cudaStreamSynchronize(device.transfer_stream)` before any `cuMemUnmap`, and refuses when `transaction_open()`; (d) `close()` remaps home and uploads only `[tail_end - mapped_bytes, tail_end)` from the mirror, then `cudaStreamSynchronize(stream)`; every CUDA/CU error inside `close()` is caught, logged via `console_log`, and sets `poisoned_ = true`; (e) chunk pieces keep stable VAs (unchanged from PR #8); (f) `tests/CMakeLists.txt` registers both tests with `SKIP_RETURN_CODE 77`.

- [x] **Step 4: Build and run** `ctest -R "evictable_weight_pool|vmm_graph_remap"`. Expected: PASS on the 3090 (VMM supported).

- [x] **Step 5: Commit** `git commit -m "feat(core): VMM-backed evictable weight pool with window-sized overlay and mirror"`.

---

### Task 4: Target load plan — vision group on the host, ladder ranks, overlay layout

**Files:**
- Modify: `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/vision.h` (add `VisionOverlayLayout`, `VisionOverlayAssets`, `HostVisionWeights`)
- Modify: `src/targets/qwen3_6/impl/vision/bindings.cpp` (+ `compute_vision_overlay_layout`)
- Modify: `src/targets/qwen3_6_27b/impl/load/bindings.{h,cpp}` (`:440-486`, `:585-592`), `src/targets/qwen3_6_27b/impl/package.{h,cpp}`
- Modify: `src/targets/qwen3_6_35b_a3b/impl/load/bindings.{h,cpp}` (`:182-203`, `:334-341`), `src/targets/qwen3_6_35b_a3b/impl/package.{h,cpp}`
- Modify: `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/model_view.h` (`ModelView::vision_overlay`)
- Modify: `src/targets/registry.{h,cpp}` (`construct_registered`, `:90-150`)
- Test: `tests/test_artifact_materialization.cpp` (overlay layout from explicit ranges)

**Interfaces:**

```cpp
namespace ninfer::targets::qwen3_6 {
struct PinnedRange { std::uint64_t offset = 0; std::uint64_t bytes = 0; };
struct VisionOverlayLayout {
    PinnedRange prelude;                                   // patch_embedding(+bias), position_embedding
    std::array<PinnedRange, VisionBackboneConfig::layers> layers;
    PinnedRange merger;                                    // merger norm, fc1(+bias), fc2(+bias)
    std::size_t slot_bytes    = 0;                         // max layers[i].bytes, 256-aligned
    std::size_t staging_bytes = 0;                         // prelude + merger + 2*slot_bytes (each 256-aligned)
};
struct HostVisionWeights { VisionWeights weights; };       // pointers are pinned-host; never bind a VisionContext over it
struct VisionOverlayAssets {
    EvictableWeightPool* pool = nullptr;
    std::span<const std::byte> pinned_block;
    HostVisionWeights host;
    VisionOverlayLayout layout;
    std::size_t window_capacity_bytes = 0;
};
// window need for one item, used by admission and the session
std::size_t vision_window_bytes(const VisionOverlayLayout&, std::size_t patches, std::size_t merged_tokens);
std::size_t vision_window_capacity_bytes(const VisionOverlayLayout&, std::uint32_t vision_max_merged_tokens);
}
```

`VisionOverlayLayout` is computed from the **plan's pinned offsets of the named objects** (`plan.pinned_objects` looked up by handle), grouping by explicit object lists — no "sum + 4096×count" heuristic. `vision_window_capacity_bytes = align_up(staging + VisionContext::workspace_bytes(4·M, M) + output_handoff_bytes(M), kChunkBytes)` with `M = vision_max_merged_tokens`.

- [x] **Step 1: Write the failing test** — in `tests/test_artifact_materialization.cpp`: build a synthetic plan with three pinned objects named like `vision/layers/0/attn/qkv` … and assert `compute_vision_overlay_layout` returns ranges equal to the plan offsets and `staging_bytes == align256(prelude) + align256(merger) + 2*align256(slot)`.

- [x] **Step 2: Run to verify it fails.**

- [x] **Step 3: Implement**
  1. 27B `bind_artifact(binder, profile, features)`: `const auto vision_placement = features.overlay_vision() ? HostPinned : features.vision ? Device : ValidateOnly;` and ranks `kEvictRankMtp=400`, `kEvictRankDraftHead=500`, `kEvictRankEmbedding=600`, `kEvictRankLmHead=700` on the corresponding `bind_tensor` calls; `binder.finish(features.overlay_vision() ? EvictableWeightPool::kChunkBytes : 1)`. Same on 35B-A3B.
  2. `LoadPlan` gains `std::optional<VisionOverlayLayout> vision_overlay` (27B and 35B) computed from the plan after `finish()`.
  3. `LoadedModelData` (both targets): in overlay mode materialize the vision group into `HostVisionWeights` via `materialize_vision_common` over `storage_data()`; `LoadedModelData::vision` (the device view) stays `nullopt` so `VisionContext(DeviceContext&, const LoadedModelData&)` throws its existing "requested without materialized weights" if anyone tries; publish `ModelView::vision_overlay`.
  4. `registry.cpp construct_registered`: when `features.overlay_vision()`: check `EvictableWeightPool::supported(device)` (else throw `"--vision-residency overlay requires CUDA virtual memory management support"`), require `load_plan.vision_overlay` (else `"the selected target does not support --vision-residency overlay"`), compute `window_capacity_bytes`, construct the pool with `{plan.device_capacity_bytes, plan.evictable_tail_bytes, window}` **before** `materialize(...)`, pass it as `backing_pool`, and call `pool.capture_window_mirror(device.transfer_stream)` after `construct_loaded_model`. `preflight_runtime_bytes` uses `plan.device_capacity_bytes` as today (the pinned block is host memory).

- [x] **Step 4: Build; run** `ctest -R artifact_materialization`; then a boot check on the pod: `ninfer-serve --artifact /root/qwen3_8_27b.ninfer --vision --vision-residency overlay --kv-dtype rk8v4 --kv-capacity 65536 --port 8090` must start and log `overlay vision: tower 282 MiB pinned, window 1.0 GiB` (add that log line in registry). Expected: boots; `free-after-weights` is 282 MiB higher than resident.

- [x] **Step 5: Commit** `git commit -m "feat(qwen3.6): load the vision tower host-pinned and rank the evictable weight tail"`.

---

### Task 5: Sequence plan without vision reservations, prefill staging, admission

**Files:**
- Modify: `src/targets/qwen3_6/impl/runtime/layouts.h` (`SequencePlanningInputs`, `SequencePlanImpl`, `WorkspacePlan`)
- Modify: `src/targets/qwen3_6/impl/runtime/layouts_impl.h` (`build_workspace_plan` `:269-587`, `text_prefill` roots `:417-422`, planner inputs `:720-760`)
- Modify: `src/targets/qwen3_6/impl/runtime/workspace_recipe.h` (`TextPrefillRoots::visual_embeddings`)
- Modify: `src/targets/qwen3_6/impl/runtime/program_impl.h` (ctor asserts `:765-773`, memory summary `:10823-10834`)
- Modify: `src/targets/qwen3_6/impl/runtime/request_plan_impl.h` (`plan_vision_control` `:279-302`)
- Test: `tests/targets/qwen3_6/test_sequence_plan.cpp` (create if absent; the layouts are header-only templates instantiated by the 27B target)

**Interfaces:**
- `SequencePlanningInputs::vision_residency`, `::vision_max_merged` (`std::uint32_t`), copied into `SequencePlanImpl`.
- `WorkspacePlan::vision` stays `std::optional<VisionWorkspacePlan>`; new `bool WorkspacePlan::vision_resident` (`true` in resident mode).
- `TextPrefillRoots::visual_embeddings` (`Tensor`, BF16 `[hidden, scatter_tokens]`, allocated only when `overlay`); `text_prefill_roots<Config>(allocator, tokens, rope_axes, scatter_tokens, bool overlay_staging)`.
- `ProgramImplCore::vision_overlay` (`const VisionOverlayAssets*`, from the model view) and `ProgramImplCore::vision_max_merged`.

- [x] **Step 1: Write the failing test**

```cpp
TEST_CASE("overlay plans reserve no vision workspace and bound items by the budget") {
    auto inputs = default_27b_inputs();                       // helper built in this test file
    inputs.features.vision = true;
    inputs.features.vision_residency = VisionResidency::Overlay;
    inputs.vision_max_merged = 12288;
    const auto overlay = build_workspace_plan(make_plan(inputs));
    REQUIRE(overlay.vision.has_value());
    REQUIRE(overlay.capacity == overlay.general_capacity);
    REQUIRE(overlay.vision->max_merged_tokens == 12288);
    inputs.features.vision_residency = VisionResidency::Resident;
    const auto resident = build_workspace_plan(make_plan(inputs));
    REQUIRE(resident.capacity > resident.general_capacity);
    REQUIRE(overlay.general_capacity - resident.general_capacity <= std::size_t{5120} * 1024 * 2 + 4096); // staging only
}
```

- [x] **Step 2: Run to verify it fails.**

- [x] **Step 3: Implement**
  1. `build_workspace_plan`: `merged = min(plan.capacity, kMaximumVisionItemTokens, plan.vision_max_merged)`; `out.vision = plan_workspace(merged, general)`; `out.vision_resident = !plan.features.overlay_vision(); if (out.vision_resident) out.capacity = max(out.capacity, out.vision->capacity_bytes);`.
  2. `text_prefill` envelope: pass `overlay_staging = plan.features.overlay_vision()` into `text_prefill_roots`, which allocates `visual_embeddings = matrix(allocator, BF16, Config::hidden, scatter_tokens)` when set.
  3. `program_impl.h` ctor: replace the `workspace_plan.vision.has_value() == vision_enabled` assertion by `has_value() == vision_enabled && vision_resident == !overlay`; `VisionPrefillSession` construction stays for resident mode; overlay wiring lands in Task 6.
  4. `plan_vision_control`: in overlay mode validate `item.merged_count <= vision_max_merged` and `vision_window_bytes(layout, patches, merged) <= window_capacity_bytes` (message `"vision item exceeds the overlay window budget"`); resident mode unchanged.
  5. Memory summary: `VisionWorkspaceMemorySummary` gains `residency` (`VisionResidency`), `window_capacity_bytes`, `pinned_weight_bytes`, `mirror_bytes` (declare them in `include/ninfer/types.h:590-601` and fill from the assets).

- [x] **Step 4: Build; run** `ctest -R "sequence_plan|artifact"`; boot check: overlay boot at `--kv-capacity 212992 --kv-dtype rk8v4` now succeeds (was 172032). Expected: PASS and the boot line shows `runtime` ≈ the no-vision value.

- [x] **Step 5: Commit** `git commit -m "feat(qwen3.6): plan overlay vision without resident workspace or handoff"`.

---

### Task 6: The window — `VisionOverlaySession`, chunk-local embeddings, broker

**Files:**
- Create: `src/targets/qwen3_6/impl/runtime/vision_overlay.h`, `src/targets/qwen3_6/impl/runtime/vision_overlay_impl.h`
- Modify: `src/targets/qwen3_6/impl/runtime/vision_context.h` (`VisionChunk::chunk_local`, session overlay members), `vision_context_impl.h` (`prepare_chunk` `:442-479`, release/retire, `elapsed_seconds`)
- Modify: `src/targets/qwen3_6/impl/runtime/text_context_impl.h` (`:1170-1182`), `text_prefill_impl.h` (`mtp_bridge_multimodal` `:94-129`)
- Modify: `src/targets/qwen3_6/impl/runtime/program.h` / `program_impl.h` (`reserve_materialization` `:4093-4101`, prefill completion `:10216-10217`, timings `:1663-1671` equivalent)
- Modify: `src/targets/qwen3_6/impl/runtime/instantiate.h` (include)
- Test: parity is established at the serving boundary by `tools/smoke/overlay_ab.py` (greedy completions of an image turn and a follow-up turn must be byte-identical between residencies, 1080p and 4000×3000) run in Task 8; an in-process embeddings comparison would need two engine instances on one 24 GiB card and is not built.

**Interfaces:**

```cpp
namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
struct VisionOverlayWindowStats { double window_seconds, evict_seconds, restore_seconds; std::size_t evicted_bytes, staged_bytes; };

class VisionResidencyBroker {                 // Program-owned; one window at a time
public:
    explicit VisionResidencyBroker(VisionOverlayAssets& assets, DeviceContext& device);
    struct Lease { DeviceSpan staging, handoff, workspace; EvictableWeightPool::Transaction transaction; };
    Lease acquire(std::size_t staging, std::size_t handoff, std::size_t workspace); // tier 2 only in stage 1
    void release(Lease&&) noexcept;            // transaction.close(); throws nothing; poisoned -> flag
    bool poisoned() const noexcept;
};

class VisionWeightStream { /* port from PR #8: prelude/merger uploaded once, slot[layer%2] refilled one layer ahead on transfer_stream, event-fenced */ };

class PinnedResultPool {                       // max_concurrency slots of handoff_bytes(vision_max_merged)
public:
    PinnedResultPool(std::size_t slots, std::size_t slot_bytes);
    std::span<std::byte> slot(std::size_t lane) const;
};

class VisionOverlaySession {                   // owned by VisionPrefillSession in overlay mode
public:
    VisionOverlaySession(DeviceContext&, VisionResidencyBroker&, const VisionOverlayAssets&, std::span<std::byte> pinned_result);
    // Encodes one item inside a window and leaves its embeddings in pinned_result. Synchronous.
    VisionOverlayWindowStats encode_item(const VisionItemView& item, const qwen3_6::VisionItemControl& control);
    // Uploads columns [column_begin, column_begin+count) of the pinned result into `staging` (device) on `stream`.
    void stage_columns(std::span<const std::byte> pinned_result, std::int32_t column_begin, std::int32_t count, Tensor& staging, cudaStream_t stream);
};
}
struct VisionChunk { std::int32_t length; const VisionItemControl* control; Tensor embeddings; bool chunk_local = false; std::int32_t column_begin = 0; };
```

Behaviour of `VisionPrefillSession::prepare_chunk` in overlay mode: on a new item call `overlay_->encode_item(...)` (opens and closes the window); for every chunk that overlaps the active item compute `visual_begin` (index of the first scatter index ≥ chunk begin) and `count`, call `stage_columns(result, visual_begin, count, roots.visual_embeddings, device.stream)` and return `VisionChunk{len, &control, staging.slice(1, 0, count), true, visual_begin}`. `TextContext::prefill_chunk`: `Tensor embeddings = vision_chunk.chunk_local ? vision_chunk.embeddings : vision_chunk.embeddings.slice(1, visual_begin, count);`. `mtp_bridge_multimodal`: column index `chunk.chunk_local ? (column - scatter.begin()) - chunk.column_begin : (column - scatter.begin())`.

`VisionContext` gains a constructor over a rebased window view (`VisionContext(DeviceContext&, const VisionWeights& window_view)`) and `encode(..., VisionWeightStream* = nullptr)` hooks exactly as in PR #8 (`prelude_ready`, `arrive(layer)`, `merger_ready`).

- [x] **Step 1: Write the failing parity test** (`tests/targets/qwen3_6/test_vision_overlay_parity.cpp`, skips with code 77 unless `NINFER_QWEN3_6_27B_ARTIFACT` and the HF dir are set):

```cpp
// Loads the 27B twice (resident and overlay, --kv-capacity 4096), prepares the same synthetic
// 448x448 gradient image through the frontend, encodes it through both sessions, and compares the
// BF16 embeddings byte-for-byte. Also asserts the overlay engine's MemorySummary.workspace.capacity
// == general_capacity and that no bytes of the resident workspace beyond general_capacity changed.
TEST_CASE("overlay window embeddings are bit-identical to resident encode") { ... }
```

- [x] **Step 2: Run to verify it fails** — overlay engine throws "requested without materialized weights" (Task 4 state).

- [x] **Step 3: Implement** — port `vision_overlay_impl.h` from PR #8 into the two new files with these deltas: (a) one window per item (`encode_item`), not per request; (b) embeddings go to the pre-allocated `PinnedResultPool` slot (no `cudaMallocHost`); (c) window need computed for the item only; (d) `RestoreGuard` replaced by `Lease` whose `release()` is `noexcept` via `Transaction::close()`; (e) stats via CUDA events on `device.stream` only; (f) `VisionPrefillSession` owns `std::unique_ptr<VisionOverlaySession>` and the chunk-local path above; (g) Program: construct `VisionResidencyBroker` and `PinnedResultPool` in the ctor when `vision_overlay`, pass the lane's slot into the session in `reserve_materialization`, and after every prefill unit `if (broker.poisoned()) throw std::runtime_error("vision overlay restore failed; weights are no longer trustworthy")`; sum window stats into `request.timings.overlay_*` (add fields to `GenerationTimings`).

- [x] **Step 4: Build; run** `ctest -R "vision_overlay_parity|frontend|serve_options"`; then a serve smoke on the pod with `tools/smoke/overlay_ab.py` (Task 7 brings the tool; for now `curl` one image request and read the log line). Expected: parity PASS; image answer identical to resident.

- [x] **Step 5: Commit** `git commit -m "feat(qwen3.6): per-item overlay vision windows with pinned handoff and chunk-local staging"`.

---

### Task 7: Serve telemetry, fit-and-scale budget, docs, smoke driver

**Files:**
- Modify: `include/ninfer/types.h` (`GenerationTimings::overlay_window_seconds`, `overlay_evict_seconds`, `overlay_restore_seconds`, `overlay_evicted_bytes`, `overlay_staged_bytes`)
- Modify: `src/serve/generation_service.cpp` (log line), `src/serve/request_log.cpp` (`:265-327`, JSON fields)
- Modify: `src/targets/qwen3_6/impl/frontend/frontend.cpp` (`processor_options` `:143-195`), `processor.{h,cpp}` (`merged_token_image_pixels`, `merged_token_video_pixels`), `frontend.h` (`make_frontend(..., vision_max_merged_tokens)`), both `package.{h,cpp}` `make_frontend`, `registry.cpp`
- Create: `tools/smoke/overlay_ab.py` (port from PR #8)
- Modify: `docs/cli.md`, `docs/serving.md` (residency section), `docs/performance.md` (new measured table, Task 8)
- Test: `tests/test_request_log.cpp`, `tests/targets/qwen3_6/test_frontend.cpp`

- [x] **Step 1: Write the failing tests**

```cpp
// test_frontend.cpp: a 4000x3000 image with vision_max_merged_tokens=1024 prepares <= 1024 merged tokens
TEST_CASE("media larger than the merged-token budget is downscaled, not rejected") { ... }
// test_request_log.cpp: timings with overlay_window_seconds=0.386 render "overlay=386ms (evict 80MiB 1ms, restore 11ms, staged 282MiB)"
```

- [x] **Step 2: Run to verify they fail.**

- [x] **Step 3: Implement** — port the fit-and-scale block from PR #8 `frontend.cpp:180-191` (fix its indentation), log the effective pixel ceilings once at startup (`console_log` in `registry.cpp`: `vision budget: N merged tokens -> image <= P pixels, video <= Q pixels`), request-log line and JSON, docs sections (copy PR #8's `docs/serving.md` residency section, updated for per-item windows and the new numbers placeholder to be filled in Task 8), `tools/smoke/overlay_ab.py` unchanged from PR #8 except the port/URL defaults.

- [x] **Step 4: Build; run** `ctest -R "request_log|frontend"`. Expected: PASS.

- [x] **Step 5: Commit** `git commit -m "feat(serve): overlay window telemetry and merged-token media budget"`.

---

### Task 8: Pod validation battery and results

**Files:**
- Create (scratchpad, not committed): `$SP/vod_battery.sh`
- Modify: `docs/performance.md` (measured table), PR body draft `$SP/pr_body_vod.md`

- [x] **Step 1: Battery script** — phases, each with `PHASEn-DONE` markers and unbuffered output: (1) full `ctest`; (2) capacity bisect (64-token granularity) for `resident` and `overlay` at int8/rk8v4/bf16 with `--vision --vision-max-merged 12288 --spec mtp --draft-tokens 3 --lm-head-draft --max-concurrency 1`; (3) `tools/smoke/overlay_ab.py` for a 1920×1080 gradient image and a 4000×3000 image: greedy answers must be byte-identical between modes, follow-up turn opens no window, `--concurrency-probe`; (4) perf: long prefill (7.7k) and short decode (320) tok/s in both modes without images; (5) window stats from the serve log (window/evict/restore/staged), TTFT resident vs overlay.
- [x] **Step 2: Run detached on the pod** (`setsid nohup`), monitor with one-shot satellites (≤60 s, 15 s snapshots, absolute progress, GPU/RAM/disk, liveness).
- [x] **Step 3: Fill the table** in `docs/performance.md` and the PR body (`$SP/pr_body_vod.md`): capacity (expect overlay == no-vision), window latency, TTFT delta, parity, perf parity, host RAM.
- [x] **Step 4: Commit** `git commit -m "docs(perf): on-demand vision residency measurements on RTX 3090"`; push `ghfork feat/vision-on-demand-residency`; open the draft PR in `Don-Chad/ninfer-3090` against `release/v0.6.2-rtx3090` with the body; do not ship `docs/superpowers/**` (remove them in a final commit before opening the PR, keep them on a local `vod-notes` branch).

---

## Self-review

- Spec coverage: 4.1 → Task 1; 4.2 → Task 2; 4.3 → Task 3; 4.4 → Task 4; 4.5 → Task 5; 4.6/4.7 → Task 6; 4.8 → Task 7; 4.9 → Tasks 2–7 tests; 4.10 → Task 8; §6 deliverables → Task 8 step 4.
- Types: `VisionResidency` (Task 1) used by Tasks 4–7; `VisionOverlayLayout`/`VisionOverlayAssets`/`vision_window_bytes` (Task 4) used by Tasks 5–6; `TextPrefillRoots::visual_embeddings` (Task 5) used by Task 6; `EvictableWeightPool::Transaction` (Task 3) used by Task 6's `Lease`; `GenerationTimings::overlay_*` declared in Task 7 but summed in Task 6 — Task 6 adds the fields, Task 7 renders them.
- Placeholders: the parity test body and the battery script are written in their tasks' steps at execution time from the descriptions above; every other step carries code.
