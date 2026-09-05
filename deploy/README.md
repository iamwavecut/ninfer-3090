# AI Farm GPU2 deployment

This profile serves the verified `qwen3_8_27b.ninfer` artifact on the dedicated RTX 3090 while
leaving the Q4_K_M llama.cpp embedder resident. It uses vision-enabled C1 with overlay vision
residency (the Vision tower lives in pinned host memory and encodes through device memory borrowed
from temporarily evicted read-only text weights, or from free KV pages when they cover the window),
a shared 203,200-token RotorQuant `rk8v4` KV pool (the largest explicit capacity that boots with
this profile; the earlier 120K profile paid for resident Vision weights, encode workspace, and the
`--vision-max-merged`-bounded output transient), MTP3, the optimized draft head, CUDA Graphs, and
compatible-prefix reuse. The single active request can use a 203,200-token prompt-plus-output
window; additional requests wait in the bounded pending queue instead of reserving a second
generation lane. `rk8v4` is a lossy KV-cache format and trades some output fidelity for the larger
context window. `context-cost-presets.json` feeds the planner the RTX 3090's measured prefill and
transfer costs (see `docs/performance.md`); without it the compiled RTX 5090 profile undervalues
retained context by more than 2x and Host KV offload never pays off.

Host memory is shared with draw-api, asr-api and vLLM on this 62 GiB box. The service pins 4 GiB
of Host KV (parks ~520k `rk8v4` tokens of finished prefixes for reuse), two Host StateImage slots
(~147 MiB each), 256 MiB of media cache and 512 MiB of live media buffers, on top of the ~1.7 GiB
the process itself needs: about 7 GiB of RSS.

Copy `.env.example` to an untracked `.env`, replace the source/image placeholders, verify the model
SHA-256, then run:

```bash
docker compose build ninfer-3090
docker compose up -d ninfer-3090
./register-discovery.sh
./smoke-protocols.sh
```

The AI Farm does not compile NInfer. Build `ninfer` and `ninfer-serve` on a rented RTX 3090 pod
from the exact commit with the Dockerfile's configuration (`Release`, apps on, tests and
benchmarks off, CUDA 13.1 toolkit on Ubuntu 24.04), boot the production command line there, copy
the two binaries into `deploy/prebuilt/`, and assemble the image on the farm without a compiler:

```bash
docker build -f deploy/Dockerfile.prebuilt -t "$NINFER_IMAGE" .
docker compose up -d --no-build ninfer-3090
```

`Dockerfile.prebuilt` is the runtime stage of the root `Dockerfile` with the binaries copied from
`deploy/prebuilt/` instead of the build stage.

The service is published on loopback, the explicit AI Farm LAN address, and the AI Farm Tailscale
address. Discovery reaches it as `http://ninfer-3090:8080`; agent clients use either routed address.
The model catalog, generation, and state endpoints intentionally accept requests with no API key or
with any key value for compatibility with mixed OpenAI and Anthropic clients. The container runs as
the host service account so its capability-dropped process can append to `deploy/logs/`. CORS is
disabled; vision is enabled.
