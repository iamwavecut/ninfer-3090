# AI Farm GPU2 deployment

This profile serves the verified `qwen3_8_27b.ninfer` artifact on the dedicated RTX 3090 while
leaving the Q4_K_M llama.cpp embedder resident. It uses vision-enabled C1 with overlay vision
residency (the Vision tower lives in pinned host memory and encodes through device memory borrowed
from temporarily evicted read-only text weights), a shared 181,376-token RotorQuant `rk8v4` KV pool
(the +61,376 tokens over the previous 120K profile are the reclaimed resident Vision weights,
encode workspace, and `--vision-max-merged`-bounded output transient),
MTP3, the optimized draft head, CUDA Graphs, and compatible-prefix reuse. The single active request
can use a 181,376-token prompt-plus-output window; additional requests wait in the bounded pending
queue instead of reserving a second generation lane. `rk8v4` is a lossy KV-cache
format and trades some output fidelity for the larger context window.

Copy `.env.example` to an untracked `.env`, replace the source/image placeholders, verify the model
SHA-256, then run:

```bash
docker compose build ninfer-3090
docker compose up -d ninfer-3090
./register-discovery.sh
./smoke-protocols.sh
```

The service is published on loopback, the explicit AI Farm LAN address, and the AI Farm Tailscale
address. Discovery reaches it as `http://ninfer-3090:8080`; agent clients use either routed address.
The model catalog, generation, and state endpoints intentionally accept requests with no API key or
with any key value for compatibility with mixed OpenAI and Anthropic clients. The container runs as
the host service account so its capability-dropped process can append to `deploy/logs/`. CORS is
disabled; vision is enabled.
