#!/usr/bin/env bash
set -euo pipefail

discovery_url="${DISCOVERY_URL:-http://127.0.0.1:50051}"

curl --fail --silent --show-error \
  --request POST "${discovery_url}/v1/services/register" \
  --header 'content-type: application/json' \
  --data-binary '{
    "service": {
      "name": "llm-openai-qwen38-ninfer",
      "base_url": "http://ninfer-3090:8080",
      "execution_model": "SERVICE_EXECUTION_MODEL_SYNC",
      "endpoints": [
        {
          "name": "chat_completions",
          "method": "HTTP_METHOD_POST",
          "path": "/v1/chat/completions",
          "timeout_ms": 570000
        }
      ],
      "default_timeout_ms": 570000,
      "max_concurrent_jobs": 1,
      "enabled": true
    },
    "upsert": true
  }'
printf '\n'
