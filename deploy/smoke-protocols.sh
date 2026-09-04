#!/usr/bin/env bash
set -euo pipefail

base_url="${NINFER_BASE_URL:-http://127.0.0.1:18086}"
api_key="${DIALOG_API_KEY:-any}"

curl --fail --silent --show-error "${base_url}/health" | jq -e '.status == "ok"' >/dev/null
curl --fail --silent --show-error "${base_url}/v1/models" \
  | jq -e '.data[0].id == "qwen3.8-27b"' >/dev/null
curl --fail --silent --show-error "${base_url}/v1/models/qwen3.8-27b" \
  | jq -e '.id == "qwen3.8-27b"' >/dev/null

for path in /v1/chat/completions /v1/responses /v1/messages; do
  status="$(curl --silent --show-error --output /dev/null --write-out '%{http_code}' \
    --request POST "${base_url}${path}")"
  # An empty body is invalid, but must reach protocol validation instead of an auth rejection.
  test "${status}" = "400"
done

curl --fail --silent --show-error "${base_url}/v1/chat/completions" \
  --header "Authorization: Bearer ${api_key}" \
  --header 'content-type: application/json' \
  --data-binary '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Reply with exactly: chat-ok"}],"max_tokens":32,"enable_thinking":false,"temperature":0}' \
  | jq -e '.choices[0].message.content | length > 0' >/dev/null

curl --fail --silent --show-error "${base_url}/v1/responses" \
  --header "Authorization: Bearer ${api_key}" \
  --header 'content-type: application/json' \
  --data-binary '{"model":"qwen3.8-27b","input":"Reply with exactly: responses-ok","max_output_tokens":32,"reasoning":{"effort":"none"},"store":false}' \
  | jq -e '.object == "response" and (.output | length > 0)' >/dev/null

curl --fail --silent --show-error "${base_url}/v1/messages" \
  --header "x-api-key: ${api_key}" \
  --header 'anthropic-version: 2023-06-01' \
  --header 'content-type: application/json' \
  --data-binary '{"model":"qwen3.8-27b","max_tokens":32,"thinking":{"type":"disabled"},"messages":[{"role":"user","content":"Reply with exactly: messages-ok"}]}' \
  | jq -e '.type == "message" and (.content | length > 0)' >/dev/null

printf 'OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages: ok\n'
