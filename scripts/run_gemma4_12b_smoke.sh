#!/usr/bin/env bash
set -Eeuo pipefail

candidate="${1:-coolthor-nvfp4a16}"
mode="${2:-smoke}"

case "$candidate" in
  coolthor-nvfp4a16)
    model_id="${GEMMA4_12B_MODEL:-coolthor/gemma-4-12B-it-NVFP4A16}"
    ;;
  google-qat-w4a16)
    model_id="${GEMMA4_12B_MODEL:-google/gemma-4-12B-it-qat-w4a16-ct}"
    ;;
  google-bf16)
    model_id="${GEMMA4_12B_MODEL:-google/gemma-4-12B-it}"
    ;;
  *)
    echo "unknown candidate: $candidate" >&2
    exit 2
    ;;
esac

image="${GEMMA4_12B_IMAGE:-vllm/vllm-openai:gemma4-unified-arm64-cu130}"
name="${GEMMA4_12B_CONTAINER:-gemma4-12b-candidate}"
served_name="${GEMMA4_12B_SERVED_NAME:-gemma4-12b}"
host="${GEMMA4_12B_HOST:-127.0.0.1}"
port="${GEMMA4_12B_PORT:-8001}"
outdir="${GEMMA4_12B_OUTDIR:-/home/devjoe/Projects/Ollama/benchmarks/gemma4-12b-replace-20260611}"
max_model_len="${GEMMA4_12B_MAX_MODEL_LEN:-4096}"
max_batched_tokens="${GEMMA4_12B_MAX_BATCHED_TOKENS:-8192}"
gpu_util="${GEMMA4_12B_GPU_UTIL:-0.85}"
timeout_steps="${GEMMA4_12B_TIMEOUT_STEPS:-240}"
language_model_only="${GEMMA4_12B_LANGUAGE_MODEL_ONLY:-0}"

mkdir -p "$outdir"
log="$outdir/${candidate}-container.log"
models_json="$outdir/${candidate}-models.json"
smoke_json="$outdir/${candidate}-smoke.json"

cleanup() {
  docker logs --tail 1000 "$name" >"$log" 2>&1 || true
  docker rm -f "$name" >/dev/null 2>&1 || true
  sudo systemctl start vllm >/dev/null 2>&1 || true
  sudo systemctl start vllm-pna-proxy >/dev/null 2>&1 || true
}
trap cleanup EXIT

sudo systemctl stop vllm-pna-proxy >/dev/null 2>&1 || true
sudo systemctl stop vllm >/dev/null 2>&1 || true
docker rm -f "$name" >/dev/null 2>&1 || true

language_model_args=()
if [[ "$language_model_only" == "1" ]]; then
  language_model_args+=(--language-model-only)
fi

docker run -d \
  --name "$name" \
  --gpus all \
  --ipc=host \
  --network host \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" \
  -e VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}" \
  -v /home/devjoe/.cache/huggingface:/root/.cache/huggingface \
  -v /home/devjoe/.cache/vllm:/root/.cache/vllm \
  "$image" \
  "$model_id" \
  --host "$host" \
  --port "$port" \
  --served-model-name "$served_name" \
  --gpu-memory-utilization "$gpu_util" \
  --max-model-len "$max_model_len" \
  --max-num-seqs 1 \
  --max-num-batched-tokens "$max_batched_tokens" \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --generation-config vllm \
  --no-enable-log-requests \
  "${language_model_args[@]}"

ready=0
for _ in $(seq 1 "$timeout_steps"); do
  if [[ -z "$(docker ps --filter name="$name" --quiet)" ]]; then
    echo "GEMMA4_12B_EXITED_EARLY"
    docker logs --tail 220 "$name" || true
    exit 10
  fi
  if curl -fsS "http://$host:$port/v1/models" >"$models_json" 2>"$outdir/${candidate}-curl.err"; then
    ready=1
    break
  fi
  sleep 5
done

if [[ "$ready" != "1" ]]; then
  echo "GEMMA4_12B_NOT_READY"
  docker logs --tail 300 "$name" || true
  exit 11
fi

echo "GEMMA4_12B_READY"
cat "$models_json"

curl -fsS "http://$host:$port/v1/chat/completions" \
  --max-time "${GEMMA4_12B_COMPLETION_TIMEOUT:-180}" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$served_name\",\"messages\":[{\"role\":\"system\",\"content\":\"請用繁體中文回答。\"},{\"role\":\"user\",\"content\":\"用一句話說明你是 Gemma 4 12B 測試服務。\"}],\"max_tokens\":64,\"temperature\":0}" \
  >"$smoke_json"

python3 -m json.tool "$smoke_json" | sed -n "1,120p"

if [[ "$mode" == "bench" ]]; then
  bench_json="$outdir/${candidate}-bench-2k-o256-c1.json"
  /home/devjoe/Projects/vllm/.venv/bin/vllm bench serve \
    --backend openai \
    --base-url "http://$host:$port" \
    --model "$served_name" \
    --tokenizer "$model_id" \
    --dataset-name random \
    --num-prompts "${GEMMA4_12B_BENCH_PROMPTS:-4}" \
    --num-warmups "${GEMMA4_12B_BENCH_WARMUPS:-1}" \
    --input-len "${GEMMA4_12B_BENCH_INPUT_LEN:-2048}" \
    --output-len "${GEMMA4_12B_BENCH_OUTPUT_LEN:-256}" \
    --ignore-eos \
    --max-concurrency "${GEMMA4_12B_BENCH_CONCURRENCY:-1}" \
    --temperature 0 \
    --disable-tqdm \
    --save-result \
    --result-dir "$outdir" \
    --result-filename "$(basename "$bench_json")"
fi
