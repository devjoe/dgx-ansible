#!/usr/bin/env bash
set -Eeuo pipefail

candidate="${1:-redhat-nvfp4}"
mode="${2:-smoke}"

case "$candidate" in
  google-bf16)
    model_id="${DIFFUSIONGEMMA_MODEL:-google/diffusiongemma-26B-A4B-it}"
    ;;
  nvidia-nvfp4)
    model_id="${DIFFUSIONGEMMA_MODEL:-nvidia/diffusiongemma-26B-A4B-it-NVFP4}"
    ;;
  redhat-nvfp4)
    model_id="${DIFFUSIONGEMMA_MODEL:-RedHatAI/diffusiongemma-26B-A4B-it-NVFP4}"
    ;;
  redhat-fp8)
    model_id="${DIFFUSIONGEMMA_MODEL:-RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic}"
    ;;
  *)
    echo "unknown candidate: $candidate" >&2
    exit 2
    ;;
esac

image="${DIFFUSIONGEMMA_IMAGE:-vllm/vllm-openai:gemma-aarch64-cu130}"
name="${DIFFUSIONGEMMA_CONTAINER:-diffusiongemma-candidate}"
served_name="${DIFFUSIONGEMMA_SERVED_NAME:-diffusiongemma}"
host="${DIFFUSIONGEMMA_HOST:-127.0.0.1}"
port="${DIFFUSIONGEMMA_PORT:-8001}"
outdir="${DIFFUSIONGEMMA_OUTDIR:-/home/devjoe/Projects/Ollama/benchmarks/diffusiongemma-20260611}"
max_model_len="${DIFFUSIONGEMMA_MAX_MODEL_LEN:-4096}"
max_batched_tokens="${DIFFUSIONGEMMA_MAX_BATCHED_TOKENS:-8192}"
gpu_util="${DIFFUSIONGEMMA_GPU_UTIL:-0.85}"
timeout_steps="${DIFFUSIONGEMMA_TIMEOUT_STEPS:-240}"
max_seqs="${DIFFUSIONGEMMA_MAX_SEQS:-1}"
extra_args="${DIFFUSIONGEMMA_EXTRA_ARGS:-}"

mkdir -p "$outdir"
log="$outdir/${candidate}-container.log"
models_json="$outdir/${candidate}-models.json"
smoke_json="$outdir/${candidate}-smoke.json"

cleanup() {
  docker logs --tail 1200 "$name" >"$log" 2>&1 || true
  docker rm -f "$name" >/dev/null 2>&1 || true
  sudo systemctl start vllm >/dev/null 2>&1 || true
  sudo systemctl start vllm-pna-proxy >/dev/null 2>&1 || true
}
trap cleanup EXIT

sudo systemctl stop vllm-pna-proxy >/dev/null 2>&1 || true
sudo systemctl stop vllm >/dev/null 2>&1 || true
docker rm -f "$name" >/dev/null 2>&1 || true

extra_argv=()
if [[ -n "$extra_args" ]]; then
  # Intentionally shell-split operator-provided vLLM flags.
  read -r -a extra_argv <<<"$extra_args"
fi

docker run -d \
  --name "$name" \
  --gpus all \
  --ipc=host \
  --network host \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" \
  -v /home/devjoe/.cache/huggingface:/root/.cache/huggingface \
  -v /home/devjoe/.cache/vllm:/root/.cache/vllm \
  "$image" \
  "$model_id" \
  --host "$host" \
  --port "$port" \
  --served-model-name "$served_name" \
  --gpu-memory-utilization "$gpu_util" \
  --max-model-len "$max_model_len" \
  --max-num-seqs "$max_seqs" \
  --max-num-batched-tokens "$max_batched_tokens" \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --generation-config vllm \
  --no-enable-log-requests \
  "${extra_argv[@]}"

ready=0
for _ in $(seq 1 "$timeout_steps"); do
  if [[ -z "$(docker ps --filter name="$name" --quiet)" ]]; then
    echo "DIFFUSIONGEMMA_EXITED_EARLY"
    docker logs --tail 260 "$name" || true
    exit 10
  fi
  if curl -fsS "http://$host:$port/v1/models" >"$models_json" 2>"$outdir/${candidate}-curl.err"; then
    ready=1
    break
  fi
  sleep 5
done

if [[ "$ready" != "1" ]]; then
  echo "DIFFUSIONGEMMA_NOT_READY"
  docker logs --tail 360 "$name" || true
  exit 11
fi

echo "DIFFUSIONGEMMA_READY"
cat "$models_json"

curl -fsS "http://$host:$port/v1/chat/completions" \
  --max-time "${DIFFUSIONGEMMA_COMPLETION_TIMEOUT:-180}" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$served_name\",\"messages\":[{\"role\":\"system\",\"content\":\"請用繁體中文回答，不要輸出思考過程。\"},{\"role\":\"user\",\"content\":\"用一句話說明你是 DiffusionGemma 測試服務。\"}],\"max_tokens\":96,\"temperature\":0}" \
  >"$smoke_json"

python3 -m json.tool "$smoke_json" | sed -n "1,140p"

if [[ "$mode" == "bench" ]]; then
  bench_input_len="${DIFFUSIONGEMMA_BENCH_INPUT_LEN:-2048}"
  bench_output_len="${DIFFUSIONGEMMA_BENCH_OUTPUT_LEN:-256}"
  bench_concurrency="${DIFFUSIONGEMMA_BENCH_CONCURRENCY:-1}"
  bench_json="$outdir/${candidate}-bench-i${bench_input_len}-o${bench_output_len}-c${bench_concurrency}.json"
  /home/devjoe/Projects/vllm/.venv/bin/vllm bench serve \
    --backend openai \
    --base-url "http://$host:$port" \
    --model "$served_name" \
    --tokenizer "$model_id" \
    --dataset-name random \
    --num-prompts "${DIFFUSIONGEMMA_BENCH_PROMPTS:-4}" \
    --num-warmups "${DIFFUSIONGEMMA_BENCH_WARMUPS:-1}" \
    --input-len "$bench_input_len" \
    --output-len "$bench_output_len" \
    --ignore-eos \
    --max-concurrency "$bench_concurrency" \
    --temperature 0 \
    --disable-tqdm \
    --save-result \
    --result-dir "$outdir" \
    --result-filename "$(basename "$bench_json")"
fi
