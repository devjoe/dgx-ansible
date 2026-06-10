#!/usr/bin/env bash
set -Eeuo pipefail

profile="${1:-base}"
spec_tokens="${2:-0}"
mode="${3:-smoke}"
port="${PRISMA_SMOKE_PORT:-8011}"
model_dir="${PRISMA_MODEL_DIR:-/home/devjoe/Projects/Ollama/models/qwen3.6-35b-prismaquant}"
draft_dir="${PRISMA_DRAFT_DIR:-/home/devjoe/Projects/Ollama/models/qwen3.6-35b-dflash}"
workdir="${VLLM_WORKDIR:-/home/devjoe/Projects/vllm}"
outdir="${PRISMA_OUTDIR:-/home/devjoe/Projects/Ollama/benchmarks/prismaquant-revival-20260611}"
served_name="${PRISMA_SERVED_NAME:-qwen36-prisma}"
max_model_len="${PRISMA_MAX_MODEL_LEN:-4096}"
max_batched_tokens="${PRISMA_MAX_BATCHED_TOKENS:-8192}"
gpu_util="${PRISMA_GPU_UTIL:-0.80}"
timeout_steps="${PRISMA_TIMEOUT_STEPS:-180}"

mkdir -p "$outdir"
log="$outdir/${profile}-spec${spec_tokens}-serve.log"
models_json="$outdir/${profile}-spec${spec_tokens}-models.json"
completion_json="$outdir/${profile}-spec${spec_tokens}-completion.json"

pid=""
cleanup() {
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    sleep 3
    kill -KILL "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  fi
  sudo systemctl start vllm >/dev/null 2>&1 || true
  sudo systemctl start vllm-pna-proxy >/dev/null 2>&1 || true
}
trap cleanup EXIT

if [[ ! -d "$model_dir" ]]; then
  echo "missing model dir: $model_dir" >&2
  exit 2
fi
if [[ "$profile" == "dflash" && ! -d "$draft_dir" ]]; then
  echo "missing draft dir: $draft_dir" >&2
  exit 2
fi

sudo systemctl stop vllm-pna-proxy >/dev/null 2>&1 || true
sudo systemctl stop vllm >/dev/null 2>&1 || true
sleep 5

cd "$workdir"
cmd=(
  "$workdir/.venv/bin/vllm" serve "$model_dir"
  --host 127.0.0.1
  --port "$port"
  --served-model-name "$served_name"
  --limit-mm-per-prompt '{"image":1}'
  --max-model-len "$max_model_len"
  --max-num-batched-tokens "$max_batched_tokens"
  --max-num-seqs 1
  --gpu-memory-utilization "$gpu_util"
  --trust-remote-code
  --enable-prefix-caching
  --enable-chunked-prefill
  --generation-config vllm
  --no-enable-log-requests
)

if [[ "$profile" == "dflash" ]]; then
  cmd+=(--speculative-config "{\"method\":\"dflash\",\"model\":\"$draft_dir\",\"num_speculative_tokens\":$spec_tokens}")
fi

rm -f "$log" "$models_json" "$completion_json"
echo "launch: ${cmd[*]}" | tee "$log"
env_args=(
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
  VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}" \
  VLLM_MARLIN_USE_ATOMIC_ADD="${VLLM_MARLIN_USE_ATOMIC_ADD:-1}"
)
if [[ -n "${VLLM_USE_V1:-}" ]]; then
  env_args+=(VLLM_USE_V1="$VLLM_USE_V1")
fi
setsid env "${env_args[@]}" "${cmd[@]}" >>"$log" 2>&1 &
pid="$!"

ready=0
for _ in $(seq 1 "$timeout_steps"); do
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "PRISMA_EXITED_EARLY"
    tail -n 180 "$log" || true
    exit 10
  fi
  if curl -fsS "http://127.0.0.1:$port/v1/models" >"$models_json" 2>"$outdir/curl.err"; then
    ready=1
    break
  fi
  sleep 5
done

if [[ "$ready" != "1" ]]; then
  echo "PRISMA_NOT_READY"
  tail -n 220 "$log" || true
  exit 11
fi

echo "PRISMA_READY"
cat "$models_json"

curl -fsS "http://127.0.0.1:$port/v1/completions" \
  --max-time "${PRISMA_COMPLETION_TIMEOUT:-180}" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$served_name\",\"prompt\":\"Write one short sentence about speculative decoding.\",\"max_tokens\":32,\"temperature\":0}" \
  >"$completion_json"

python3 -m json.tool "$completion_json" | sed -n '1,100p'

if [[ "$mode" == "bench" ]]; then
  bench_json="$outdir/${profile}-spec${spec_tokens}-bench-2k-o256-c1.json"
  "$workdir/.venv/bin/vllm" bench serve \
    --backend openai \
    --base-url "http://127.0.0.1:$port" \
    --model "$served_name" \
    --tokenizer "$model_dir" \
    --dataset-name random \
    --num-prompts "${PRISMA_BENCH_PROMPTS:-6}" \
    --num-warmups "${PRISMA_BENCH_WARMUPS:-1}" \
    --input-len "${PRISMA_BENCH_INPUT_LEN:-2048}" \
    --output-len "${PRISMA_BENCH_OUTPUT_LEN:-256}" \
    --ignore-eos \
    --max-concurrency "${PRISMA_BENCH_CONCURRENCY:-1}" \
    --temperature 0 \
    --disable-tqdm \
    --save-result \
    --result-dir "$outdir" \
    --result-filename "$(basename "$bench_json")"
fi

echo "PRISMA_LOG_TAIL"
tail -n 100 "$log" || true
