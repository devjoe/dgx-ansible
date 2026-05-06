# dgx-ansible

Declarative management of the NVIDIA DGX Spark (`gx10.local` / `192.168.99.2`)
running **Ollama (Tier A)** and **vLLM (Tier B)** for the [fb-reader](https://github.com/devjoe/fb-reader)
Chrome extension. Companion repo to `~/Projects/local-inference/` (which holds
the Mac side + cross-host benchmark scripts + the canonical model-choice docs).

All host state — systemd env vars, firewall rules, model list, vLLM serving
knobs — lives in **`group_vars/dgx.yml`**. Run `make deploy` to converge.

## Agent quick start

Stop after #1 unless the task needs more:

1. **This README** — current state + open work + the deploy/benchmark commands.
2. **`group_vars/dgx.yml`** — single source of truth for every knob. If you
   want to change something, edit here, then `make deploy`.
3. **`roles/{ollama,vllm,benchmark}/tasks/main.yml`** — what `make deploy` and
   `make benchmark` actually execute, in order.
4. **`docs/dgx-spark-vllm-model-selection-2026-05-06.md`** — pick the best
   shared DGX vLLM model for fb-reader + OpenCode (Qwen3.6 FP8 vs Gemma4+assistant).
5. **`docs/handover-prismaquant.md`** — paused investigation into a
   95 tok/s PrismaQuant + DFlash config. Read only if reviving that thread.

For Mac-side config, model-choice rationale, or cross-endpoint benchmarks, go
to `~/Projects/local-inference/README.md`.

## Current state (2026-05-06)

| | Tier A (Ollama) | Tier B (vLLM) |
|---|---|---|
| Endpoint | `gx10.local:11434` | `gx10.local:8000` |
| Deployed model | `qwen3.6:35b-a3b` (MoE, ~36 tok/s) | `Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound` (+ DFlash) |
| Role | text classification fallback / quality backstop | multimodal Tier B classifier |
| Service unit | `ollama` (systemd) | `vllm` (systemd) |
| Image handling | n/a | client-side prefetch in fb-reader → `data:image/jpeg;base64,...` |
| Speed flags | `FLASH_ATTENTION=0`, `KV_CACHE_TYPE=fp16` | Marlin atomic add, `gpu-memory-utilization=0.85` |
| Keep-alive | `OLLAMA_KEEP_ALIVE=24h` | n/a (vLLM holds the model in VRAM permanently) |

The image-sanitizing proxy (`vllm-sanitizer`) was **removed 2026-04** in commit
`ed898fd`. fb-reader now fetches + JPEG-re-encodes images client-side against
the user's authenticated FB session. vLLM has been simplified to a single
service bound directly on `0.0.0.0:8000`.

## Open work / next steps

### Observability (v1 shipped 2026-05-04 — collecting baseline)

`roles/observability/` ships VictoriaMetrics + Grafana + DCGM exporter
+ node_exporter + vmagent + vLLM scrape + canary systemd timer. Stand
it up with:

```bash
cp group_vars/dgx.yml.vault.example group_vars/dgx.yml.vault
ansible-vault encrypt group_vars/dgx.yml.vault   # set Grafana admin pw
echo "<your-vault-password>" > .vault_pass
make deploy-obs
make status-obs
make canary-once       # trigger the canary service immediately to backfill
```

Grafana lands at `http://gx10.local:3000` (anonymous viewer; admin
login uses the password from the vault). Dashboard `Inference
Overview (v1)` is auto-loaded by the provisioner — it pulls from
`local-inference/obs/dashboards/` at deploy time.

**v1 success target**: 7 consecutive days of canary runs visible
on the dashboard. Once we have ~7 days of σ-baseline, v2 wires
Telegram alerts (decode tok/s drop > 10%; cold-load absolute thresholds;
any-error triggers). v2 sequencing is in
`~/Projects/local-inference/docs/observability-design-2026-05.md`.

### Tracking

- **PrismaQuant + DFlash investigation paused** — see `docs/handover-prismaquant.md`.
  Resume only when there's a clean V0-engine flag set for `compressed-tensors`
  on Blackwell.
- **Bench model drift**: `bench_model: qwen3.5:latest` in `group_vars/dgx.yml`,
  but the deployed primary is `qwen3.6:35b-a3b`. Decide whether benchmark
  should track the deployed model or stay on qwen3.5 as a stable A/B baseline.

## Quick reference

```bash
make help                  # list targets
make install-deps          # install ansible-galaxy collections
make ping                  # SSH + sudo reach
make deploy                # converge Ollama + vLLM to group_vars state
make status                # GET /api/ps (Ollama loaded models)
make status-vllm           # systemctl is-active vllm + GET /v1/models
make benchmark             # 3-run timed eval against Ollama → tok/s + JSON
make benchmark-vllm        # text + data-URI image regression check
make benchmark-vllm-perf   # vLLM perf matrix (prefill/decode × concurrency)
make unload                # POST keep_alive:0 to free VRAM
make lint                  # ansible --syntax-check on all playbooks
make deploy-obs            # stand up obs stack (needs .vault_pass + dgx.yml.vault)
make status-obs            # systemctl state of every observability unit
make canary-once           # trigger the obs canary service immediately
```

## vLLM speed spot-check (Mac-side)

For a quick latency sanity check from your Mac, use the bundled script:

```bash
python3 scripts/run_vllm_classification.py --api-base http://gx10.local:8000/v1 --model qwen3.6-35b
```

If you see a suspicious ~5s fixed latency per request, it is usually a
client-side networking issue (macOS/Python routing differences between IPv4
and IPv6 for `.local` hostnames). Retry:

```bash
python3 scripts/run_vllm_classification.py --connect-mode auto
```

`ASK_BECOME=1` prefix forces an interactive sudo password prompt (use it on
the very first `make deploy` before NOPASSWD sudo is set up):

```bash
ASK_BECOME=1 make deploy
```

## A/B testing speed flags

`OLLAMA_FLASH_ATTENTION` and `OLLAMA_KV_CACHE_TYPE` are well-known to regress
on some hardware × model combos
([ollama#12432](https://github.com/ollama/ollama/issues/12432),
[#11949](https://github.com/ollama/ollama/issues/11949),
[#9683](https://github.com/ollama/ollama/issues/9683),
[#6769](https://github.com/ollama/ollama/issues/6769)). They default to OFF on
the DGX. To benchmark on/off:

```bash
make deploy && make benchmark               # baseline (off)
sed -i '' 's/ollama_flash_attention: false/ollama_flash_attention: true/' group_vars/dgx.yml
make deploy && make benchmark               # FA on
git checkout group_vars/dgx.yml && make deploy   # revert
```

Each `make deploy` re-renders `/etc/systemd/system/ollama.service.d/override.conf`
and restarts Ollama exactly once when the file actually changes. Same flag has
**opposite verdicts** on Mac (`FA=1` wins) vs DGX (`FA=1` regresses) — never
blindly copy flags between hosts. See
`~/Projects/local-inference/docs/endpoints.md` for evidence.

## Repo layout

```
dgx-ansible/
├── README.md                   # ← you are here
├── ansible.cfg                 # ssh + output + roles_path
├── inventory.ini               # [dgx] gx10.local
├── requirements.yml            # community.general, ansible.posix
├── site.yml                    # main playbook (ollama + vllm roles)
├── benchmark.yml               # measure-only Ollama playbook
├── benchmark-vllm.yml          # vLLM up-check + data-URI regression
├── group_vars/
│   ├── dgx.yml                 # SOURCE OF TRUTH (env, models, vLLM knobs, obs)
│   └── dgx.yml.vault.example   # template for encrypted secrets (Grafana pw, Telegram)
├── playbooks/
│   └── deploy-observability.yml  # observability v1 standalone playbook
├── roles/
│   ├── ollama/                 # install + systemd + firewall + model pulls
│   ├── vllm/                   # venv + systemd + firewall + health check
│   ├── benchmark/              # unload, warm, time eval calls
│   └── observability/          # VM + Grafana + exporters + canary timer (v1)
├── docs/
│   └── handover-prismaquant.md # paused investigation (resume notes)
└── Makefile                    # deploy / benchmark / status / deploy-obs / lint
```

Everything under `.archive/` is dead code kept for reference only — safe to
ignore. `scratch/` (gitignored) holds external clones (e.g. ray-project/llmperf
for one-off load tests).

## What `make deploy` does

1. Install Ollama from the upstream script (skipped if already installed).
2. Render `/etc/systemd/system/ollama.service.d/override.conf` from
   `roles/ollama/templates/ollama-override.conf.j2` using `group_vars/dgx.yml`.
3. Open UFW port `11434` from `lan_cidr`.
4. Start/enable the `ollama` systemd unit; restart it if the override changed.
5. Pull any models in `ollama_models` not already present (multi-GB, async
   with a 30 min ceiling).
6. Mirror Ansible-managed state into the operator workdir `~/Projects/Ollama/`
   on the DGX (override mirror, README, `benchmarks/` dir).
7. Assert `GET /api/tags` returns 200 and contains every requested model.
8. Provision the vLLM venv, render `/etc/systemd/system/vllm.service`, open
   the LAN port, start the unit, wait for `/v1/models` to return 200.

## What `make benchmark` does (Ollama)

1. `POST /api/generate keep_alive=0` to force-unload the current model so a
   changed `num_ctx` actually takes effect.
2. Poll `/api/ps` until empty.
3. Slurp the live `override.conf` to capture which flags were active.
4. Run `bench_runs` (default 3) `/api/chat` calls with `/no_think` +
   `think: false`, capturing `eval_count` / `eval_duration`.
5. Print per-run + mean `tok/s` and persist a JSON record under
   `~/Projects/Ollama/benchmarks/`.

## Why Ansible (and not X)

- **vs shell + ssh**: Ansible is idempotent — same `make deploy` works on a
  freshly-wiped DGX or one already converged. Shell needs hand-rolled guards.
- **vs NixOS**: Would require wiping the vendor OS. Too invasive for a
  turn-key box.
- **vs Docker Compose**: GPU passthrough on ARM is still a papercut trail;
  Ollama's systemd unit handles the NVIDIA runtime cleanly.
- **vs Terraform**: Provisioning tool, not post-install state management.

## Related

- `~/Projects/local-inference/` — Mac launchd plist, cross-endpoint benchmark,
  observability design doc, model-choice rationale. Treat as the **co-pilot
  repo** to this one.
- [fb-reader/docs/dgx-spark-ollama-setup.md](https://github.com/devjoe/fb-reader/blob/main/docs/dgx-spark-ollama-setup.md) —
  human-readable narrative of the same setup.
- [fb-reader/docs/llm-context-and-kv-cache.md](https://github.com/devjoe/fb-reader/blob/main/docs/llm-context-and-kv-cache.md) —
  why we picked these flag defaults.

## License

MIT (infra code, no secrets). `inventory.ini` is committed because the hostname
and username are not secret. Never commit a vault password file (`.vault_pass`,
`.vault_password` are gitignored).
