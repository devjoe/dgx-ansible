# dgx-ansible

Declarative management of the NVIDIA DGX Spark (`gx10.local` / `192.168.99.2`) running Ollama for the [fb-reader](https://github.com/devjoe/fb-reader) Chrome extension.

All host state — systemd env vars, firewall rules, model list — lives in `group_vars/dgx.yml`. Run `make deploy` to converge the host. Run `make benchmark` to measure `tok/s` against the current configuration.

## Prerequisites

On the MacBook:

- `ansible` (via `brew install ansible` or `pipx install ansible-core`)
- SSH access to `gx10.local` as a sudo-capable user (configured in `inventory.ini`, default `devjoe`)
- Passwordless sudo on the DGX (`devjoe ALL=(ALL) NOPASSWD:ALL` in `/etc/sudoers.d/`) OR run playbooks with `--ask-become-pass`

## Quick start

```bash
git clone git@github.com:devjoe/dgx-ansible.git
cd dgx-ansible
make install-deps          # ansible collections
make ping                  # verify SSH + sudo reach
make deploy                # converge DGX to group_vars state
make status                # inspect currently-loaded models
make benchmark             # timed eval — prints tok/s summary
```

## A/B testing speed flags

The flags `OLLAMA_FLASH_ATTENTION` and `OLLAMA_KV_CACHE_TYPE` are well-known to regress on some hardware × model combinations ([ollama#12432](https://github.com/ollama/ollama/issues/12432), [#11949](https://github.com/ollama/ollama/issues/11949), [#9683](https://github.com/ollama/ollama/issues/9683), [#6769](https://github.com/ollama/ollama/issues/6769)). They default to OFF. To benchmark on/off:

```bash
# Baseline (off)
make deploy && make benchmark

# Flip FLASH_ATTENTION on
sed -i '' 's/ollama_flash_attention: false/ollama_flash_attention: true/' group_vars/dgx.yml
make deploy && make benchmark

# Revert
git checkout group_vars/dgx.yml
make deploy
```

Each `make deploy` re-renders `/etc/systemd/system/ollama.service.d/override.conf` and restarts Ollama exactly once when the file actually changes.

## Repo layout

```
dgx-ansible/
├── ansible.cfg              # ssh/output settings
├── inventory.ini            # [dgx] gx10.local
├── requirements.yml         # community.general, ansible.posix
├── site.yml                 # main playbook (ollama role)
├── benchmark.yml            # measure-only playbook (benchmark role)
├── group_vars/dgx.yml       # single source of truth
├── roles/
│   ├── ollama/              # install + systemd + firewall + model pulls
│   └── benchmark/           # unload, warm, time eval calls
└── Makefile                 # deploy / benchmark / status / unload / ping
```

## What converging does

1. Install Ollama from the upstream script (skipped if already installed).
2. Render `/etc/systemd/system/ollama.service.d/override.conf` from `ollama-override.conf.j2` using values in `group_vars/dgx.yml`.
3. Open UFW port `11434` from `lan_cidr`.
4. Start/enable the `ollama` systemd unit; restart it if the override file changed.
5. Pull any models listed in `ollama_models` that aren't already present (multi-GB, runs async with a 30-minute ceiling).
6. Assert `GET /api/tags` returns 200 and contains every model we just pulled.

## What the benchmark does

1. `POST /api/generate` with `keep_alive=0` to force-unload the current model (so a changed `num_ctx` actually takes effect).
2. Poll `/api/ps` until `models: []`.
3. Slurp the current `override.conf` to print exactly which flags were active.
4. Run `bench_runs` (default 3) chat completions with `/no_think` + `think: false` and capture `eval_count` / `eval_duration`.
5. Print per-run and mean `tok/s` in a block you can paste into a commit message.

## Why Ansible (and not X)

- **vs shell + ssh**: Ansible is idempotent by default — the same `make deploy` works on a freshly-wiped DGX and on one that's already converged. Shell needs hand-rolled guards.
- **vs NixOS**: Would require wiping the vendor OS. Too invasive for a turn-key box.
- **vs Docker Compose**: GPU passthrough on ARM is still a papercut trail; Ollama's systemd unit already handles the NVIDIA runtime cleanly.
- **vs Terraform**: Provisioning tool, not post-install state management.

## Related docs

- [fb-reader/docs/dgx-spark-ollama-setup.md](https://github.com/devjoe/fb-reader/blob/main/docs/dgx-spark-ollama-setup.md) — human-readable narrative of the same setup.
- [fb-reader/docs/llm-context-and-kv-cache.md](https://github.com/devjoe/fb-reader/blob/main/docs/llm-context-and-kv-cache.md) — why we picked these flag defaults.

## License

MIT (infra code, no secrets). `inventory.ini` is committed because the hostname and username are not secret; if you fork this for another machine, edit inventory locally and don't commit credentials.
