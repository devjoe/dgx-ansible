# DGX Spark software update guidance (2026-05-11)

Point-in-time guidance for deciding whether to apply the current DGX Spark
system update on the GB10 box used by this inference fleet.

## Recommendation

Upgrade during a planned maintenance window, but do not interrupt active
fine-tune, long benchmark, or model-download work.

For our current usage, the update is worth taking:

- DGX is a developer / benchmark / fallback host, not a hard production serving
  dependency for `fb-reader` primary classification.
- The current official stack is DGX OS 7.5.0, NVIDIA driver 580.142, CUDA
  Toolkit 13.0.2, and Canonical kernel 6.17.
- NVIDIA now recommends using DGX Dashboard as the primary update path, with
  manual `apt dist-upgrade` + `fwupdmgr` as the fallback path.
- The April 2026 update adds enterprise deployment features and dashboard
  release-highlights, which make future fleet or air-gapped maintenance easier.

This does not change model routing by itself. Keep the DGX serving default on
the current Ansible-managed Qwen DFlash stack until a replay benchmark proves a
better default.

## What changed upstream

### Current software baseline

The official DGX Spark release notes currently list:

| Component | Version |
|---|---|
| NVIDIA DGX OS | 7.5.0 |
| NVIDIA GPU Driver | 580.142 |
| NVIDIA CUDA Toolkit | 13.0.2 |
| Canonical Kernel | 6.17 |
| UEFI | 1.108.20 |
| EC | 3.3.2 |
| USB PD | 0.5.22 |
| SoC | 2.152.15 |

These versions apply to DGX Spark Founders Edition. Partner GB10 systems may
lag.

### April 2026 release

The April release is mostly operational / enterprise enablement:

- Enterprise Management Guide.
- OOBE bypass for IT provisioning.
- USB and local repository support for installation and updates.
- Air-gapped deployment and update support.
- Customized enterprise ISOs through cloud-init.
- DGX Dashboard release highlights for deciding update urgency.
- Fixes for Bluetooth keyboard pairing and setup-time network detection.

### Earlier improvements that still matter

Some useful items are real, but belong to older release notes and should not be
attributed specifically to April 2026:

- JupyterLab CUDA 13.0.2 + PyTorch stack and the fixed image-generation example
  are from the November 2025 release.
- DGX Dashboard unified-memory reporting cleanup is from the November 2025
  release.
- ConnectX-7 hot-plug idle power savings of up to 18 W, WiFi / Bluetooth UEFI
  disablement, and multi-monitor / non-native-resolution fixes are from the
  January 2026 release.
- Multi-DGX Spark performance regression fixes are from the February 2026
  release.

## When to delay

Delay the update if any of these are true:

- A long training, fine-tune, replay, or model-download job is active.
- A project is pinned to CUDA 12.x, older PyTorch, a specific NCCL version, or
  a custom Triton / CUDA extension that has not been smoke-tested.
- Out-of-tree kernel modules or DKMS packages are installed.
- A multi-Spark / distributed workflow depends on exact NCCL or MPI behavior.

For those cases, snapshot the environment first and run the smoke tests below
before resuming normal work.

## Suggested runbook

1. Confirm current state through the automated preflight:

   ```bash
   make os-preflight
   ```

   The playbook writes a timestamped snapshot under
   `~/Projects/Ollama/os-update/` on the DGX. It records OS release, kernel,
   NVIDIA driver, CUDA, PyTorch CUDA availability, service state, Docker
   containers, model endpoints, disk, cache sizes, and reboot-required state.

   Manual equivalent:

   ```bash
   sudo nvidia-release-info || true
   cat /etc/dgx-release || true
   uname -a
   nvidia-smi
   nvcc --version
   ```

2. Pause services and workloads:

   ```bash
   make os-maint-stop
   ```

   This stops `observability-canary.timer`, `vllm`, and `ollama`. It does not
   run apt, firmware updates, or reboot.

   Manual equivalent:

   ```bash
   sudo systemctl stop vllm || true
   sudo systemctl stop ollama || true
   docker ps
   ```

3. Back up state that is expensive to recreate:

   ```bash
   du -sh ~/.cache/huggingface ~/.cache/vllm ~/.cache/ollama 2>/dev/null || true
   ```

   Prioritize checkpoints, custom envs, and local config. Model caches can be
   redownloaded, but losing them makes the maintenance window much longer.

4. Use DGX Dashboard for the update when available.

   NVIDIA documents the dashboard as the preferred update path because it
   coordinates OS, driver, and firmware updates. Manual fallback:

   ```bash
   sudo apt update
   sudo apt dist-upgrade
   sudo fwupdmgr refresh
   sudo fwupdmgr upgrade
   sudo reboot
   ```

5. Smoke-test after reboot:

   ```bash
   make os-post-smoke
   ```

   Manual equivalent:

   ```bash
   nvidia-smi
   nvcc --version
   python3 - <<'PY'
   import torch
   print(torch.__version__)
   print(torch.cuda.is_available())
   if torch.cuda.is_available():
       x = torch.randn(1024, 1024, device="cuda")
       print((x @ x).mean().item())
   PY
   ```

6. Restore the Ansible-managed serving stack:

   ```bash
   make os-restore
   ```

7. Run one regression check and one canary before calling the update good:

   ```bash
   make os-validate
   ```

   If canary latency materially changes, run the full `fb-reader` Tier B replay
   separately before changing model or runtime decisions.

## Decision for this fleet

For the current `local-inference` / `fb-reader` / `dgx-ansible` setup:

- Recommended action: upgrade, but schedule it.
- Do not treat the OS update as a model-performance win until canary or replay
  data proves it.
- After the update, rerun the DGX canary first. Only rerun full Tier B replay
  if the canary shows a material change or if vLLM / CUDA containers are also
  being upgraded.

## Local run note (2026-05-11)

Automated pre/post steps were added and exercised from this repo:

- `make os-preflight` wrote
  `/home/devjoe/Projects/Ollama/os-update/preflight-20260511T123005.txt`.
- `make os-maint-stop` stopped `vllm`, `ollama`, and
  `observability-canary.timer` before the manual maintenance step.
- `make os-post-smoke` wrote
  `/home/devjoe/Projects/Ollama/os-update/post-smoke-20260511T140115.txt`.
- `make os-restore` converged the Ansible-managed serving and observability
  stack.
- `make os-validate` passed the vLLM text + data-URI image regression check and
  triggered one DGX canary.

Observed state after the maintenance cycle:

- `/etc/dgx-release` still reports `DGX_SWBUILD_VERSION="7.4.0"` and
  `DGX_OTA_VERSION="7.5.0"`, so treat OTA version as the effective update marker
  unless NVIDIA documents a different release-file convention.
- Kernel stayed on `6.17.0-1014-nvidia`.
- NVIDIA driver stayed on `580.142`, CUDA reported by the driver stayed `13.0`.
- System `nvcc` is not on PATH and system Python has no torch package; the
  service path is the vLLM venv.
- vLLM venv smoke passed with `torch 2.11.0+cu130`, `cuda_available True`, and
  a CUDA matmul.
- DGX canary after restore succeeded:
  - short: TTFT 220 ms, decode 122.2 tok/s
  - medium: TTFT 144 ms, decode 137.1 tok/s
  - long: TTFT 172 ms, decode 90.8 tok/s

The canary numbers are materially faster than the May 7-11 nightly baseline
visible in journal (`short` roughly 73-78 tok/s, `medium` roughly 81-85 tok/s,
`long` roughly 52-54 tok/s). Keep watching the next scheduled runs before
treating this as a stable performance uplift.

## Sources

- NVIDIA DGX Spark Release Notes, last checked 2026-05-11:
  https://docs.nvidia.com/dgx/dgx-spark/release-notes.html
- NVIDIA DGX Spark OS and Component Update Guide, last checked 2026-05-11:
  https://docs.nvidia.com/dgx/dgx-spark/os-and-component-update.html
- NVIDIA Developer Forums, "DGX Spark Software Updates - April 2026 Release":
  https://forums.developer.nvidia.com/t/dgx-spark-software-updates-04-2026/368114
