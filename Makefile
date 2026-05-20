SHELL := /bin/bash
# Set ASK_BECOME=1 to prompt for sudo password interactively (for first
# run before setting up NOPASSWD sudo). Example:
#   ASK_BECOME=1 make deploy
ANSIBLE := ansible-playbook $(if $(ASK_BECOME),--ask-become-pass,)
INVENTORY ?= inventory.ini
DGX_SSH_KEY ?= $(HOME)/Library/Application Support/NVIDIA/Sync/config/nvsync.key
ANSIBLE_EXTRA ?=
ANSIBLE_ARGS := -i $(INVENTORY) $(ANSIBLE_EXTRA)
STANCE_AB_RISK_IDS ?= contested_sovereignty_001,forced_sovereignty_pro_001,forced_sovereignty_anti_001,tw_sensitive_cross_strait_001,tw_sensitive_party_001,tw_sensitive_identity_001,tw_sensitive_energy_001,tw_sensitive_media_001
GEMMA_MTP_PRHEAD_SHA ?= d8b3826648da6b407f8c55457a2103be9aeb5d83
GEMMA_MTP_PRHEAD_URL ?= https://raw.githubusercontent.com/vllm-project/vllm/$(GEMMA_MTP_PRHEAD_SHA)/vllm/model_executor/models/gemma4_mtp.py
GEMMA_MTP_PRHEAD_REMOTE ?= /home/devjoe/Projects/Ollama/gemma-mtp-speed/gemma4_mtp-$(GEMMA_MTP_PRHEAD_SHA).py

.DEFAULT_GOAL := help

.PHONY: help ping ping-ipv4 deploy benchmark benchmark-vllm benchmark-vllm-perf gemma-mtp-fastbench gemma-mtp-fastbench-ipv4 gemma-mtp-fastbench-mm0 gemma-mtp-fastbench-mm0-ipv4 gemma-mtp-fastbench-prhead gemma-mtp-fastbench-prhead-ipv4 gemma-mtp-fastbench-mm0-prhead gemma-mtp-fastbench-mm0-prhead-ipv4 gemma-mtp-speed-targeted gemma-mtp-speed-targeted-ipv4 gemma-mtp-speed-matrix gemma-mtp-speed-matrix-ipv4 stance-ab stance-ab-ipv4 stance-ab-risk stance-ab-risk-ipv4 wifi-ipv4-only wifi-ipv4-only-ipv4 status status-vllm status-vllm-ipv4 unload models.yml lint install-deps deploy-obs status-obs canary-once os-preflight os-maint-stop os-post-smoke os-restore os-validate

help:  ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

install-deps:  ## Install required Ansible collections
	ansible-galaxy collection install -r requirements.yml

ping:  ## Test connectivity + auth to DGX
	ansible $(ANSIBLE_ARGS) dgx -m ansible.builtin.ping

ping-ipv4:  ## Test direct IPv4 fallback connectivity to DGX
	$(MAKE) ping INVENTORY=inventory.ipv4.ini ANSIBLE_EXTRA='--private-key "$(DGX_SSH_KEY)"'

deploy:  ## Converge DGX to group_vars state (idempotent)
	$(ANSIBLE) $(ANSIBLE_ARGS) site.yml

benchmark:  ## Unload, warm, run N timed eval calls → tok/s
	$(ANSIBLE) $(ANSIBLE_ARGS) benchmark.yml

benchmark-vllm:  ## Sanity-check vLLM Tier B (text + data-URI image)
	$(ANSIBLE) $(ANSIBLE_ARGS) benchmark-vllm.yml

benchmark-vllm-perf:  ## Measure vLLM perf matrix (prefill/decode × concurrency)
	$(ANSIBLE) $(ANSIBLE_ARGS) benchmark-vllm-perf.yml

gemma-mtp-speed-matrix:  ## Run Gemma4 MTP decode + stance risk speed profiles
	$(ANSIBLE) $(ANSIBLE_ARGS) playbooks/gemma-mtp-speed-matrix.yml

gemma-mtp-speed-matrix-ipv4:  ## Run Gemma4 MTP speed profiles through direct IPv4
	$(MAKE) gemma-mtp-speed-matrix INVENTORY=inventory.ipv4.ini ANSIBLE_EXTRA='--private-key "$(DGX_SSH_KEY)"'

gemma-mtp-speed-targeted:  ## Run Gemma4 prodctx-g1 and fastctx-g4 after launch-path changes
	$(ANSIBLE) $(ANSIBLE_ARGS) playbooks/gemma-mtp-speed-matrix.yml --extra-vars 'gemma_mtp_profile_ids=prodctx-g1-u055,fastctx-g4-u085'

gemma-mtp-speed-targeted-ipv4:  ## Run targeted Gemma4 speed profiles through direct IPv4
	$(MAKE) gemma-mtp-speed-targeted INVENTORY=inventory.ipv4.ini ANSIBLE_EXTRA='--private-key "$(DGX_SSH_KEY)"'

gemma-mtp-fastbench:  ## Run external-methodology-style Gemma4 decode-only fastbench
	$(ANSIBLE) $(ANSIBLE_ARGS) playbooks/gemma-mtp-speed-matrix.yml --extra-vars 'gemma_mtp_profile_ids=external-fastbench-g4-u085'

gemma-mtp-fastbench-ipv4:  ## Run Gemma4 fastbench through direct IPv4
	$(MAKE) gemma-mtp-fastbench INVENTORY=inventory.ipv4.ini ANSIBLE_EXTRA='--private-key "$(DGX_SSH_KEY)"'

gemma-mtp-fastbench-mm0:  ## Run exact external mm0 Gemma4 fastbench profile
	$(ANSIBLE) $(ANSIBLE_ARGS) playbooks/gemma-mtp-speed-matrix.yml --extra-vars 'gemma_mtp_profile_ids=external-fastbench-mm0-g4-u085'

gemma-mtp-fastbench-mm0-ipv4:  ## Run exact external mm0 fastbench through direct IPv4
	$(MAKE) gemma-mtp-fastbench-mm0 INVENTORY=inventory.ipv4.ini ANSIBLE_EXTRA='--private-key "$(DGX_SSH_KEY)"'

gemma-mtp-fastbench-prhead:  ## Run Gemma4 no-mm fastbench with PR-head gemma4_mtp.py mounted
	$(ANSIBLE) $(ANSIBLE_ARGS) playbooks/gemma-mtp-speed-matrix.yml --extra-vars 'gemma_mtp_profile_ids=external-fastbench-g4-u085 gemma_mtp_patch_url=$(GEMMA_MTP_PRHEAD_URL) gemma_mtp_patch_host_path=$(GEMMA_MTP_PRHEAD_REMOTE)'

gemma-mtp-fastbench-prhead-ipv4:  ## Run PR-head Gemma4 fastbench through direct IPv4
	$(MAKE) gemma-mtp-fastbench-prhead INVENTORY=inventory.ipv4.ini ANSIBLE_EXTRA='--private-key "$(DGX_SSH_KEY)"'

gemma-mtp-fastbench-mm0-prhead:  ## Reproduce exact mm0 fastbench with PR-head gemma4_mtp.py mounted
	$(ANSIBLE) $(ANSIBLE_ARGS) playbooks/gemma-mtp-speed-matrix.yml --extra-vars 'gemma_mtp_profile_ids=external-fastbench-mm0-g4-u085 gemma_mtp_patch_url=$(GEMMA_MTP_PRHEAD_URL) gemma_mtp_patch_host_path=$(GEMMA_MTP_PRHEAD_REMOTE)'

gemma-mtp-fastbench-mm0-prhead-ipv4:  ## Run PR-head exact mm0 fastbench through direct IPv4
	$(MAKE) gemma-mtp-fastbench-mm0-prhead INVENTORY=inventory.ipv4.ini ANSIBLE_EXTRA='--private-key "$(DGX_SSH_KEY)"'

stance-ab:  ## Run Qwen DFlash vs Gemma4 MTP stance/uncertainty A/B on DGX
	$(ANSIBLE) $(ANSIBLE_ARGS) playbooks/stance-ab.yml

stance-ab-ipv4:  ## Run stance A/B through the direct IPv4 fallback inventory
	$(MAKE) stance-ab INVENTORY=inventory.ipv4.ini ANSIBLE_EXTRA='--private-key "$(DGX_SSH_KEY)"'

stance-ab-risk:  ## Run only Taiwan / forced-framing stance risk prompts
	$(ANSIBLE) $(ANSIBLE_ARGS) playbooks/stance-ab.yml --extra-vars 'stance_ab_ids=$(STANCE_AB_RISK_IDS)'

stance-ab-risk-ipv4:  ## Run the stance risk slice through the direct IPv4 fallback inventory
	$(MAKE) stance-ab-risk INVENTORY=inventory.ipv4.ini ANSIBLE_EXTRA='--private-key "$(DGX_SSH_KEY)"'

wifi-ipv4-only:  ## Keep DGX outward Wi-Fi on IPv4 only; disables IPv6 on 10Design2
	$(ANSIBLE) $(ANSIBLE_ARGS) playbooks/dgx-wifi-ipv4-only.yml

wifi-ipv4-only-ipv4:  ## Keep Wi-Fi IPv4-only through the direct IPv4 fallback inventory
	$(MAKE) wifi-ipv4-only INVENTORY=inventory.ipv4.ini ANSIBLE_EXTRA='--private-key "$(DGX_SSH_KEY)"'

status:  ## Show what Ollama currently has loaded
	@ansible $(ANSIBLE_ARGS) dgx -m ansible.builtin.uri \
		-a "url=http://localhost:11434/api/ps return_content=yes" \
		--one-line | sed 's/.*"content": //' | sed 's/}}}$$/}}/' | python3 -m json.tool

status-vllm:  ## Show vLLM service state + /v1/models response
	@ansible $(ANSIBLE_ARGS) dgx -m ansible.builtin.shell \
		-a "systemctl is-active vllm; curl -s http://localhost:8000/v1/models | head -c 200" \
		--one-line

status-vllm-ipv4:  ## Show vLLM state through the direct IPv4 fallback inventory
	$(MAKE) status-vllm INVENTORY=inventory.ipv4.ini ANSIBLE_EXTRA='--private-key "$(DGX_SSH_KEY)"'

unload:  ## Force-unload the benchmark model (reclaim VRAM)
	@ansible $(ANSIBLE_ARGS) dgx -m ansible.builtin.uri \
		-a 'url=http://localhost:11434/api/generate method=POST body_format=json body={"model":"qwen3.5:latest","keep_alive":0}'

lint:  ## Syntax-check playbooks without touching the host
	$(ANSIBLE) $(ANSIBLE_ARGS) site.yml --syntax-check
	$(ANSIBLE) $(ANSIBLE_ARGS) benchmark.yml --syntax-check
	$(ANSIBLE) $(ANSIBLE_ARGS) benchmark-vllm.yml --syntax-check
	$(ANSIBLE) $(ANSIBLE_ARGS) benchmark-vllm-perf.yml --syntax-check
	$(ANSIBLE) $(ANSIBLE_ARGS) playbooks/gemma-mtp-speed-matrix.yml --syntax-check
	$(ANSIBLE) $(ANSIBLE_ARGS) playbooks/stance-ab.yml --syntax-check
	$(ANSIBLE) $(ANSIBLE_ARGS) playbooks/deploy-observability.yml --syntax-check
	$(ANSIBLE) $(ANSIBLE_ARGS) playbooks/os-preflight.yml --syntax-check
	$(ANSIBLE) $(ANSIBLE_ARGS) playbooks/os-maint-stop.yml --syntax-check
	$(ANSIBLE) $(ANSIBLE_ARGS) playbooks/os-post-smoke.yml --syntax-check
	$(ANSIBLE) $(ANSIBLE_ARGS) playbooks/dgx-wifi-ipv4-only.yml --syntax-check

# --- Observability (v1: data path; v2 will add Telegram alerts) -----------
# Pass the vault file as extra-vars so playbook syntax-check doesn't
# require it. Vault password file path is configurable via VAULT_PASS.
VAULT_FILE ?= group_vars/dgx.yml.vault
VAULT_PASS ?= .vault_pass

deploy-obs:  ## Stand up VM + Grafana + exporters + canary timer on the DGX
	$(ANSIBLE) $(ANSIBLE_ARGS) playbooks/deploy-observability.yml \
		--extra-vars "@$(VAULT_FILE)" \
		--vault-password-file $(VAULT_PASS)

status-obs:  ## Show systemd state of every observability unit on the DGX
	@ansible $(ANSIBLE_ARGS) dgx -m ansible.builtin.shell \
		-a "systemctl is-active observability-victoriametrics observability-grafana observability-dcgm-exporter observability-node-exporter observability-vmagent observability-canary.timer" \
		--one-line

canary-once:  ## Trigger the DGX canary timer's underlying service immediately
	@ansible $(ANSIBLE_ARGS) dgx -m ansible.builtin.systemd -a "name=observability-canary.service state=started" --become

# --- DGX OS / firmware update workflow ----------------------------------

os-preflight:  ## Collect pre-upgrade DGX OS/CUDA/service state
	$(ANSIBLE) $(ANSIBLE_ARGS) playbooks/os-preflight.yml

os-maint-stop:  ## Stop inference/canary services before manual OS upgrade
	$(ANSIBLE) $(ANSIBLE_ARGS) playbooks/os-maint-stop.yml

os-post-smoke:  ## Run post-reboot DGX OS/CUDA/PyTorch smoke checks
	$(ANSIBLE) $(ANSIBLE_ARGS) playbooks/os-post-smoke.yml

os-restore:  ## Restore Ansible-managed serving + observability state
	$(ANSIBLE) $(ANSIBLE_ARGS) site.yml
	$(ANSIBLE) $(ANSIBLE_ARGS) playbooks/deploy-observability.yml \
		--extra-vars "@$(VAULT_FILE)" \
		--vault-password-file $(VAULT_PASS)
	$(MAKE) status-vllm

os-validate:  ## Run vLLM regression check + one DGX canary after restore
	$(ANSIBLE) $(ANSIBLE_ARGS) benchmark-vllm.yml
	$(MAKE) canary-once
