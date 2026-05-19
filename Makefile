SHELL := /bin/bash
# Set ASK_BECOME=1 to prompt for sudo password interactively (for first
# run before setting up NOPASSWD sudo). Example:
#   ASK_BECOME=1 make deploy
ANSIBLE := ansible-playbook $(if $(ASK_BECOME),--ask-become-pass,)
INVENTORY ?= inventory.ini
DGX_SSH_KEY ?= $(HOME)/Library/Application Support/NVIDIA/Sync/config/nvsync.key
ANSIBLE_EXTRA ?=
ANSIBLE_ARGS := -i $(INVENTORY) $(ANSIBLE_EXTRA)

.DEFAULT_GOAL := help

.PHONY: help ping ping-ipv4 deploy benchmark benchmark-vllm benchmark-vllm-perf stance-ab stance-ab-ipv4 wifi-ipv4-only wifi-ipv4-only-ipv4 status status-vllm status-vllm-ipv4 unload models.yml lint install-deps deploy-obs status-obs canary-once os-preflight os-maint-stop os-post-smoke os-restore os-validate

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

stance-ab:  ## Run Qwen DFlash vs Gemma4 MTP stance/uncertainty A/B on DGX
	$(ANSIBLE) $(ANSIBLE_ARGS) playbooks/stance-ab.yml

stance-ab-ipv4:  ## Run stance A/B through the direct IPv4 fallback inventory
	$(MAKE) stance-ab INVENTORY=inventory.ipv4.ini ANSIBLE_EXTRA='--private-key "$(DGX_SSH_KEY)"'

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
