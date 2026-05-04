SHELL := /bin/bash
# Set ASK_BECOME=1 to prompt for sudo password interactively (for first
# run before setting up NOPASSWD sudo). Example:
#   ASK_BECOME=1 make deploy
ANSIBLE := ansible-playbook $(if $(ASK_BECOME),--ask-become-pass,)

.DEFAULT_GOAL := help

.PHONY: help ping deploy benchmark benchmark-vllm status status-vllm unload models.yml lint install-deps deploy-obs status-obs canary-once

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

install-deps:  ## Install required Ansible collections
	ansible-galaxy collection install -r requirements.yml

ping:  ## Test connectivity + auth to DGX
	ansible dgx -m ansible.builtin.ping

deploy:  ## Converge DGX to group_vars state (idempotent)
	$(ANSIBLE) site.yml

benchmark:  ## Unload, warm, run N timed eval calls → tok/s
	$(ANSIBLE) benchmark.yml

benchmark-vllm:  ## Sanity-check vLLM Tier B (text + data-URI image)
	$(ANSIBLE) benchmark-vllm.yml

status:  ## Show what Ollama currently has loaded
	@ansible dgx -m ansible.builtin.uri \
		-a "url=http://localhost:11434/api/ps return_content=yes" \
		--one-line | sed 's/.*"content": //' | sed 's/}}}$$/}}/' | python3 -m json.tool

status-vllm:  ## Show vLLM service state + /v1/models response
	@ansible dgx -m ansible.builtin.shell \
		-a "systemctl is-active vllm; curl -s http://localhost:8000/v1/models | head -c 200" \
		--one-line

unload:  ## Force-unload the benchmark model (reclaim VRAM)
	@ansible dgx -m ansible.builtin.uri \
		-a 'url=http://localhost:11434/api/generate method=POST body_format=json body={"model":"qwen3.5:latest","keep_alive":0}'

lint:  ## Syntax-check playbooks without touching the host
	$(ANSIBLE) site.yml --syntax-check
	$(ANSIBLE) benchmark.yml --syntax-check
	$(ANSIBLE) benchmark-vllm.yml --syntax-check
	$(ANSIBLE) playbooks/deploy-observability.yml --syntax-check

# --- Observability (v1: data path; v2 will add Telegram alerts) -----------
# Pass the vault file as extra-vars so playbook syntax-check doesn't
# require it. Vault password file path is configurable via VAULT_PASS.
VAULT_FILE ?= group_vars/dgx.yml.vault
VAULT_PASS ?= .vault_pass

deploy-obs:  ## Stand up VM + Grafana + exporters + canary timer on the DGX
	$(ANSIBLE) playbooks/deploy-observability.yml \
		--extra-vars "@$(VAULT_FILE)" \
		--vault-password-file $(VAULT_PASS)

status-obs:  ## Show systemd state of every observability unit on the DGX
	@ansible dgx -m ansible.builtin.shell \
		-a "systemctl is-active observability-victoriametrics observability-grafana observability-dcgm-exporter observability-node-exporter observability-ollama-exporter observability-vmagent observability-canary.timer" \
		--one-line

canary-once:  ## Trigger the DGX canary timer's underlying service immediately
	@ansible dgx -m ansible.builtin.systemd -a "name=observability-canary.service state=started" --become
