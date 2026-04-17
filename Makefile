SHELL := /bin/bash
# Set ASK_BECOME=1 to prompt for sudo password interactively (for first
# run before setting up NOPASSWD sudo). Example:
#   ASK_BECOME=1 make deploy
ANSIBLE := ansible-playbook $(if $(ASK_BECOME),--ask-become-pass,)

.DEFAULT_GOAL := help

.PHONY: help ping deploy benchmark status unload models.yml lint install-deps

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

status:  ## Show what Ollama currently has loaded
	@ansible dgx -m ansible.builtin.uri \
		-a "url=http://localhost:11434/api/ps return_content=yes" \
		--one-line | sed 's/.*"content": //' | sed 's/}}}$$/}}/' | python3 -m json.tool

unload:  ## Force-unload the benchmark model (reclaim VRAM)
	@ansible dgx -m ansible.builtin.uri \
		-a 'url=http://localhost:11434/api/generate method=POST body_format=json body={"model":"qwen3.5:latest","keep_alive":0}'

lint:  ## Syntax-check playbooks without touching the host
	$(ANSIBLE) site.yml --syntax-check
	$(ANSIBLE) benchmark.yml --syntax-check
