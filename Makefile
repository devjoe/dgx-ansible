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
FB_READER_AB_CORPUS ?= tmp/tier-b-corpus-2026-05-06T07-53-23-804Z/tier-b-cases.json
FB_READER_AB_LIMIT ?=
FB_READER_AB_STANCE_IDS ?= $(STANCE_AB_RISK_IDS)
QWEN_CONDITIONAL_PROMPT_STANCE_IDS ?= $(STANCE_AB_RISK_IDS)
QWEN_CONDITIONAL_PROMPT_WATCH_CORPUS ?= prompts/qwen_settled_watch_regression.json
TIERB_NVFP4_CANDIDATE_IDS ?= qwen36-nvfp4,gemma4-nvfp4
TIERB_NVFP4_QWEN_IMAGE ?= nvcr.io/nvidia/vllm:26.02-py3
TIERB_NVFP4_GEMMA_IMAGE ?= vllm/vllm-openai:gemma4-0505-cu130
NEWS_CONTEXT_STANCE_CORPUS ?= prompts/news_context_stance_corpus.json
NEWS_FULLTEXT_STANCE_SPEC ?= prompts/news_fulltext_stance_sources.json
NEWS_FULLTEXT_STANCE_CORPUS ?= tmp/news-fulltext-stance-corpus.json
NEWS_FULLTEXT_STRICT_STANCE_CORPUS ?= tmp/news-fulltext-stance-strict-corpus.json
NEWS_FULLTEXT_PREPASS_STANCE_CORPUS ?= tmp/news-fulltext-stance-prepass-corpus.json
NEWS_FULLTEXT10_STANCE_SPEC ?= prompts/news_fulltext10_stance_sources.json
NEWS_FULLTEXT10_PREPASS_STANCE_CORPUS ?= tmp/news-fulltext10-stance-prepass-corpus.json
DS4_DIR_STEERING_CONTESTED ?= prompts/ds4/contested.txt
DS4_DIR_STEERING_SETTLED ?= prompts/ds4/settled.txt
DS4_DIR_STEERING_CORPUS ?= tmp/ds4-dir-steering-corpus.json
DS4_DIR_STEERING_LIMIT ?=
QWEN_DIR_STEERING_LIMIT ?=
QWEN_DIR_STEERING_IDS ?=
QWEN_DIR_STEERING_PROFILE_IDS ?=
QWEN_DIR_STEERING_PROMPT_2X2_IDS ?= ds4_contested_001,ds4_contested_002,ds4_contested_003,ds4_contested_004,ds4_contested_005,ds4_contested_006,ds4_contested_007,ds4_contested_008,ds4_contested_009,ds4_contested_010,ds4_contested_011,ds4_contested_012,ds4_settled_053,ds4_settled_055,ds4_settled_065,ds4_settled_069,ds4_settled_070,ds4_settled_071,ds4_settled_075,ds4_settled_081,ds4_settled_102,ds4_settled_103,ds4_settled_113,ds4_settled_115,ds4_settled_119
QWEN_DIR_STEERING_PROMPT_2X2_PROFILE_IDS ?= noop-dflash-current-prompt,noop-dflash-stakeholder-prompt,steer-l32-35-s020-current-prompt,steer-l32-35-s020-stakeholder-prompt
QWEN_DIR_STEERING_PROMPT_CONDITIONAL_PROFILE_IDS ?= noop-dflash-conditional-prompt,steer-l32-35-s020-conditional-prompt
QWEN_DIR_STEERING_SWEEP_PROFILE_IDS ?= noop-dflash,steer-l34-s005-ablate,steer-l34-s010-ablate,steer-l34-s020-ablate,steer-l32-35-s005-ablate,steer-l32-35-s010-ablate,steer-l32-35-s020-ablate,steer-l36-39-s005-ablate,steer-l36-39-s010-ablate,steer-l36-39-s020-ablate
QWEN_DIR_STEERING_SWEEP_LIMIT ?= 24
QWEN_DIR_STEERING_DIRECTIONS ?= /home/devjoe/Projects/Ollama/benchmarks/qwen-dir-steering-extract-20260521T071702Z/directions.pt
QWEN_DIR_STEERING_FB_READER_PROFILE_ID ?= steer-l32-35-s020-ablate
QWEN_DIR_STEERING_EXTRACT_CORPUS ?= tmp/qwen-dir-steering-extraction-corpus.json
QWEN_DIR_STEERING_EXTRACT_MANUAL ?= reports/qwen-dir-steering-20260521T044900Z/noop-dflash-manual-review.json
QWEN_DIR_STEERING_EXTRACT_MAX_ITEMS ?= 4
QWEN_DIR_STEERING_EXTRACT_LAYERS ?= 0,10,20,30,40
QWEN_DIR_STEERING_EXTRACT_MAX_LENGTH ?= 512
QWEN_DIR_STEERING_EXTRACT_INSTALL_DEPS ?= true
GEMMA_MTP_PRHEAD_SHA ?= d8b3826648da6b407f8c55457a2103be9aeb5d83
GEMMA_MTP_PRHEAD_URL ?= https://raw.githubusercontent.com/vllm-project/vllm/$(GEMMA_MTP_PRHEAD_SHA)/vllm/model_executor/models/gemma4_mtp.py
GEMMA_MTP_PRHEAD_REMOTE ?= /home/devjoe/Projects/Ollama/gemma-mtp-speed/gemma4_mtp-$(GEMMA_MTP_PRHEAD_SHA).py
GEMMA_MTP_PRHEAD_STANCE_PROFILES ?= prodctx-g1-u055,prodctx-g4-u055,fastctx-g4-u085

.DEFAULT_GOAL := help

.PHONY: help ping ping-ipv4 deploy benchmark benchmark-vllm benchmark-vllm-perf fb-reader-ab-prhead fb-reader-ab-prhead-ipv4 fb-reader-ab-prhead-full-stance fb-reader-ab-prhead-full-stance-ipv4 tierb-nvfp4-candidates tierb-nvfp4-candidates-ipv4 qwen-conditional-prompt-gate qwen-conditional-prompt-gate-ipv4 news-context-stance-ab-prhead news-context-stance-ab-prhead-ipv4 news-fulltext-stance-corpus news-fulltext-stance-ab-prhead news-fulltext-stance-ab-prhead-ipv4 news-fulltext-strict-stance-corpus news-fulltext-strict-stance-ab-prhead news-fulltext-strict-stance-ab-prhead-ipv4 news-fulltext-prepass-stance-corpus news-fulltext-prepass-stance-ab-prhead news-fulltext-prepass-stance-ab-prhead-ipv4 news-fulltext10-prepass-stance-corpus news-fulltext10-prepass-stance-ab-prhead news-fulltext10-prepass-stance-ab-prhead-ipv4 ds4-dir-steering-fetch ds4-dir-steering-corpus ds4-dir-steering-ab-prhead ds4-dir-steering-ab-prhead-ipv4 qwen-dir-steering-ds4 qwen-dir-steering-ds4-ipv4 qwen-dir-steering-hook-smoke qwen-dir-steering-hook-smoke-ipv4 qwen-dir-steering-prompt2x2 qwen-dir-steering-prompt2x2-ipv4 qwen-dir-steering-prompt-conditional qwen-dir-steering-prompt-conditional-ipv4 qwen-dir-steering-sweep qwen-dir-steering-sweep-ipv4 qwen-dir-steering-fb-reader qwen-dir-steering-fb-reader-ipv4 qwen-dir-steering-extraction-corpus qwen-dir-steering-extract qwen-dir-steering-extract-ipv4 gemma-mtp-endpoint-parity-prhead gemma-mtp-endpoint-parity-prhead-ipv4 gemma-mtp-fastbench gemma-mtp-fastbench-ipv4 gemma-mtp-fastbench-mm0 gemma-mtp-fastbench-mm0-ipv4 gemma-mtp-fastbench-prhead gemma-mtp-fastbench-prhead-ipv4 gemma-mtp-fastbench-mm0-prhead gemma-mtp-fastbench-mm0-prhead-ipv4 gemma-mtp-speed-targeted gemma-mtp-speed-targeted-ipv4 gemma-mtp-speed-targeted-prhead gemma-mtp-speed-targeted-prhead-ipv4 gemma-mtp-speed-matrix gemma-mtp-speed-matrix-ipv4 stance-ab stance-ab-ipv4 stance-ab-risk stance-ab-risk-ipv4 wifi-ipv4-only wifi-ipv4-only-ipv4 status status-vllm status-vllm-ipv4 unload models.yml lint install-deps deploy-obs status-obs canary-once os-preflight os-maint-stop os-post-smoke os-restore os-validate

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

fb-reader-ab-prhead:  ## Run fb-reader Tier B replay + stance-v2: Qwen vs Gemma4 PR-head
	$(ANSIBLE) $(ANSIBLE_ARGS) playbooks/fb-reader-ab-prhead.yml --extra-vars 'fb_reader_ab_corpus=$(FB_READER_AB_CORPUS) fb_reader_ab_limit=$(FB_READER_AB_LIMIT) fb_reader_ab_stance_ids=$(FB_READER_AB_STANCE_IDS) gemma4_mtp_patch_url=$(GEMMA_MTP_PRHEAD_URL) gemma4_mtp_patch_host_path=$(GEMMA_MTP_PRHEAD_REMOTE)'

fb-reader-ab-prhead-ipv4:  ## Run fb-reader A/B through direct IPv4
	$(MAKE) fb-reader-ab-prhead INVENTORY=inventory.ipv4.ini ANSIBLE_EXTRA='--private-key "$(DGX_SSH_KEY)"'

fb-reader-ab-prhead-full-stance:  ## Run fb-reader A/B with the full 21-item stance-v2 corpus
	$(MAKE) fb-reader-ab-prhead FB_READER_AB_STANCE_IDS=

fb-reader-ab-prhead-full-stance-ipv4:  ## Run full 21-item stance-v2 A/B through direct IPv4
	$(MAKE) fb-reader-ab-prhead-full-stance INVENTORY=inventory.ipv4.ini ANSIBLE_EXTRA='--private-key "$(DGX_SSH_KEY)"'

tierb-nvfp4-candidates:  ## Run fb-reader replay + stance-v2 against Qwen/Gemma NVFP4 candidates
	$(ANSIBLE) $(ANSIBLE_ARGS) playbooks/tierb-nvfp4-candidates.yml --extra-vars 'tierb_ab_corpus=$(FB_READER_AB_CORPUS) tierb_ab_limit=$(FB_READER_AB_LIMIT) tierb_ab_stance_ids=$(FB_READER_AB_STANCE_IDS) tierb_ab_candidate_ids=$(TIERB_NVFP4_CANDIDATE_IDS) tierb_ab_qwen_vllm_image=$(TIERB_NVFP4_QWEN_IMAGE) tierb_ab_gemma_vllm_image=$(TIERB_NVFP4_GEMMA_IMAGE)'

tierb-nvfp4-candidates-ipv4:  ## Run NVFP4 candidate A/B through direct IPv4
	$(MAKE) tierb-nvfp4-candidates INVENTORY=inventory.ipv4.ini ANSIBLE_EXTRA='--private-key "$(DGX_SSH_KEY)"'

qwen-conditional-prompt-gate:  ## Run prompt-only Qwen conditional gate + 7 settled-watch regressions
	$(ANSIBLE) $(ANSIBLE_ARGS) playbooks/qwen-conditional-prompt-gate.yml --extra-vars 'qwen_conditional_prompt_fb_reader_corpus=$(FB_READER_AB_CORPUS) qwen_conditional_prompt_fb_reader_limit=$(FB_READER_AB_LIMIT) qwen_conditional_prompt_stance_ids=$(QWEN_CONDITIONAL_PROMPT_STANCE_IDS) qwen_conditional_prompt_watch_corpus_src=$(CURDIR)/$(QWEN_CONDITIONAL_PROMPT_WATCH_CORPUS)'

qwen-conditional-prompt-gate-ipv4:  ## Run prompt-only Qwen conditional gate through direct IPv4
	$(MAKE) qwen-conditional-prompt-gate INVENTORY=inventory.ipv4.ini ANSIBLE_EXTRA='--private-key "$(DGX_SSH_KEY)"'

news-context-stance-ab-prhead:  ## Run Trump/Xi current-news stance-v2 A/B with PR-head Gemma
	$(ANSIBLE) $(ANSIBLE_ARGS) playbooks/stance-v2-ab-prhead.yml --extra-vars 'stance_v2_ab_corpus_src=$(CURDIR)/$(NEWS_CONTEXT_STANCE_CORPUS) gemma4_mtp_patch_url=$(GEMMA_MTP_PRHEAD_URL) gemma4_mtp_patch_host_path=$(GEMMA_MTP_PRHEAD_REMOTE)'

news-context-stance-ab-prhead-ipv4:  ## Run current-news stance-v2 A/B through direct IPv4
	$(MAKE) news-context-stance-ab-prhead INVENTORY=inventory.ipv4.ini ANSIBLE_EXTRA='--private-key "$(DGX_SSH_KEY)"'

news-fulltext-stance-corpus:  ## Fetch current-news fulltext into tmp runtime corpus
	python3 scripts/build_news_fulltext_stance_corpus.py --spec "$(NEWS_FULLTEXT_STANCE_SPEC)" --output "$(NEWS_FULLTEXT_STANCE_CORPUS)"

news-fulltext-stance-ab-prhead: news-fulltext-stance-corpus  ## Run fulltext current-news stance-v2 A/B
	$(ANSIBLE) $(ANSIBLE_ARGS) playbooks/stance-v2-ab-prhead.yml --extra-vars 'stance_v2_ab_corpus_src=$(CURDIR)/$(NEWS_FULLTEXT_STANCE_CORPUS) gemma4_mtp_patch_url=$(GEMMA_MTP_PRHEAD_URL) gemma4_mtp_patch_host_path=$(GEMMA_MTP_PRHEAD_REMOTE) stance_v2_ab_timeout=360 stance_v2_ab_max_tokens=1100'

news-fulltext-stance-ab-prhead-ipv4:  ## Run fulltext current-news stance-v2 A/B through direct IPv4
	$(MAKE) news-fulltext-stance-ab-prhead INVENTORY=inventory.ipv4.ini ANSIBLE_EXTRA='--private-key "$(DGX_SSH_KEY)"'

news-fulltext-strict-stance-corpus:  ## Fetch fulltext into tmp corpus with strict source-grounded contract
	python3 scripts/build_news_fulltext_stance_corpus.py --spec "$(NEWS_FULLTEXT_STANCE_SPEC)" --output "$(NEWS_FULLTEXT_STRICT_STANCE_CORPUS)" --answer-contract source_grounded --item-id-suffix strict

news-fulltext-strict-stance-ab-prhead: news-fulltext-strict-stance-corpus  ## Run strict fulltext current-news stance-v2 A/B
	$(ANSIBLE) $(ANSIBLE_ARGS) playbooks/stance-v2-ab-prhead.yml --extra-vars 'stance_v2_ab_corpus_src=$(CURDIR)/$(NEWS_FULLTEXT_STRICT_STANCE_CORPUS) gemma4_mtp_patch_url=$(GEMMA_MTP_PRHEAD_URL) gemma4_mtp_patch_host_path=$(GEMMA_MTP_PRHEAD_REMOTE) stance_v2_ab_timeout=420 stance_v2_ab_max_tokens=1500'

news-fulltext-strict-stance-ab-prhead-ipv4:  ## Run strict fulltext current-news stance-v2 A/B through direct IPv4
	$(MAKE) news-fulltext-strict-stance-ab-prhead INVENTORY=inventory.ipv4.ini ANSIBLE_EXTRA='--private-key "$(DGX_SSH_KEY)"'

news-fulltext-prepass-stance-corpus:  ## Fetch fulltext into tmp corpus with claim prepass prompts
	python3 scripts/build_news_fulltext_stance_corpus.py --spec "$(NEWS_FULLTEXT_STANCE_SPEC)" --output "$(NEWS_FULLTEXT_PREPASS_STANCE_CORPUS)" --answer-contract claim_prepass --item-id-suffix prepass

news-fulltext-prepass-stance-ab-prhead: news-fulltext-prepass-stance-corpus  ## Run fulltext stance A/B with claim-extraction/verifier prepass
	$(ANSIBLE) $(ANSIBLE_ARGS) playbooks/stance-v2-ab-prhead.yml --extra-vars 'stance_v2_ab_corpus_src=$(CURDIR)/$(NEWS_FULLTEXT_PREPASS_STANCE_CORPUS) gemma4_mtp_patch_url=$(GEMMA_MTP_PRHEAD_URL) gemma4_mtp_patch_host_path=$(GEMMA_MTP_PRHEAD_REMOTE) stance_v2_ab_timeout=480 stance_v2_ab_max_tokens=1500 stance_v2_ab_prepass_max_tokens=1800'

news-fulltext-prepass-stance-ab-prhead-ipv4:  ## Run fulltext claim-prepass stance A/B through direct IPv4
	$(MAKE) news-fulltext-prepass-stance-ab-prhead INVENTORY=inventory.ipv4.ini ANSIBLE_EXTRA='--private-key "$(DGX_SSH_KEY)"'

news-fulltext10-prepass-stance-corpus:  ## Fetch 10-article fulltext corpus with claim prepass prompts
	python3 scripts/build_news_fulltext_stance_corpus.py --spec "$(NEWS_FULLTEXT10_STANCE_SPEC)" --output "$(NEWS_FULLTEXT10_PREPASS_STANCE_CORPUS)" --answer-contract claim_prepass --item-id-suffix prepass

news-fulltext10-prepass-stance-ab-prhead: news-fulltext10-prepass-stance-corpus  ## Run 10-article fulltext stance A/B with claim prepass
	$(ANSIBLE) $(ANSIBLE_ARGS) playbooks/stance-v2-ab-prhead.yml --extra-vars 'stance_v2_ab_corpus_src=$(CURDIR)/$(NEWS_FULLTEXT10_PREPASS_STANCE_CORPUS) gemma4_mtp_patch_url=$(GEMMA_MTP_PRHEAD_URL) gemma4_mtp_patch_host_path=$(GEMMA_MTP_PRHEAD_REMOTE) stance_v2_ab_timeout=480 stance_v2_ab_max_tokens=1500 stance_v2_ab_prepass_max_tokens=1800'

news-fulltext10-prepass-stance-ab-prhead-ipv4:  ## Run 10-article claim-prepass stance A/B through direct IPv4
	$(MAKE) news-fulltext10-prepass-stance-ab-prhead INVENTORY=inventory.ipv4.ini ANSIBLE_EXTRA='--private-key "$(DGX_SSH_KEY)"'

ds4-dir-steering-fetch:  ## Refresh DS4 dir-steering example fixtures from upstream
	mkdir -p prompts/ds4
	curl -L -o "$(DS4_DIR_STEERING_CONTESTED)" https://raw.githubusercontent.com/audreyt/ds4/main/dir-steering/examples/contested.txt
	curl -L -o "$(DS4_DIR_STEERING_SETTLED)" https://raw.githubusercontent.com/audreyt/ds4/main/dir-steering/examples/settled.txt

ds4-dir-steering-corpus:  ## Build DS4 contested/settled dir-steering corpus
	python3 scripts/build_ds4_dir_steering_corpus.py --contested "$(DS4_DIR_STEERING_CONTESTED)" --settled "$(DS4_DIR_STEERING_SETTLED)" --output "$(DS4_DIR_STEERING_CORPUS)"

ds4-dir-steering-ab-prhead: ds4-dir-steering-corpus  ## Run DS4 contested/settled dir-steering A/B
	$(ANSIBLE) $(ANSIBLE_ARGS) playbooks/stance-v2-ab-prhead.yml --extra-vars 'stance_v2_ab_corpus_src=$(CURDIR)/$(DS4_DIR_STEERING_CORPUS) stance_v2_ab_limit=$(DS4_DIR_STEERING_LIMIT) gemma4_mtp_patch_url=$(GEMMA_MTP_PRHEAD_URL) gemma4_mtp_patch_host_path=$(GEMMA_MTP_PRHEAD_REMOTE) stance_v2_ab_timeout=240 stance_v2_ab_max_tokens=700'

ds4-dir-steering-ab-prhead-ipv4:  ## Run DS4 dir-steering A/B through direct IPv4
	$(MAKE) ds4-dir-steering-ab-prhead INVENTORY=inventory.ipv4.ini ANSIBLE_EXTRA='--private-key "$(DGX_SSH_KEY)"'

qwen-dir-steering-ds4: ds4-dir-steering-corpus  ## Run isolated Qwen dir-steering DS4 calibration profiles
	$(ANSIBLE) $(ANSIBLE_ARGS) playbooks/qwen-dir-steering-ds4.yml --extra-vars 'qwen_dir_steering_corpus_src=$(CURDIR)/$(DS4_DIR_STEERING_CORPUS) qwen_dir_steering_limit=$(QWEN_DIR_STEERING_LIMIT) qwen_dir_steering_ids=$(QWEN_DIR_STEERING_IDS) qwen_dir_steering_profile_ids=$(QWEN_DIR_STEERING_PROFILE_IDS) qwen_dir_steering_directions_path=$(QWEN_DIR_STEERING_DIRECTIONS)'

qwen-dir-steering-ds4-ipv4:  ## Run isolated Qwen dir-steering DS4 calibration through direct IPv4
	$(MAKE) qwen-dir-steering-ds4 INVENTORY=inventory.ipv4.ini ANSIBLE_EXTRA='--private-key "$(DGX_SSH_KEY)"'

qwen-dir-steering-hook-smoke:  ## Smoke-test experiment Qwen activation hook against DS4 slice
	$(MAKE) qwen-dir-steering-ds4 QWEN_DIR_STEERING_LIMIT=4 QWEN_DIR_STEERING_PROFILE_IDS=noop-dflash,steer-l32-35-s005-ablate

qwen-dir-steering-hook-smoke-ipv4:  ## Smoke-test Qwen activation hook through direct IPv4
	$(MAKE) qwen-dir-steering-hook-smoke INVENTORY=inventory.ipv4.ini ANSIBLE_EXTRA='--private-key "$(DGX_SSH_KEY)"'

qwen-dir-steering-prompt2x2:  ## Run DS4 slice with current/stakeholder prompt x no-op/steered Qwen
	$(MAKE) qwen-dir-steering-ds4 QWEN_DIR_STEERING_PROFILE_IDS=$(QWEN_DIR_STEERING_PROMPT_2X2_PROFILE_IDS) QWEN_DIR_STEERING_IDS=$(QWEN_DIR_STEERING_PROMPT_2X2_IDS)

qwen-dir-steering-prompt2x2-ipv4:  ## Run Qwen steering prompt 2x2 through direct IPv4
	$(MAKE) qwen-dir-steering-prompt2x2 INVENTORY=inventory.ipv4.ini ANSIBLE_EXTRA='--private-key "$(DGX_SSH_KEY)"'

qwen-dir-steering-prompt-conditional:  ## Run DS4 slice with conditional stakeholder prompt
	$(MAKE) qwen-dir-steering-ds4 QWEN_DIR_STEERING_PROFILE_IDS=$(QWEN_DIR_STEERING_PROMPT_CONDITIONAL_PROFILE_IDS) QWEN_DIR_STEERING_IDS=$(QWEN_DIR_STEERING_PROMPT_2X2_IDS)

qwen-dir-steering-prompt-conditional-ipv4:  ## Run conditional prompt probe through direct IPv4
	$(MAKE) qwen-dir-steering-prompt-conditional INVENTORY=inventory.ipv4.ini ANSIBLE_EXTRA='--private-key "$(DGX_SSH_KEY)"'

qwen-dir-steering-sweep:  ## Run Qwen activation hook layer/scale sweep
	$(MAKE) qwen-dir-steering-ds4 QWEN_DIR_STEERING_LIMIT=$(QWEN_DIR_STEERING_SWEEP_LIMIT) QWEN_DIR_STEERING_PROFILE_IDS=$(QWEN_DIR_STEERING_SWEEP_PROFILE_IDS)

qwen-dir-steering-sweep-ipv4:  ## Run Qwen activation hook layer/scale sweep through direct IPv4
	$(MAKE) qwen-dir-steering-sweep INVENTORY=inventory.ipv4.ini ANSIBLE_EXTRA='--private-key "$(DGX_SSH_KEY)"'

qwen-dir-steering-fb-reader:  ## Run fb-reader replay: Qwen DFlash vs steered Qwen
	$(ANSIBLE) $(ANSIBLE_ARGS) playbooks/qwen-dir-steering-fb-reader.yml --extra-vars 'fb_reader_ab_corpus=$(FB_READER_AB_CORPUS) fb_reader_ab_limit=$(FB_READER_AB_LIMIT) fb_reader_ab_stance_ids=$(FB_READER_AB_STANCE_IDS) qwen_dir_steering_profile_id=$(QWEN_DIR_STEERING_FB_READER_PROFILE_ID) qwen_dir_steering_directions_path=$(QWEN_DIR_STEERING_DIRECTIONS)'

qwen-dir-steering-fb-reader-ipv4:  ## Run fb-reader steered-Qwen replay through direct IPv4
	$(MAKE) qwen-dir-steering-fb-reader INVENTORY=inventory.ipv4.ini ANSIBLE_EXTRA='--private-key "$(DGX_SSH_KEY)"'

qwen-dir-steering-extraction-corpus:  ## Build Qwen dir-steering extraction corpus from manual DS4 review
	python3 scripts/build_qwen_dir_steering_extraction_corpus.py --manual-review "$(QWEN_DIR_STEERING_EXTRACT_MANUAL)" --output "$(QWEN_DIR_STEERING_EXTRACT_CORPUS)"

qwen-dir-steering-extract: qwen-dir-steering-extraction-corpus  ## Run offline Qwen hidden-state direction extraction smoke
	$(ANSIBLE) $(ANSIBLE_ARGS) playbooks/qwen-dir-steering-extract.yml --extra-vars 'qwen_dir_steering_extract_corpus_src=$(CURDIR)/$(QWEN_DIR_STEERING_EXTRACT_CORPUS) qwen_dir_steering_extract_max_items=$(QWEN_DIR_STEERING_EXTRACT_MAX_ITEMS) qwen_dir_steering_extract_layers=$(QWEN_DIR_STEERING_EXTRACT_LAYERS) qwen_dir_steering_extract_max_length=$(QWEN_DIR_STEERING_EXTRACT_MAX_LENGTH) qwen_dir_steering_extract_install_deps=$(QWEN_DIR_STEERING_EXTRACT_INSTALL_DEPS)'

qwen-dir-steering-extract-ipv4:  ## Run offline Qwen hidden-state extraction through direct IPv4
	$(MAKE) qwen-dir-steering-extract INVENTORY=inventory.ipv4.ini ANSIBLE_EXTRA='--private-key "$(DGX_SSH_KEY)"'

gemma-mtp-endpoint-parity-prhead:  ## Compare Gemma4 /v1/completions vs /v1/chat/completions with PR-head
	$(ANSIBLE) $(ANSIBLE_ARGS) playbooks/gemma-mtp-speed-matrix.yml --extra-vars 'gemma_mtp_profile_ids=endpoint-parity-g4-u085 gemma_mtp_patch_url=$(GEMMA_MTP_PRHEAD_URL) gemma_mtp_patch_host_path=$(GEMMA_MTP_PRHEAD_REMOTE)'

gemma-mtp-endpoint-parity-prhead-ipv4:  ## Run PR-head endpoint parity through direct IPv4
	$(MAKE) gemma-mtp-endpoint-parity-prhead INVENTORY=inventory.ipv4.ini ANSIBLE_EXTRA='--private-key "$(DGX_SSH_KEY)"'

gemma-mtp-speed-matrix:  ## Run Gemma4 MTP decode + stance risk speed profiles
	$(ANSIBLE) $(ANSIBLE_ARGS) playbooks/gemma-mtp-speed-matrix.yml

gemma-mtp-speed-matrix-ipv4:  ## Run Gemma4 MTP speed profiles through direct IPv4
	$(MAKE) gemma-mtp-speed-matrix INVENTORY=inventory.ipv4.ini ANSIBLE_EXTRA='--private-key "$(DGX_SSH_KEY)"'

gemma-mtp-speed-targeted:  ## Run Gemma4 prodctx-g1 and fastctx-g4 after launch-path changes
	$(ANSIBLE) $(ANSIBLE_ARGS) playbooks/gemma-mtp-speed-matrix.yml --extra-vars 'gemma_mtp_profile_ids=prodctx-g1-u055,fastctx-g4-u085'

gemma-mtp-speed-targeted-ipv4:  ## Run targeted Gemma4 speed profiles through direct IPv4
	$(MAKE) gemma-mtp-speed-targeted INVENTORY=inventory.ipv4.ini ANSIBLE_EXTRA='--private-key "$(DGX_SSH_KEY)"'

gemma-mtp-speed-targeted-prhead:  ## Run targeted Gemma4 speed/stance profiles with PR-head gemma4_mtp.py
	$(ANSIBLE) $(ANSIBLE_ARGS) playbooks/gemma-mtp-speed-matrix.yml --extra-vars 'gemma_mtp_profile_ids=$(GEMMA_MTP_PRHEAD_STANCE_PROFILES) gemma_mtp_patch_url=$(GEMMA_MTP_PRHEAD_URL) gemma_mtp_patch_host_path=$(GEMMA_MTP_PRHEAD_REMOTE)'

gemma-mtp-speed-targeted-prhead-ipv4:  ## Run PR-head targeted speed/stance profiles through direct IPv4
	$(MAKE) gemma-mtp-speed-targeted-prhead INVENTORY=inventory.ipv4.ini ANSIBLE_EXTRA='--private-key "$(DGX_SSH_KEY)"'

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
		-a "systemctl is-active vllm; systemctl is-active vllm-pna-proxy; curl -s http://localhost:8000/v1/models | head -c 200" \
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
	$(ANSIBLE) $(ANSIBLE_ARGS) playbooks/qwen-dir-steering-ds4.yml --syntax-check
	$(ANSIBLE) $(ANSIBLE_ARGS) playbooks/qwen-dir-steering-extract.yml --syntax-check
	$(ANSIBLE) $(ANSIBLE_ARGS) playbooks/qwen-dir-steering-fb-reader.yml --syntax-check
	$(ANSIBLE) $(ANSIBLE_ARGS) playbooks/qwen-conditional-prompt-gate.yml --syntax-check
	$(ANSIBLE) $(ANSIBLE_ARGS) playbooks/fb-reader-ab-prhead.yml --syntax-check
	$(ANSIBLE) $(ANSIBLE_ARGS) playbooks/tierb-nvfp4-candidates.yml --syntax-check
	$(ANSIBLE) $(ANSIBLE_ARGS) playbooks/stance-v2-ab-prhead.yml --syntax-check
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
