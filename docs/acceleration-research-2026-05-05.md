# DGX Spark LLM 推論加速研究

**日期**：2026-05-05
**模型**：glm-5.1:cloud (Anthropic Claude, via Claude Code)

---

## 架構背景

| 機器 | 框架 | 模型 | 目前速度 |
|---|---|---|---|
| Mac M4 Max | Ollama | gemma4:e4b | 78 tok/s（頻寬天花板） |
| DGX Spark (GB10) | vLLM | Intel AutoRound INT4 Qwen3.6-35B-A3B + DFlash k=8 | 待優化 |

重要變更：DGX Spark 已移除 Ollama，vLLM 獨佔 GPU。Mac 上的 local-inference 專案負責 Ollama + Gemma 4 E4B。

---

## 一、DGX Spark vLLM 優化（影響力排序）

### 1.1 調高 GPU memory utilization：0.65 → 0.85~0.90

**影響力：最大單一改動。**

Ollama 不再佔用 DGX 記憶體，vLLM 可獨佔 GPU。目前 `vllm_gpu_memory_utilization: "0.65"` 是為了跟 Ollama 共存而壓低的。更多 KV cache 空間直接轉化為更高吞吐。

建議：先試 0.85，用 `make benchmark-vllm` 測試穩定後再試 0.90。

### 1.2 切換 FlashInfer attention 後端

**影響力：中高。**

目前使用 `--attention-backend flash_attn`，FlashInfer 在以下場景更快：
- Batched decode + chunked prefill（正是目前的設定）
- 支援 FP8 KV cache（未來 sm_121 kernel 到了就能用）
- 更好的記憶體效率

```
--attention-backend flashinfer
```

風險：GB10/Blackwell 上 FlashInfer 未經充分測試，可能需要回退到 flash_attn。

參考：vLLM 0.7+ 官方推薦 FlashInfer 作為預設 backend，特別是搭配 chunked prefill。

### 1.3 加 speculative-disable-by-batch-size

**影響力：中低（防護性）。**

```
--speculative-disable-by-batch-size 4
```

當 batch 滿載（4 sequences）時，speculative decoding 收益遞減，自動停用避免開銷。搭配目前的 `--max-num-seqs 4`。

### 1.4 加 swap-space 安全網

**影響力：低（防護性）。**

```
--swap-space 2
```

2 GiB CPU swap，長 context 或記憶體碎片時防止 OOM 崩潰。CPU swap 延遲懲罰 ~10-50ms/block，但比整個 process 死掉好。

### 1.5 嘗試 PrismaQuant 4.75bit 路徑

**影響力：可能最高（95+ tok/s）。**

DGX 上已有模型檔案。阻塞原因是 vLLM V1 engine 在 Blackwell 上不支援 `compressed-tensors`（PrismaQuant 的量化格式），拋 `NotImplementedError`。

嘗試解法：
```
Environment="VLLM_USE_V1=0"
```

強制 V0 engine。如果可行，搭配 DFlash k=6 speculative decoding，理論可達 95+ tok/s（目前 AutoRound INT4 + DFlash k=8 的基準未知，但預期顯著提升）。

參考：`docs/handover-prismaquant.md`

### 1.6 測試 EAGLE3 取代 DFlash

**影響力：中（1.4-1.9x 加速，但有 TTFT trade-off）。**

vLLM 官方已支援 Qwen3 EAGLE3（PR #21835，2025/8 合併）。AngelSlim 在 H20 GPU 的基準測試：

| 模型 | Vanilla tok/s | EAGLE3 tok/s | 加速比 | Accept Length |
|---|---|---|---|---|
| Qwen3-30B-A3B | 320.87 | 438.07 | 1.36x | 2.04 |
| Qwen3-32B | 43.32 | 74.10 | 1.71x | 1.91 |

**但 TTFT 退化嚴重**（issue #39790）：P99 TTFT 惡化 2-4x。如果 fb-reader 的使用模式是短 prompt + 長生成 → 值得試；長 prompt + 短生成 → TTFT 退化可能是問題。

EAGLE3 的優勢是使用模型自身的 drafting head，不需要獨立 draft model（DFlash 需要 z-lab/Qwen3.6-35B-A3B-DFlash）。

### 1.7 Dynamic Speculation Length（PR #35301）

**影響力：高（最多 3x），但尚未合併。**

信心門檻早期退出機制：draft model 每步計算 argmax token 的 softmax 機率，當信心低於閾值時停止 drafting，避免浪費計算。

A100 基準測試結果（Llama-3.1-8B）：
- MT-Bench c=1, τ=0.8: **1.80x** vs baseline, **2.74x** vs static SD
- Random c=8, τ=0.6: **3.00x** vs baseline

PR 仍在 review 中（截至 2026/3），未合併。值得持續追蹤。

---

## 二、DGX Spark KV Cache 量化：不要用

[Memoriant 基準測試](https://github.com/Memoriant/dgx-spark-kv-cache-benchmark) 證實了**統一記憶體悖論**：

GB10 的 128 GB LPDDR5X（~273 GB/s）不像獨立顯卡受 VRAM 容量壓力，但頻寬遠低於 HBM3（~3,350 GB/s）。KV cache 量化的解量化計算成本超過頻寬節省。

| Context | fp16 tok/s | q8_0 tok/s | q4_0 tok/s | q4_0 vs fp16 |
|---|---|---|---|---|
| ~1.5K | 45.2 | 45.3 | 45.6 | +0.9% |
| ~24K | 44.6 | 39.7 | 39.3 | **-11.9%** |
| ~110K | 38.0 | 25.0 | 24.0 | **-36.8%** |

TurboQuant（turbo3/turbo4）在 32K context 也慢 22-24%。

**結論：DGX Spark 上的 Ollama 和 vLLM 都應維持 fp16 KV cache。**

目前的 `ollama_kv_cache_type: "fp16"` 和 vLLM 預設都是正確的。不要改成 q8_0 或 q4_0。

**注意**：此結論僅適用於 DGX Spark（GB10）統一記憶體架構。在獨立顯卡（RTX 4080/4090/5090）上，q8_0 基本上是免費的（~0.1% 速度損失，2x VRAM 節省）。

---

## 三、Ollama Flash Attention 更新

### Gemma3 KV cache + Flash Attention 修復

Gemma3 4B/12B 在啟用 KV cache 量化 + flash attention 時極慢的 bug 已修復：
- 根因：缺少 quantized KV cache + flash attention 的 CUDA kernel → GPU/CPU 來回跳動
- 修復：[PR #12245](https://github.com/ollama/ollama/issues/9683)，**Ollama v0.12.5** 包含

### Qwen3 Flash Attention

Issue #12432 是 AMD ROCm 安裝路徑問題，NVIDIA 不受影響。修復在 v0.12.4-RC6+。

### 建議

升級 Ollama 到 >= 0.12.5 後，A/B 測試 `OLLAMA_FLASH_ATTENTION=1`。

Mac 端已確認 flash attention +1 對 gemma4:e4b 有 +4% 加速。
DGX 端（如果重新跑 Ollama）仍需獨立測試，因為之前記錄的 regression 可能已修復。

---

## 四、Mac M4 Max Ollama：已到硬體天花板

78 tok/s 已達 M4 Max 頻寬物理極限的 92%。已 A/B 測試過所有硬體層優化：

| 設定 | 結果 |
|---|---|
| `OLLAMA_FLASH_ATTENTION=1` | +4%（已啟用） |
| `OLLAMA_KV_CACHE_TYPE=q8_0` | -10%（已移除，維持 fp16） |
| `OLLAMA_NUM_PARALLEL=4` | 無吞吐增益，延遲 4x（已拒絕） |
| Speculative decoding (E2B→E4B) | -21%（已拒絕） |
| MLX runner | -10~-12%（已拒絕） |
| `gemma4:26b`（MoE 升級） | 同 78 tok/s，+8.4 GB RAM（已拒絕） |

唯一硬體突破路徑：**MTP（Multi-Token Prediction）**，需要社群把 Gemma 4 MTP 權重移植到 llama.cpp（Google 未公開 HF 版本的 MTP 權重）。

---

## 五、應用層加速（輸出格式壓縮）

### 目前格式

fb-reader 每篇貼文的分類輸出是完整 JSON 物件（7 個欄位），平均 ~51 output tokens：

```json
{
  "commercial": 0.9,
  "political": 0,
  "emotional": 0.2,
  "personal": 0,
  "source_type": "original",
  "unsourced_claim": false,
  "manipulation_techniques": [],
  "claim_snippets": [],
  "summary": "日文老師推銷線上課程並以留言互動導購"
}
```

### 4-digit encoding（已嘗試，需退回）

4 個數字字元代表 4 個軸分數，例如 `"5180"` → `[0.5, 0.1, 0.8, 0.0]`。Output tokens 從 51 降到 5，延遲從 ~0.6s 降到 ~0.25s。

**問題**：Gemma 4 E4B 不夠聰明，無法可靠地只用 4 位數字處理分類。已改回 JSON。

### JSON array 壓縮（中間方案）

把 JSON object 改成 JSON array，只保留 4 個軸分數：

```json
[0.9, 0, 0.2, 0]
```

- Output tokens：51 → ~26（減半）
- 延遲：0.65s → 0.55s
- 適合只需要 4 軸分數的路徑
- 仍然需要 `source_type`/`manipulation_techniques` 的 batch 路徑則維持完整 JSON object

### 其他可探索的壓縮方向

1. **簡化 summary**：summary 是最長的字串欄位。如果 fb-reader 不需要每篇貼文的 summary，移除可省 ~10-15 tokens。
2. **二元化分數**：把 0.0-1.0 浮點數改成 0/1 二元分類（如果應用場景可接受），從 `"commercial": 0.9` 變成 `"c": 1`，省更多 tokens。
3. **短鍵名**：`"commercial"` → `"c"`, `"political"` → `"p"`, `"emotional"` → `"e"`, `"personal"` → `"P"`。每個鍵名省 6-8 tokens。
4. **條件式欄位**：`manipulation_techniques` 和 `claim_snippets` 大多數時候是空陣列 `[]`。如果只在偵測到時才輸出，可省 ~5 tokens/次。
5. **混合策略**：4 軸用 array `[0.9, 0, 0.2, 0]`，IQ 欄位用條件式輸出。

### think: false（已部署，必須）

Ollama API 層級的 `think: false` 是 Gemma 4 的**強制設定**。即使 system prompt 有 `/no_think`，Gemma 4 仍會洩漏 hidden CoT tokens。只有 API 層級的 flag 能完全阻止。已部署在所有呼叫端。

---

## 六、中短期追蹤項目

| 項目 | 潛力 | 狀態 | 追蹤 |
|---|---|---|---|
| PrismaQuant 4.75bit + DFlash k=6 | 95+ tok/s | 阻塞：V1 engine 不支援 compressed-tensors | 試 `VLLM_USE_V1=0` |
| NVFP4 via TRT-LLM | 2.6x over FP8 | TRT-LLM 1.3.0rc12 對 Qwen3-MoE 壞的 | 等 TRT-LLM 1.4+ |
| vLLM FP8 sm_121 kernel | ~1.5-2x over INT4 | CUTLASS 缺 sm_121 specialization | vLLM GitHub issues |
| SGLang on GB10 | 未知 | sm_121 不支援 | SGLang GitHub issues |
| Dynamic Speculation Length (PR #35301) | 最多 3x | PR 仍在 review | vLLM PR status |
| Gemma 4 MTP in llama.cpp | Mac 突破 | Google 未公開 MTP 權重 | llama.cpp community |
| vLLM FP8 KV cache on GB10 | 未知 | FlashInfer FP8 KV cache 需 sm_121 kernel | vLLM + CUTLASS 更新 |

---

## 七、NVIDIA 官方 DGX Spark 優化參考

[NVIDIA 技術部落格（2026/1）](https://developer.nvidia.com/blog/new-software-and-model-optimizations-supercharge-nvidia-dgx-spark/) 重點：

- **NVFP4 量化**：記憶體減少 ~40%，Qwen-235B 在雙 DGX Spark 上達 2.6x 加速（vs FP8）
- **llama.cpp MoE 優化**：平均 35% 效能提升（直接影響 Ollama 的 MoE 模型）
- **EAGLE-3**：內建 drafting head，不需要獨立 draft model
- **TensorRT-LLM** 已支援 Qwen3 8B/14B/32B/30B-A3B（FP8/NVFP4）
- **NVFP4 的硬體加速解量化**：使用 Blackwell Tensor Core 專用矽片而非軟體 kernel，這是 KV cache 量化的正確路徑（不像 q4_0/q8_0 走軟體解量化）

---

## 八、立即可做的行動清單

### DGX vLLM（高影響）

1. `vllm_gpu_memory_utilization` 從 0.65 提高到 0.85，A/B 測試後再試 0.90
2. 加 `--attention-backend flashinfer`，測試穩定性
3. 加 `--speculative-disable-by-batch-size 4`
4. 加 `--swap-space 2`
5. 嘗試 PrismaQuant：加 `VLLM_USE_V1=0` 環境變數

### Mac Ollama（已到天花板）

6. 確認 `OLLAMA_FLASH_ATTENTION=1` 和 `think: false` 持續啟用
7. 探索 JSON array 壓縮或其他輸出格式縮減方案

### 通用

8. 升級 Ollama 到 >= 0.12.5（Mac 端），確認 llama.cpp MoE 35% 優化已包含
9. 追蹤 PrismaQuant V0 engine 和 TRT-LLM 1.4+ 的進度

---

## 參考來源

- [Memoriant DGX Spark KV Cache Benchmark](https://github.com/Memoriant/dgx-spark-kv-cache-benchmark)
- [NVIDIA DGX Spark Optimization Blog (2026/1)](https://developer.nvidia.com/blog/new-software-and-model-optimizations-supercharge-nvidia-dgx-spark/)
- [vLLM Speculative Decoding Documentation](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
- [vLLM EAGLE3 Qwen3 Support PR #21835](https://github.com/vllm-project/vllm/pull/21835)
- [vLLM Dynamic Speculation Length PR #35301](https://github.com/vllm-project/vllm/pull/35301)
- [vLLM EAGLE3 TTFT Regression Issue #39790](https://github.com/vllm-project/vllm/issues/39790)
- [vLLM Speculators Library](https://github.com/vllm-project/speculators)
- [kvcache-bench Tool](https://github.com/back2matching/kvcache-bench)
- [Ollama Gemma3 KV Cache Fix Issue #9683](https://github.com/ollama/ollama/issues/9683)
- [Ollama Qwen3-2507 Flash Attention Issue #12432](https://github.com/ollama/ollama/issues/12432)
- [KV Cache Quantization in Ollama (smcleod.net)](https://smcleod.net/2024/12/bringing-k/v-context-quantisation-to-ollama/)
- [TensorRT-LLM DGX Spark Beta Commit](https://github.com/NVIDIA/TensorRT-LLM/commit/aa410c57bcea620c765daf7b698c29eac7cd686b)
- [AWS Speculative Decoding on Trainium Blog](https://aws.amazon.com/blogs/machine-learning/accelerating-decode-heavy-llm-inference-with-speculative-decoding-on-aws-trainium-and-vllm/)
- [MMSpec VLM Speculative Decoding Benchmark (arXiv:2603.14989)](https://arxiv.org/abs/2603.14989v1)