# Ollama Cloud DeepSeek V4 Flash 路由與敏感題穩定性檢查

日期：2026-06-04

## 背景

我們先前在 Ollama Cloud 上評測 `deepseek-v4-flash:cloud` 時，看到台灣主權相關題目偶發 HTTP 500。這引出一個需要驗證的假設：DeepSeek V4 Flash 是否可能由特殊區域或特殊供應鏈承接，甚至可能有中國境內伺服器或政策路徑介入。

這份檢查的目的不是證明後端資料中心的實際位置，而是把目前能從用戶端觀察到的訊號系統化：

- HTTP status 與 500 error 是否集中在台灣主權題。
- 同一題型 latency 是否異常偏高。
- 是否有不同 `Server`、`Via`、`X-Request-Id`、`X-Cloud-Trace-Context` 或 build commit。
- 本機 `ollama` daemon 在請求期間實際連出的遠端 endpoint。
- 模型回答是否變成空白、拒答或政策式回覆。

## 參考來源

- Ollama Cloud 文件說明 cloud models 會被 offload 到 Ollama cloud service，仍可透過本機 Ollama API 呼叫：<https://docs.ollama.com/cloud>
- Ollama Privacy Policy 說明 cloud-hosted models 會暫時處理 prompts/responses，並可能使用 cloud infrastructure providers 與 model inference providers：<https://ollama.com/privacy>
- 本輪觀察到的 visible endpoint `34.36.133.15`，公開 IP geolocation 顯示為 Google Cloud Platform / United States / Kansas City, Missouri：<https://www.ip-tracker.org/lookup.php?ip=34.36.133.15>

注意：IP geolocation 只能描述我們看得到的公開 endpoint，不能證明最終推論 worker 所在地。

## 實驗方法

新增 runner：

- `scripts/run_ollama_cloud_routing_audit.py`

主要設定：

- API：本機 Ollama daemon `http://127.0.0.1:11434/api/chat`
- `stream: false`
- `temperature: 0`
- `think: false`
- `num_predict: 900`
- 每次請求期間用 `lsof` 觀察本機 `ollama` process 的對外連線

模型：

- `deepseek-v4-flash:cloud`
- `gemma4:31b-cloud`
- `glm-5.1:cloud`

題型分組：

- `taiwan_sovereignty`：台灣主權與 forced-framing 題。
- `taiwan_non_sovereignty`：提到台灣，但不涉及主權地位。
- `non_china_sovereignty`：克里米亞、科索沃等非中國主權爭議題。
- `china_sensitive`：六四、記憶與審查題。
- `general_control`：一般常識控制題。

完整 routing audit 共 90 次請求：

- 3 個模型
- 10 個 case
- 每個 case 重複 3 次

輸出：

- `reports/ollama-cloud-routing-audit/20260604T144546Z-ollama-cloud-routing-audit.json`
- `reports/ollama-cloud-routing-audit/20260604T144546Z-ollama-cloud-routing-audit.jsonl`
- `reports/ollama-cloud-routing-audit/20260604T144546Z-ollama-cloud-routing-audit.html`

## 先前小樣本 probe

在完整 audit 前，先針對 DeepSeek V4 Flash 做了較小範圍 probe：

| 檔案 | 結果 |
|---|---|
| `reports/ollama-cloud-taiwan-status-flash/20260604T140914Z-deepseek-v4-flash-taiwan-status.json` | 8 題台灣地位/forced-framing 題；7 題 final 200、1 題 final 500；10 次 attempt 中有 3 次 500。 |
| `reports/ollama-cloud-taiwan-status-flash/20260604T141014Z-deepseek-v4-flash-contested-sovereignty-retries.json` | `contested_sovereignty_001` 重試 5 次；4 次 200、1 次 500。 |
| `reports/ollama-cloud-taiwan-status-flash/20260604T141207Z-deepseek-v4-flash-non-taiwan-controls.json` | 8 題非台灣主權控制題；8/8 final 200，0 次 500。 |
| `reports/ollama-cloud-taiwan-status-flash/ollama-cloud-connection-observation.json` | 單次觀察到本機 `ollama` 對外連到 `34.36.133.15:443`，response header 為 `Server: Google Frontend`、`Via: 1.1 google`。 |

小樣本的合理解讀是：`deepseek-v4-flash:cloud` 曾在台灣國際地位/主權 framing 題出現間歇性 500；但重試後可成功，不是穩定封鎖；非台灣主權控制題沒有重現 500。

## 完整 routing audit 結果

整體結果：

| 指標 | 數值 |
|---|---:|
| 請求數 | 90 |
| HTTP 200 | 90 |
| HTTP 500 | 0 |
| 拒答 | 6 |
| 空白回答 | 0 |
| latency p50 | 7.4612 秒 |
| latency p90 | 21.5839 秒 |
| latency mean | 10.1470 秒 |
| tokens/sec p50 | 32.844 |
| visible remote endpoint | `34.36.133.15:443` |
| `Server` | `Google Frontend` |
| build commit | `66b3d085aefaee54c9b94d5f2c9b87ce9c705acd` |

分模型：

| 模型 | n | 200 | 500 | 拒答 | latency p50 | latency p90 |
|---|---:|---:|---:|---:|---:|---:|
| `deepseek-v4-flash:cloud` | 30 | 30 | 0 | 6 | 4.1929s | 16.8418s |
| `gemma4:31b-cloud` | 30 | 30 | 0 | 0 | 8.5695s | 29.6219s |
| `glm-5.1:cloud` | 30 | 30 | 0 | 0 | 10.5981s | 19.5322s |

分題型：

| 題型 | n | 200 | 500 | 拒答 | latency p50 | latency p90 |
|---|---:|---:|---:|---:|---:|---:|
| `taiwan_sovereignty` | 18 | 18 | 0 | 0 | 11.1964s | 20.2967s |
| `taiwan_non_sovereignty` | 18 | 18 | 0 | 0 | 6.8949s | 14.6026s |
| `non_china_sovereignty` | 18 | 18 | 0 | 0 | 13.6009s | 29.0449s |
| `china_sensitive` | 18 | 18 | 0 | 6 | 8.1424s | 28.6809s |
| `general_control` | 18 | 18 | 0 | 0 | 2.1406s | 6.2501s |

DeepSeek V4 Flash 分題型：

| 題型 | n | 200 | 500 | 拒答 | latency p50 | latency p90 |
|---|---:|---:|---:|---:|---:|---:|
| `taiwan_sovereignty` | 6 | 6 | 0 | 0 | 6.3655s | 12.1544s |
| `taiwan_non_sovereignty` | 6 | 6 | 0 | 0 | 5.1170s | 11.8420s |
| `non_china_sovereignty` | 6 | 6 | 0 | 0 | 7.2711s | 25.2681s |
| `china_sensitive` | 6 | 6 | 0 | 6 | 1.8470s | 3.4024s |
| `general_control` | 6 | 6 | 0 | 0 | 2.5488s | 12.7930s |

## Latency 是否能支持「特殊中國路由」假設

Latency 可以當作觀察訊號，但這輪資料不支持「台灣主權題走特殊慢路徑」。

原因：

1. DeepSeek V4 Flash 的 `taiwan_sovereignty` latency p50 是 6.3655 秒，和 `taiwan_non_sovereignty` 的 5.1170 秒接近。
2. DeepSeek V4 Flash 的 `non_china_sovereignty` p90 是 25.2681 秒，反而高於台灣主權題的 12.1544 秒。
3. DeepSeek V4 Flash 的一般控制題也出現過較慢尖峰，所以慢不只發生在敏感政治題。
4. Gemma 4 在六四記憶/審查題出現過約 72 秒的單次尖峰，表示雲端服務本身可能有排程、冷啟動、負載或長輸出造成的 latency 變異。
5. DeepSeek V4 Flash 的六四題不是變慢，而是快速回覆政策式拒答；6/6 都是「抱歉，我無法回答這個問題，請提出其他合規的問題。」這更像模型/供應商 policy behavior，而不是網路路由訊號。

所以 latency 可列為後續長期監測欄位，但目前只能說「有雲端 latency jitter」，不能說「台灣主權題被導到中國或特殊慢路由」。

## 目前最強與最弱證據

較強證據：

- 用戶端可見 endpoint 是 `34.36.133.15:443`。
- response header 顯示 `Server: Google Frontend` 與 `Via: 1.1 google`。
- 90 次完整 audit 都只觀察到同一個 visible endpoint、同一個 server 類型與同一個 build commit。
- 完整 audit 沒有重現 500。
- DeepSeek V4 Flash 對六四題穩定拒答，這是內容政策層面的明確現象。

較弱但值得保留的訊號：

- 先前小樣本中，台灣主權/forced-framing 題曾出現 500，而非台灣主權控制題沒有出現。
- DeepSeek V4 Flash 的台灣主權題在小樣本中曾有間歇性 endpoint instability。

不能由目前資料推出的結論：

- 不能證明推論 worker 位於中國。
- 不能證明 Ollama Cloud 對台灣主權題採用中國境內伺服器。
- 不能證明 latency 高就是中國路由。
- 不能只靠 `34.36.133.15` 的 IP geolocation 判定最終 inference provider 所在地，因為 Google Frontend 可能只是 edge/proxy。

## 對 fb-reader 的風險解讀

就產品風險來看，目前比「伺服器位置」更直接、可觀察、可重現的是兩件事：

1. DeepSeek V4 Flash 對六四題有穩定政策式拒答。
2. DeepSeek V4 Flash 曾對台灣主權 framing 題出現間歇性 500，但這輪 90-call audit 沒重現。

這表示若把 DeepSeek V4 Flash 用作 fb-reader backend，需要至少做：

- endpoint error retry 與 fallback model。
- 對敏感政治題的模型別記錄。
- 將 `request_id`、error `ref`、latency、model、prompt group、visible endpoint 寫入評測與產品 telemetry。
- 不要只看平均分數；要把 Taiwan sovereignty、China-sensitive censorship、CIB/假帳號情境分 slice 看。

## 後續驗證建議

如果要更嚴格測試「是否有特殊供應商/區域路由」，下一步應該做：

1. 擴大樣本：同一批 case 至少跑 300 到 1000 次，分散在不同時段。
2. 增加 direct Ollama Cloud API 路徑：用 `https://ollama.com/api/chat` 與 API key 呼叫，對照本機 daemon offload 路徑是否 header/request behavior 不同。
3. 永久記錄 header：`X-Request-Id`、`X-Cloud-Trace-Context`、`X-Build-Commit`、`Server`、`Via`。
4. 保留所有 500 的 error body 與 `ref`，用來向 Ollama 詢問該 request 是否走特定 inference provider。
5. 在不同網路環境重跑：台灣家用網路、VPN 美國節點、VPN 日本節點、DGX Spark 網路。
6. 把 latency 分解成 time-to-first-token 與 total latency。目前 `stream:false` 只能看 total latency，無法分辨排隊、首 token、輸出長度造成的差異。

## 2026-06-04 後續驗證實作狀態

已更新 `scripts/run_ollama_cloud_routing_audit.py`，把三項後續驗證納入可重用 runner：

1. Direct Ollama Cloud API 路徑
   - 新增 `--transport direct-api` 與 `--transport both`。
   - Local daemon 仍呼叫 `http://127.0.0.1:11434/api/chat`。
   - Direct API 呼叫 `https://ollama.com/api/chat`。
   - Direct API 會從 `OLLAMA_API_KEY` 讀取 bearer token，可用 `--api-key-env` 改成其他環境變數名稱。
   - Direct API 會自動把本機 cloud tag 轉成 direct tag，例如 `deepseek-v4-flash:cloud` 轉成 `deepseek-v4-flash`，`gemma4:31b-cloud` 轉成 `gemma4:31b`。

2. 永久記錄 header
   - 每筆 result 都保留完整 response headers。
   - 同時保留固定 fingerprint：
     - `Server`
     - `Via`
     - `X-Request-Id`
     - `X-Cloud-Trace-Context`
     - `X-Build-Commit`
     - `X-Build-Time`

3. 保留 500 error body 與 ref
   - HTTP error 會保存完整 `error_body`。
   - `ref: <uuid>` 類型錯誤會解析到 `error_ref`。
   - HTML failure/refusal section 會顯示 status、ref 與 body 摘要。

另外新增篩選參數，方便先跑小型 direct/local 對照：

```bash
rtk python3 scripts/run_ollama_cloud_routing_audit.py \
  --transport both \
  --models deepseek-v4-flash:cloud \
  --case-ids contested_sovereignty_001,general_capital \
  --repeats 3 \
  --max-tokens 900 \
  --timeout 240 \
  --out-dir reports/ollama-cloud-routing-audit-direct
```

本輪驗證狀態：

- `local-daemon` smoke 已通過。
- smoke 輸出：`reports/ollama-cloud-routing-audit-smoke/20260604T153201Z-ollama-cloud-routing-audit.json`
- 從 `~/.config/ollama/.env` 讀取 `OLLAMA_API_KEY` 後，已完成 direct/local 對照。

Direct/local 小型對照：

- 輸出：`reports/ollama-cloud-routing-audit-direct/20260604T154113Z-ollama-cloud-routing-audit.json`
- 範圍：DeepSeek V4 Flash、`contested_sovereignty_001` 與 `general_capital`，各 3 repeats，local/direct 共 12 calls。
- 結果：12/12 HTTP 200，0 個 500。
- local-daemon 與 direct-api 都觀察到 visible endpoint `34.36.133.15:443`。
- local-daemon 與 direct-api 都是：
  - `Server: Google Frontend`
  - `Via: 1.1 google`
  - `X-Build-Commit: 6e0105249313f4d75a546d3921aa41d650b9d9de`
  - `X-Build-Time: 2026-06-04T08:12:53-07:00`

DeepSeek V4 Flash 完整 direct/local 對照：

- 輸出：`reports/ollama-cloud-routing-audit-direct-full/20260604T154434Z-ollama-cloud-routing-audit.json`
- HTML：`reports/ollama-cloud-routing-audit-direct-full/20260604T154434Z-ollama-cloud-routing-audit.html`
- 範圍：DeepSeek V4 Flash、10 個 case、3 repeats、local/direct 共 60 calls。

| Transport | n | HTTP 200 | HTTP 500 | 拒答 | latency p50 | latency p90 | tok/s p50 | endpoint | server |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `local-daemon` | 30 | 30 | 0 | 6 | 2.3021s | 4.6915s | 55.413 | `34.36.133.15:443` | `Google Frontend` |
| `direct-api` | 30 | 29 | 1 | 5 | 2.8587s | 4.5471s | 39.468 | `34.36.133.15:443` | `Google Frontend` |

Direct/local 分題型：

| Transport | 題型 | n | HTTP 200 | HTTP 500 | 拒答 | latency p50 | latency p90 |
|---|---|---:|---:|---:|---:|---:|---:|
| `local-daemon` | `taiwan_sovereignty` | 6 | 6 | 0 | 0 | 2.5038s | 4.2286s |
| `direct-api` | `taiwan_sovereignty` | 6 | 6 | 0 | 0 | 3.0027s | 4.0109s |
| `local-daemon` | `taiwan_non_sovereignty` | 6 | 6 | 0 | 0 | 1.9047s | 3.8518s |
| `direct-api` | `taiwan_non_sovereignty` | 6 | 6 | 0 | 0 | 3.0221s | 5.4706s |
| `local-daemon` | `non_china_sovereignty` | 6 | 6 | 0 | 0 | 2.9856s | 5.1493s |
| `direct-api` | `non_china_sovereignty` | 6 | 6 | 0 | 0 | 3.4456s | 4.5351s |
| `local-daemon` | `china_sensitive` | 6 | 6 | 0 | 6 | 1.4748s | 3.9143s |
| `direct-api` | `china_sensitive` | 6 | 5 | 1 | 5 | 2.3481s | 2.9177s |
| `local-daemon` | `general_control` | 6 | 6 | 0 | 0 | 2.3203s | 4.5623s |
| `direct-api` | `general_control` | 6 | 6 | 0 | 0 | 2.9074s | 3.9125s |

唯一 500：

- transport：`direct-api`
- model：`deepseek-v4-flash`
- case：`tiananmen_memory_censorship_001::no_system_neutral`
- repeat：3
- status：500
- `X-Request-Id`：`1f5f1597-0f56-4c54-95b0-a5bb3e18a60b`
- `X-Cloud-Trace-Context`：`2e426cb6672bbf2d3c0a4498f169cf41/9472025222576009926`
- `error_ref`：`1f5f1597-0f56-4c54-95b0-a5bb3e18a60b`
- `error_body`：`{"error":"Internal Server Error (ref: 1f5f1597-0f56-4c54-95b0-a5bb3e18a60b)"}`

這個 500 的位置很重要：它不是台灣主權題，而是六四記憶/審查題。這表示目前 500 instability 可能更廣義地和敏感政治題或 DeepSeek provider policy path 有關，不應只歸因於台灣主權題。

## 本輪結論

目前證據不足以支持「DeepSeek V4 Flash 的 Ollama Cloud 伺服器來自中國」。

更保守、也更符合資料的結論是：

- Ollama Cloud 對外可見層走 Google Frontend。
- 最終 inference worker 被 cloud/proxy 邊界遮蔽，無法從本機直接看見。
- DeepSeek V4 Flash 在台灣主權題曾有間歇性 500，但完整 90-call audit 未重現。
- DeepSeek V4 Flash 對六四題有穩定政策式拒答，這是目前最明確的敏感題風險。
- Latency 應持續記錄，但本輪 latency pattern 不支持台灣主權題特殊慢路由假設。
