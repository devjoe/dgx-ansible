# Facebook 貼文參考稿：一次關於本地 AI 後端校準的實驗

> 目的：這份文件不是正式報告，而是協助整理成 Facebook 貼文的參考稿。內容可以再改成更個人化、更口語的語氣。

## 貼文草稿

最近花了一些時間，在 DGX Spark 上測試不同本地 AI 模型與提示方式，目標是替 `fb-reader` 找一個更可靠的後端。

`fb-reader` 想解決的問題不是單純「模型會不會回答」，而是它能不能幫讀者判斷社群媒體內容：哪些是可驗證的事實，哪些是政治立場，哪些是框架帶風向，哪些又可能涉及協同行為或來源脈絡。

這裡最困難的地方是：模型不能只是「永遠中立」。

如果題目本身真的有爭議，例如台灣國際地位、能源政策、政治不實資訊、官媒敘事或疑似 CIB 行為，模型應該提醒讀者看到多方立場、證據強弱與資訊缺口。

但如果題目其實是 settled fact，例如「東京是不是日本首都」、「南極洲是不是最南端的大陸」、「西西里島是不是屬於義大利」，模型就不應該過度補上主權、歷史、法律或地方政治 caveat。那會讓讀者誤以為簡單事實也處於高度爭議狀態。

我們這次的實驗，就是在找這個平衡點。

一開始我參考了 Audrey Tang 的 DS4 dir-steering 實驗。那個方向很有啟發性：它不是只靠 prompt，而是嘗試在模型內部 activation space 做 steering，讓模型比較容易進入「對爭議題保持謹慎」的狀態。不過，DS4 的 steering 向量不能直接搬到 Qwen；模型架構與 hidden dimension 都不同。我們先把問題拆成幾層：

1. 現在的 Qwen DFlash 能不能先當 baseline？
2. Gemma 4 MTP 是否在這類 stance / settled calibration 上明顯更好？
3. 如果 Qwen 有缺點，是 prompt 可以解決，還是真的需要 steering hook？
4. 修正 settled 題過度謹慎時，會不會反過來傷害台灣、CIB、主權爭議題的判讀？

測下來有幾個有趣結果。

第一，Qwen 在真正 contested 的題目上其實表現不差。DS4 的 120 題 contested prompts 裡，經過人工覆判，conditional prompt 的 Qwen 是 120/120 pass。也就是它沒有把主權、政策、價值衝突這些題目過早講成單一已定案答案。

第二，Qwen 的主要問題反而在 settled control。它有時會把簡單事實題講得太像政治或法律爭議。例如科西嘉、西西里、Bell 電話專利、東京首都法律 caveat、南極洲主權聲索等。答案不一定錯，但語氣太小心，會讓讀者讀起來像「這件事其實很不確定」。

第三，單純換一個更精準的 system prompt，效果已經很明顯。我們測了一個 conditional prompt，大意是：

> 如果問題是 settled factual answer，就直接簡短回答，不要發明 stakeholder debate、法律 caveat 或歷史爭議。
> 如果問題涉及 contested policy、主權、身分認同、來源歸因、操縱、CIB 或公共爭議，才公平呈現主要利害關係人的觀點與少數共識。

這個做法比「一律公平呈現所有 stakeholder」更好。因為後者容易把所有題目都變成爭議題；conditional prompt 則先判斷題型，再決定要直接回答還是展開脈絡。

第四，人工覆判很重要。自動規則一開始把 conditional prompt 的 settled-control 分數看成 111/120，但我們把 240 題全部人工讀完後，結果是：

- contested：120/120 pass
- settled-control：113 題 clean pass，7 題 watch，0 題 fail
- 那 7 題 watch 都是「事實回答正確，但 caveat 太重」，不是「答錯」或「立場崩壞」

我們另外把這 7 題做成 regression set，再測一次 current prompt 和 conditional prompt。結果 conditional prompt 把 settled-compatible 從 6/7 提升到 7/7，而且同時保住台灣與 CIB risk slice：

- Taiwan/CIB risk slice：8/8 compatible
- forced-frame adoption：0
- Taiwan-sensitive over-settlement：0

也就是說，至少在目前測到的範圍內，讓模型在 settled 題上更直接，沒有導致它在台灣、CIB、主權爭議題上變得草率。

這讓我目前的判斷是：先推進 prompt-only 的 conditional prompt，比急著上 steering hook 更務實。

Steering 仍然有研究價值。尤其如果我們想精準降低「settled 題過度 caveat」的語氣，activation steering 可能是適合的工具。但它也有風險：如果方向抓得太粗，可能把模型整體的謹慎性壓低，反而傷害真正 contested 的政治與 CIB 分析。

所以目前比較好的路線是：

1. 先把 conditional prompt 接進 `fb-reader` 的真實 Tier B prompt。
2. 用 50 個 captured Tier B cases 測 JSON/schema/latency。
3. 用 Taiwan/CIB risk slice 測 forced-frame 與 over-settlement。
4. 用 7 題 settled-watch regression 測模型是否又開始過度 caveat。
5. 只有在 prompt-only 不夠時，再考慮更窄的「settled-directness steering」。

這次實驗給我的最大收穫是：模型評估不能只問「哪個模型比較強」，也不能只看 tok/s 或 benchmark headline。

對社群媒體分析來說，更重要的問題是：

- 它知道什麼時候要直接回答嗎？
- 它知道什麼時候必須保留爭議性嗎？
- 它會不會被 loaded frame 帶著走？
- 它會不會把官媒或疑似協同敘事誤當成中立事實？
- 它能不能在不過度指控的前提下，呈現來源與證據風險？

這些都不是單一分數能回答的問題。最後仍然需要人工閱讀、分層測試，以及把失敗案例固定成 regression set。

目前我會把 Qwen DFlash 繼續視為短期 practical default，但搭配更嚴格的 prompt、claim prepass，以及 Taiwan/CIB 專用 gate。Gemma 4 仍然值得觀察，尤其在 settled/control calibration 上有優點；但至少在目前的 10 篇新聞全文與 prepass 測試裡，Qwen 並沒有被明顯拉開，而且它已經是部署中的路徑。

換句話說，這不是「哪個模型贏了」的故事，而是「如何讓模型知道何時該謹慎，何時不該裝作所有事都還有爭議」。

我覺得這會是社群媒體 AI 工具很核心的一課。

## 可引用數字

- DS4 full conditional gate：
  - `noop-dflash-current-prompt`：contested 120/120，settled-control 96/120。
  - `noop-dflash-conditional-prompt`：contested 120/120，settled-control 111/120。
  - `steer-l32-35-s020-conditional-prompt`：contested 119/120，自動規則下 settled-control 113/120；該 contested miss 人工看起來像規則誤判。

- `noop-dflash-conditional-prompt` 240 題人工覆判：
  - contested：120/120 pass。
  - settled-control：113 pass，7 watch，0 fail。
  - 7 個 watch：Corsica、Bell patent、Sicily、Corsica 中文題、Antarctica、Bell patent 中文題、Tokyo capital。

- 2026-05-22 conditional prompt gate：
  - Tier B replay：50 cases，49/50 parse/schema ok，0 timeout，p50 2.95s，p90 6.35s。
  - Taiwan/CIB risk slice current prompt：8/8 compatible，0 forced-frame adoption，0 Taiwan-sensitive over-settlement。
  - Taiwan/CIB risk slice conditional prompt：8/8 compatible，0 forced-frame adoption，0 Taiwan-sensitive over-settlement。
  - 7 題 settled-watch regression current prompt：6/7 settled compatible。
  - 7 題 settled-watch regression conditional prompt：7/7 settled compatible。

- 10 篇新聞全文 prepass 測試：
  - Qwen：10/10 manual pass。
  - Gemma：10/10 manual pass。
  - 手動閱讀沒有發現 material over-settlement 或 frame-adoption failure。
  - Qwen 保持 practical default，原因是已部署、新聞路徑略快，且 prepass 後 source-fidelity 表現可用。

## 寫作時可保留的 caveat

- 這是工程評估，不是模型能力的正式學術結論。
- DS4 corpus 是 calibration stress test，不等於真實新聞與社群貼文。
- 自動規則適合 triage，但不能取代人工閱讀。
- `fb-reader` 的 Tier B replay 目前仍使用既有 request body；conditional prompt 要真正進產品，還需要接進 `fb-reader` prompt 後再重跑。
- 不應用模型直接判定特定帳號是中國假帳號；比較合理的是呈現 source lineage、官方敘事相似度、同步訊號、證據強弱與 attribution confidence。

## 可能的短版開頭

最近在 DGX Spark 上測本地 AI 後端時，最有趣的發現不是「哪個模型比較強」，而是模型要學會兩種相反能力：真正有爭議的議題要保持謹慎，已經 settled 的事實則不要故意講得像還有很多爭議。

## 可能的短版結尾

我現在越來越覺得，社群媒體 AI 工具的關鍵不是讓模型永遠中立，而是讓它知道：什麼時候該保留脈絡，什麼時候該直接說事實，什麼時候該提醒讀者「這個框架本身正在帶你往某個方向走」。
