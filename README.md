# Splendor AlphaZero 人機對戰提示工具

這個小工具用來搭配 [`cestpasphoto/alpha-zero-general`](https://github.com/cestpasphoto/alpha-zero-general) 的 Splendor 預訓練模型，在人機對戰時先用同一套 MCTS 搜尋目前局面，顯示候選走法的：

- **估計勝率**：由 MCTS 的 action Q 值換算成約略勝率（兩人局約為 `(Q + 1) / 2`）。
- **訪問占比**：AlphaZero 搜尋後對各走法的 visit distribution。
- **訪問次數**：此走法在根節點被模擬探索的次數。
- **先驗機率**：神經網路原始 policy 對該走法的偏好。

> 注意：這不是修改棋規或幫你自動落子；它只是在人類輸入前列出「AI 認為較好」的走法，方便你在人機對戰中參考。

## 安裝 upstream engine

先另外準備 `cestpasphoto/alpha-zero-general`：

```bash
git clone https://github.com/cestpasphoto/alpha-zero-general.git ../alpha-zero-general
cd ../alpha-zero-general
pip install onnxruntime onnx onnxscript numba tqdm colorama coloredlogs
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
```

該 upstream repository 已包含 Splendor 的 `pretrained_2players.pt`、`pretrained_3players.pt`、`pretrained_4players.pt`。

### Windows / Python 3.13 常見錯誤：`No module named 'onnxscript'`

如果啟動本機 GUI 後瀏覽器或終端機出現：

```text
ModuleNotFoundError: No module named 'onnxscript'
```

代表新版 PyTorch 在匯出 ONNX inference model 時需要額外套件。請在同一個 Python 環境執行：

```bash
pip install onnxscript
```

如果仍有 ONNX 相關錯誤，建議更新這三個套件：

```bash
pip install --upgrade onnx onnxscript onnxruntime
```

### Windows / Python 3.13 常見錯誤：`adaptive_max_pool2d` ONNX conversion

如果你看到：

```text
No ONNX function found for aten.adaptive_max_pool2d
```

這通常是新版 PyTorch 預設使用 dynamo ONNX exporter，但 Splendor model 內的 `adaptive_max_pool2d` 在該 exporter 上轉換失敗。本專案預設會在啟動時改用 legacy ONNX exporter，所以請先更新到最新版後重新執行原本指令即可。

如果你有加 `--modern-onnx-export`，請先移除；只有在你明確想測試 PyTorch 新 exporter 時才需要它。



## 完整資料層整合版（同步建議 + 下棋 + Undo）

如果你要「不會不同步」的完整資料層整合，請使用本機整合版：

```bash
python tools/splendor_hint_gui.py \
  --azg-path ../alpha-zero-general \
  --checkpoint ../alpha-zero-general/splendor/pretrained_2players.pt \
  --top 8 \
  --numMCTSSims 200 \
  --integrated-card-ui
```

這個模式不使用官方 hosted iframe；它是「官方風格外觀，但由本機後端驅動」的同步前端，會透過 `tools/splendor_hint_gui.py` 的 `/api/state`、`/api/move`、`/api/undo` 讀寫同一份 Python 遊戲狀態。`/api/state` 會回傳結構化盤面資料，包括：

- 依 tier 分組的 visible cards。
- 貴族與 bank gems。
- 每位玩家的 gems、已購買卡、保留卡、貴族、分數。
- legal moves。
- AlphaZero/MCTS hint rows。

因此資料層是同步的：建議、下棋、AI 回合自動回應與 Undo 都會一起更新。若你需要可靠同步，建議使用 `--integrated-card-ui`；它比 `--official-card-hints` 更適合實際對局。

## 官方卡片外觀 + 建議走法（伴隨模式）

如果你想要官方 <https://cestpasphoto.github.io/splendor.html> 的卡片式外觀，同時在旁邊看到本工具的 AlphaZero 建議，可以啟動伴隨模式：

```bash
python tools/splendor_hint_gui.py \
  --azg-path ../alpha-zero-general \
  --checkpoint ../alpha-zero-general/splendor/pretrained_2players.pt \
  --top 8 \
  --numMCTSSims 200 \
  --official-card-hints
```

這會開啟一個整合頁面：

- 背景：官方卡片式 Splendor GUI。
- 右側浮動面板：本機 MCTS 建議走法。

如果你比較喜歡左右分割視窗，可以把 `--official-card-hints` 換成 `--official-companion`。

> 重要限制：這是「視覺伴隨模式」，不是完整資料層整合。官方 hosted 頁面無法讓本機 Python 後端可靠讀寫其內部遊戲狀態，所以官方網頁和本機建議面板**不會自動同步局面**。若要同步資料層，請改用推薦的 `--integrated-card-ui` 官方風格同步前端。

## 有建議走法的圖形介面（本機 GUI）

如果你要「圖形介面 + 建議走法」，請使用本專案的本機 browser GUI，而不是官方 hosted GUI：

```bash
python tools/splendor_hint_gui.py \
  --azg-path ../alpha-zero-general \
  --checkpoint ../alpha-zero-general/splendor/pretrained_2players.pt \
  --top 8 \
  --numMCTSSims 200
```

啟動後會開啟 `http://127.0.0.1:8766/`，頁面會顯示：

- 目前盤面。
- `建議走法` 表格：`win%` / `visits` / `prob%` / `prior%` / move。
- 每個建議旁的 `下這手` 按鈕。
- 所有合法 action 的按鈕。
- `Undo` 按鈕，回到你上一次決策前。

如果你想讓 AI 先手，加上：

```bash
--ai-first
```

如果瀏覽器沒有自動打開，手動開啟終端機顯示的網址，預設是：

```text
http://127.0.0.1:8766/
```

> 官方 <https://cestpasphoto.github.io/splendor.html> 是完整圖形化遊戲，但無法顯示本工具的 MCTS 建議表；要看建議走法請使用 `tools/splendor_hint_gui.py`。


### 為什麼提示版 GUI 長得和官方網頁不一樣？

`tools/splendor_hint_gui.py` 是本專案的「提示版 GUI」：它直接跑 Python 版 `alpha-zero-general` / MCTS，並把建議走法顯示在瀏覽器中。為了能取得 `win%`、`visits`、`prior%`，目前盤面區使用 upstream Python `printBoard` 的文字盤面，再搭配右側建議表與按鈕。

官方 <https://cestpasphoto.github.io/splendor.html> 是另一套前端，畫面是漂亮的卡片式 UI，但它沒有提供本專案 MCTS root node 的建議表。也就是：

- 要官方卡片外觀：使用 `python tools/open_splendor_gui.py --official`，但沒有本工具的建議走法表。
- 要 `win%` / `visits` / `prior%` 建議：使用 `python tools/splendor_hint_gui.py`，目前是文字盤面 + 建議表。

## 圖形化介面（GUI）對戰

如果你想使用像 <https://cestpasphoto.github.io/splendor.html> 那樣的圖形化介面，最簡單的方式是直接開啟官方 browser GUI：

```bash
python tools/open_splendor_gui.py --official
```

也可以開啟本專案的本地說明頁，頁面會嵌入官方 GUI，並保留快速說明與外部開啟按鈕：

```bash
python tools/open_splendor_gui.py
```

如果你的瀏覽器不允許直接用 `file://` 嵌入外部頁面，可以改用本地 HTTP server：

```bash
python tools/open_splendor_gui.py --serve
```

在 GUI 裡：

- 選 `You vs AI`：你先手，AI 後手。
- 選 `AI vs you`：AI 先手，你後手。
- 上方難度可從 `Come on` / `Easy` / `Medium` / `Native` / `Boosted` / `God-like` 中選擇；想挑戰最強就選 `God-like`。
- 點寶石或卡片來選動作；這是官方圖形化版的操作方式，不需要在終端機輸入 action 編號。

> 目前「高勝率走法提示」與 `undo/u` 是終端機 launcher 的功能；官方 browser GUI 可用來圖形化對戰，但不會顯示本工具從 MCTS root node 取出的 `win%` / `visits` / `prior%` 表格。

## 人機對戰時顯示高勝率走法

在本 repository 執行：

```bash
python tools/azg_human_hints.py \
  --azg-path ../alpha-zero-general \
  --top 8 \
  --numMCTSSims 800 \
  splendor \
  ../alpha-zero-general/splendor/pretrained_2players.pt \
  human-hints
```

如果你想讓人類先手：

```bash
python tools/azg_human_hints.py \
  --azg-path ../alpha-zero-general \
  --top 8 \
  --numMCTSSims 800 \
  splendor \
  human-hints \
  ../alpha-zero-general/splendor/pretrained_2players.pt
```


## Undo 功能確認

`cestpasphoto/alpha-zero-general` 原本的 console `HumanPlayer` 只接受數字走法與 `+` 顯示全部走法；這個 launcher 另外包了一層可 undo 的人類玩家與 arena。

在人類輸入提示時可以使用：

- `undo` / `u`：回到「目前人類玩家上一次決策前」；在人機對戰中，這通常會同時退回你上一手以及 AI 回應的那一手，讓你重新選擇自己的上一手。
- `undo 1` / `u1`：只退回上一個 ply（上一手），適合只想退回剛剛完成的 AI 或玩家動作。
- `--disable-undo`：停用上述 undo 指令，恢復為只接受走法輸入。

每次 undo 後會重設 MCTS 搜尋樹，避免提示沿用已被退回局面的搜尋狀態。

## 參數建議

- `--numMCTSSims 200`：比較快，提示較粗略。
- `--numMCTSSims 800`：upstream README 中 Splendor 常用的較強搜尋深度，較慢但提示較可信。
- `--top 5` / `--top 10`：控制每回合顯示幾個候選走法。
- `--sort-by q`：依估計勝率排序（預設）。
- `--sort-by visits`：依 MCTS 訪問占比排序，通常更接近 AlphaZero 實際會選的走法。

## 輸出欄位解讀

範例：

```text
AlphaZero hints for current position:
#   action  win%    visits  prob%   prior%  move
1   7       64.2%   318     39.8%   12.5%   buy ...
2   51      58.1%   203     25.4%    9.1%   take ...
```

- `win%`：由 MCTS 根節點對該 action 的 Q 值換算，適合當作「這手長期有利程度」的粗估。
- `prob%`：搜尋訪問分布；若你想模仿 AI 實戰落子，通常可優先看這欄。
- `prior%`：神經網路尚未搜尋前對該走法的直覺偏好。
