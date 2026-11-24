# AT3 技術復現｜EGE-UNet

在 ISIC2017 皮膚鏡病灶分割上，用作者的 EGE-UNet 程式從零訓練 30 個 epoch。壓縮包附實驗所用的 1,750 組縮圖和遮罩、驗證集選出的權重，以及測試集逐張結果。

## 執行

在解壓後的 `ege-unet-at3` 資料夾開啟終端，使用 Python 3.11。

```bash
python -m pip install -r requirements.txt
python run.py --device cpu --checkpoint results/short-run/best.pt --output results/cpu-eval
```

第二行使用附帶的權重，重新評估完整測試集。依賴安裝完成後可離線執行。CPU 與 CUDA 的浮點運算可能使少數接近閾值的像素產生差異。

要重跑訓練，先安裝支援 CUDA 的 PyTorch，再於同一環境執行下列指令。本次使用 PyTorch 2.5.0 / CUDA 12.1 與 RTX 4060 Laptop GPU。

```bash
python -m pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python run.py --device cuda --epochs 30 --output results/rerun
```

只確認訓練和評估流程時，可在 CPU 上各取 8 張圖，以 64 × 64 解析度跑 1 個 epoch。

```bash
python run.py --device cpu --epochs 1 --size 64 --batch-size 2 --limit-per-split 8 --output results/smoke
```

輸出路徑每次使用新的資料夾。即使資料夾內已有舊權重或結果，程式也會保留它們。

## 資料與程式

- `prepare_data.py` 從 ISIC Archive 取得 Challenge 2017 Task 1 的公開縮圖及官方遮罩。從官方 2,000 張訓練圖固定抽取 1,000 張（seed 42），保留 150 張驗證圖和 600 張獨立測試圖。資料缺失時可執行 `python prepare_data.py` 重新準備，預設下載設定對應本實驗。
- `data/isic2017/manifest.csv` 列出每張影像的 ISIC ID、分割集合和來源網址。使用最近鄰插值將標註縮成 256 × 256 的二值遮罩。使用雙線性插值調整影像尺寸並除以 255，訓練時對影像及標註同步做翻轉及 90° 倍數旋轉。
- `vendor/egeunet.py` 沿用作者的 GHPA、GAB 及五個中間遮罩輸出。相對上游只調整匯入，原授權附於 `vendor/LICENSE`。 [EGE-UNet 官方程式碼](https://github.com/JCruan519/EGE-UNet/tree/f52ba30c6bf7d0ca479c2c9d4a3cbda999f49d3a)
- `run.py` 以 BCE + Dice 損失訓練，保留原模型的五層深度監督權重。每個 epoch 評估驗證集，取前景 Dice 最高的權重，閾值固定為 0.5。測試時先合計全部像素的 TP、FP、FN，再求 Dice 和 IoU。 [作者評估程式](https://github.com/JCruan519/EGE-UNet/blob/f52ba30c6bf7d0ca479c2c9d4a3cbda999f49d3a/engine.py)

## 結果檔案

| 檔案 | 內容 |
|---|---|
| `results/short-run/metrics.json` | 測試指標、混淆矩陣計數、訓練設定、環境和資料清單雜湊 |
| `results/short-run/history.csv` | 逐輪訓練損失、驗證 Dice／IoU、學習率及耗時 |
| `results/short-run/test_per_image.csv` | 600 張測試影像的逐張指標 |
| `results/short-run/best.pt` | 本次訓練所得權重 |
| `results/short-run/*.png` | 訓練曲線、測試結果與六組預測 |

資料使用與引用見 [ISIC2017 官方資料頁](https://challenge.isic-archive.com/data/#2017) 和 [ISIC2017 挑戰論文](https://arxiv.org/abs/1710.05006)。原方法與 300 輪實驗見 [EGE-UNet 論文](https://arxiv.org/html/2307.08473v1)。
