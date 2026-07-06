# ROI Heatmap Ablation 中文報告整理

## 1. 這個測試在做什麼

這個實驗的目的，是測試「RF heatmap 資訊」對於未來 ROI 預測有沒有幫助。

我們比較兩個模型：

- `6dof`：只使用頭戴裝置的姿態歷史，也就是位置和旋轉資訊。
- `heatmap_6dof`：使用同樣的姿態歷史，再加上 RF heatmap 萃取出的特徵。

這裡的任務不是直接預測一個精確座標，而是把空間切成很多小區塊，模型要預測「未來某個時間點，使用者會落在哪一個 ROI 區塊」。

## 2. ROI 切成幾塊

本次實驗使用 3D ROI grid，三個軸各切 `8` 等分：

`8 x 8 x 8 = 512` 個 ROI 區塊。

也就是說，模型每次要在 `512` 個候選區塊中，預測未來位置會在哪一格。

本次 ROI 的座標範圍是：

- 最小座標：x=-80.50, y=-115.06, z=71.87
- 最大座標：x=91.29, y=182.81, z=186.22

每一格大約大小為：

- X 方向：21.47
- Y 方向：37.23
- Z 方向：14.29

## 3. 用到了多少 dataset

本次 split 總共有 825 個 session，其中 train 577 個、validation 123 個、test 125 個。

這裡是以 session 為單位切 train/validation/test，不是把同一個 session 裡的 frame 隨機混在 train 和 test。這樣比較能避免資料洩漏，測出來的 test 結果也比較可信。

不同 horizon 的 sample 數會略有不同，因為每一筆資料都需要能對齊到未來 50、100、150、200 ms 的 target。Heatmap 版本 sample 較少，是因為它除了 pose 之外，還必須要有可對齊的 heatmap。

| Horizon | 6DoF train/test samples | Heatmap+6DoF train/test samples |
|---:|---:|---:|
| 50 ms | 82213 / 17715 | 72534 / 15688 |
| 100 ms | 81638 / 17591 | 72497 / 15679 |
| 150 ms | 80489 / 17343 | 72423 / 15659 |
| 200 ms | 79913 / 17219 | 72386 / 15649 |

## 4. 模型輸入用了什麼

`6dof` 使用最近 250 ms 的 pose history。程式中每筆 sample 會保留 8 個 pose history points，並轉成相對時間、相對位置、相對旋轉、速度、角速度等特徵。

`heatmap_6dof` 在上述 pose 特徵之外，額外加入 RF heatmap compact features，包含 heatmap 的 top peaks、heatmap 重心、基本統計量，以及 pooled 之後的 8 x 8 heatmap 表示。

## 5. 預測 horizon 是什麼

本次測試四種未來時間：

- 50 ms：預測 50 ms 後的 ROI。
- 100 ms：預測 100 ms 後的 ROI。
- 150 ms：預測 150 ms 後的 ROI。
- 200 ms：預測 200 ms 後的 ROI。

一般來說，horizon 越長越難，因為越遠的未來越不確定。

## 6. Training 之後的數值怎麼看

- `train samples`：訓練時實際用到的 aligned samples 數。
- `test samples`：測試時實際評估的 aligned samples 數。
- `training loss`：訓練過程中的錯誤目標值，越低通常代表模型越能 fit training data。本次兩個模型在 8 個 epochs 中 loss 都有下降，表示訓練正常收斂。
- `Top-1 accuracy`：模型第一名預測就是正確 ROI 的比例，越高越好。
- `Top-3 / Top-5 / Top-10 accuracy`：正確 ROI 有沒有出現在前 3、前 5、前 10 個候選裡，越高越好。這對「可以預先準備多個候選 ROI」的系統很重要。
- `Mean ROI distance error`：模型第一名預測的 ROI 中心，和正確 ROI 中心的平均距離，越低越好。
- `Median ROI distance error`：距離誤差的中位數，越低越好，比 mean 更不容易被少數極端錯誤影響。
- `Mean latency`：單次預測平均花費時間，越低越好。
- `P95 latency`：95% 的預測都會比這個時間更快，用來看比較壞情況下的延遲。
- `Peak memory`：評估時的峰值記憶體使用量，越低越好。

## 7. 主要結果

Heatmap 加進來後，Top-1 沒有變好，平均下降 1.06 個百分點。也就是說，如果只看「第一名是否完全命中」，heatmap 版本不是比較好的。

但是 heatmap 對候選集合的幫助非常明顯：

- Top-3 平均提升 9.90 個百分點。
- Top-5 平均提升 12.73 個百分點。
- Top-10 平均提升 15.23 個百分點。

Mean ROI distance error 平均改善 3.89。這代表即使第一名沒有完全猜中，heatmap 版本的第一名預測通常也更接近正確位置。

資源成本方面，`6dof` 的 peak memory 約 1.2 MB，`heatmap_6dof` 約 42.7 MB。延遲方面兩者都很低，約 0.12 到 0.15 ms；heatmap 版本稍慢，但差距很小。

## 8. 報告可以怎麼下結論

可以這樣寫：

> 本實驗將 3D 空間切成 512 個 ROI 區塊，比較只使用 6DoF 姿態歷史，以及加入 RF heatmap 特徵後的預測效果。結果顯示，RF heatmap 沒有提升 Top-1 exact hit rate，但顯著提升 Top-3、Top-5、Top-10 candidate accuracy，且降低平均 ROI 距離誤差。這代表 heatmap 資訊對於「多候選 ROI 預取或排序」特別有價值；若系統能同時處理前幾個候選區域，heatmap 版本能提供更可靠的候選集合。

## 9. 圖表檔案

- ROI 切分示意圖：`figures/roi_grid.png`
- Dataset split：`figures/dataset_split.png`
- Top-k accuracy：`figures/top1_accuracy.png`, `figures/top3_accuracy.png`, `figures/top5_accuracy.png`, `figures/top10_accuracy.png`
- Heatmap 帶來的 accuracy improvement：`figures/accuracy_delta.png`
- ROI distance error：`figures/roi_error.png`
- Latency：`figures/latency.png`
- Memory：`figures/memory.png`
- Training loss：`figures/training_loss.png`

## 10. 完整結果表

| Horizon | Model | Train samples | Test samples | Top-1 | Top-3 | Top-5 | Top-10 | Mean ROI error | Mean latency | Peak memory |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 ms | 6DoF only | 82213 | 17715 | 14.36% | 31.11% | 39.33% | 57.01% | 46.91 | 0.124 ms | 1.25 MB |
| 50 ms | Heatmap + 6DoF | 72534 | 15688 | 12.92% | 40.69% | 52.28% | 72.32% | 43.13 | 0.144 ms | 42.75 MB |
| 100 ms | 6DoF only | 81638 | 17591 | 14.42% | 31.24% | 39.69% | 57.29% | 46.80 | 0.123 ms | 1.21 MB |
| 100 ms | Heatmap + 6DoF | 72497 | 15679 | 13.48% | 41.04% | 52.45% | 72.59% | 42.47 | 0.154 ms | 42.73 MB |
| 150 ms | 6DoF only | 80489 | 17343 | 14.35% | 31.56% | 39.84% | 57.31% | 46.56 | 0.127 ms | 1.20 MB |
| 150 ms | Heatmap + 6DoF | 72423 | 15659 | 13.29% | 41.32% | 52.60% | 72.39% | 42.73 | 0.148 ms | 42.73 MB |
| 200 ms | 6DoF only | 79913 | 17219 | 14.23% | 31.55% | 40.82% | 57.72% | 46.60 | 0.128 ms | 1.17 MB |
| 200 ms | Heatmap + 6DoF | 72386 | 15649 | 13.43% | 42.01% | 53.27% | 72.96% | 42.96 | 0.148 ms | 42.75 MB |
