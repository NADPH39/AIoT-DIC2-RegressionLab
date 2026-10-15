# AIoT Regression Lab — DO In-Class 2

純前端（HTML + JavaScript + Chart.js）的 CRISP-DM 回歸實驗室，可直接部署到 GitHub Pages。

> 作業要求：do two problems but choose one of the topic to upload；第二個直接套 skill / workflow
> 參考：老師的 [50 Startups MLR Explorer](https://huanchen1107.github.io/AIoT-DIC2-MultipleLR/)

## 兩個 Problem（上方下拉選單切換）＋1 分

| Problem | 方法 | 預設資料 |
| --- | --- | --- |
| **Problem 1：台積電 2330.TW 股價預測** | AR(p) 自我迴歸、PACF、AIC 選 p、時間序切分、naive 基準比較、遞迴預測 + 95% 區間 | `data/2330_TW.csv`（臺灣證交所 STOCK_DAY，2025/12/01–2026/10/14 日收盤價），可用 FinMind API 即時抓任何台股 |
| **Problem 2：50 Startups 多元線性回歸** | 套用老師的 workflow：CRISP-DM + 6 種特徵選擇（Lasso、ElasticNet、Forward、Backward p-value、RFE、Exhaustive）、k = 1…p 的 RMSE / R² 曲線、OLS 係數表（SE、t、p、95% CI、VIF）、殘差 Top 10、即時預測 | `data/50_Startups.csv` |

## 可任意輸入 Dataset ＋2 分

資料集列可以：
- 貼上任意 CSV 網址（例：`https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv`），GitHub `blob` 連結會自動轉成 raw；
- 上傳本機 CSV；
- 點範例 chip（Boston Housing、Advertising、AirPassengers…）。

MLR 會自動判斷數值 / 類別 / ID 欄，類別欄做 one-hot（drop_first），缺值以平均數填補，目標變數可改選。
AR 會自動找日期欄與數值欄（支援民國日期、`YYYY-MM`）。

## 重現性

- `train_test_split` 在瀏覽器裡用 MT19937 重現 numpy `RandomState`，切分結果與 `sklearn.model_selection.train_test_split(test_size, random_state)` 完全相同。
- 50 Startups（test_size=0.2, random_state=0）：k=2（R&D Spend, Marketing Spend）Test RMSE **8,198.80**、R² **0.9474**；全部 5 個特徵 Test RMSE **9,137.99** → 過度擬合，與課堂結果一致。
- OLS 推論統計（SE、t、p、CI、F、VIF）以 statsmodels 相同公式計算，已用 numpy / scipy 交叉驗證。

## 檔案結構

```
index.html                 # 建置好的單一網頁（GitHub Pages 入口）
src/template.html          # 版面
src/style.css              # 樣式
src/ml.js                  # 回歸演算法（OLS、Lasso/ElasticNet、特徵選擇、AR、統計分配）
src/app.js                 # 介面邏輯
data/50_Startups.csv
data/2330_TW.csv
build.py                   # python build.py → 重新產生 index.html
.github/workflows/pages.yml
```

修改 `src/` 後執行 `python build.py` 重新產生 `index.html`。

## 部署到 GitHub Pages

1. 在 GitHub 建一個新的 public repo（例如 `AIoT-DIC2-RegressionLab`），把這個資料夾的所有檔案（含 `.github`、`.nojekyll`）推上 `main` 分支：
   ```bash
   git init && git add . && git commit -m "DIC2 regression lab"
   git branch -M main
   git remote add origin https://github.com/<你的帳號>/AIoT-DIC2-RegressionLab.git
   git push -u origin main
   ```
2. 打開 repo 的 **Settings → Pages**，在 **Build and deployment → Source** 選 **GitHub Actions**（推薦；會自動跑 `.github/workflows/pages.yml`），或選 **Deploy from a branch → main / (root) → Save**。
3. 約 1 分鐘後就能在 `https://<你的帳號>.github.io/AIoT-DIC2-RegressionLab/` 看到網頁。

## 資料來源

- 50 Startups：[Avik-Jain / 100-Days-Of-ML-Code](https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv)
- 2330.TW：臺灣證券交易所 個股日成交資訊（STOCK_DAY）；即時資料：[FinMind](https://finmindtrade.com/)
- 版面參考：老師的 50 Startups MLR Explorer、StatsUnlock Multiple Linear Regression Calculator
