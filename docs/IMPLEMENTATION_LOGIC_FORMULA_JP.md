# 数式入り（論文寄り）実装ロジック（卒論用）

最終更新: 2025-12-25

本書は現行実装を **数式視点で整理**したものです。式は概念整理であり、
厳密な証明や最適化の完全記述ではありません。

---

## 1. 記法

- 点群: \( \mathbf{P} = \{\mathbf{p}_i\}_{i=1}^N, \; \mathbf{p}_i \in \mathbb{R}^3 \)
- 法線: \( \mathbf{n}_i \)（\(\|\mathbf{n}_i\|=1\)）
- 平面: \(\Pi=(\mathbf{n},\mathbf{p}_0)\)
- 円柱: 軸点 \(\mathbf{a}\), 方向 \(\mathbf{d}\)（\(\|\mathbf{d}\|=1\)）, 半径 \(r\), 長さ \(L\)

---

## 2. 前処理

### 2.1 統計的外れ値除去
各点の近傍距離 \(\mu_i\) を計算し、
\[
\mu_i > \mu + \alpha\sigma
\]
を満たす点を除去。

### 2.2 法線推定
局所近傍のPCAにより最小固有値方向を法線とする。

---

## 3. 平面推定（RANSAC）

距離:
\[
\text{dist}(\mathbf{p}_i, \Pi) = | \mathbf{n}^\top (\mathbf{p}_i - \mathbf{p}_0) |
\]

インライヤ集合:
\[
\mathcal{I} = \{ i \mid \text{dist}(\mathbf{p}_i, \Pi) < \tau \}
\]

法線は \(n_z \ge 0\) に揃える。

代表点は重心 \(\bar{\mathbf{p}}\) を平面に射影:
\[
\mathbf{p}_0 = \bar{\mathbf{p}} - (\mathbf{n}^\top \bar{\mathbf{p}} + d)\,\mathbf{n}
\]

---

## 4. 円柱推定

### 4.1 軸方向（SVD）
サンプル点群 \(\mathbf{X}\) を中心化し、
\[
\mathbf{X} = \mathbf{U}\Sigma\mathbf{V}^\top
\]
最大分散方向を軸 \(\mathbf{d}\) とする。

### 4.2 半径
\[
\mathbf{v}_i = \mathbf{p}_i - \mathbf{a} - ((\mathbf{p}_i-\mathbf{a})^\top \mathbf{d})\mathbf{d}
\]
\[
\rho_i = \|\mathbf{v}_i\|, \quad r = \text{median}(\rho_i)
\]

### 4.3 インライヤ条件
\[
|\rho_i - r| < \tau
\]
法線利用時:
\[
\frac{\mathbf{v}_i}{\|\mathbf{v}_i\|} \cdot \mathbf{n}_i > \cos(\theta)
\]

### 4.4 長さ推定
軸方向投影:
\[
\ell_i = (\mathbf{p}_i - \mathbf{a})^\top \mathbf{d}
\]
外れ値を除去した範囲:
\[
L \approx Q_{1-q}(\ell) - Q_q(\ell)
\]

---

## 5. Seed‑Expand 平面

seed点集合:
\[
\mathbf{P}_s = \{ \mathbf{p}_i \mid \|\mathbf{p}_i-\mathbf{p}_{seed}\| < R_s \}
\]

候補集合:
\[
\mathbf{P}_c = \{\mathbf{p}_i \mid \|\mathbf{p}_i-\mathbf{p}_{seed}\| < R_{max},\; |\mathbf{n}^\top(\mathbf{p}_i-\mathbf{p}_0)| < \tau \}
\]

連結成分抽出:
- 近傍半径 \(r_g\) でグラフを作成
- seedに連結する成分のみ残す

反復リファイン（最小二乗）:
\[
\mathbf{n} = \arg\min_{\|\mathbf{n}\|=1} \sum_{i\in \mathcal{I}} (\mathbf{n}^\top(\mathbf{p}_i-\bar{\mathbf{p}}))^2
\]

閾値の自動調整（MAD）:
\[
\tau_{new} = \text{median}(d_i) + k\cdot 1.4826\,\text{MAD}(d_i)
\]

---

## 6. Seed‑Expand 円柱

候補点集合:
\[
\mathbf{P}_c = \{\mathbf{p}_i \mid |\rho_i - r| < \tau\}
\]

再推定:
\[
(\mathbf{a}, \mathbf{d}, r) = \arg\min \sum_{i\in \mathcal{I}} (\|\mathbf{v}_i\|-r)^2
\]

---

## 7. Cylinder Probe

代理円柱の評価:
\[
\text{score} = \frac{|\mathcal{I}|}{\max(\text{median residual}, \epsilon)}
\]

最終化:
- 代理円柱に基づく表面候補抽出
- 連結成分抽出
- 軸/半径/長さの再推定

---

## 8. Stairs Mode

水平判定:
\[
\arccos(n_z) \le \theta_{max}
\]

高さマージ:
\[
|h_i - h_j| < \epsilon_h
\]

---

## 9. 評価指標

- 残差中央値: \(\text{median}(|d_i|)\)
- インライヤ率: \(|\mathcal{I}|/N\)
- 半径/長さ誤差: 既知寸法との差
- 軸傾き誤差: \(\arccos(\mathbf{d} \cdot \mathbf{d}_{gt})\)

---

必要なら、論文形式（関連研究・実験設計・考察）に拡張可能です。
