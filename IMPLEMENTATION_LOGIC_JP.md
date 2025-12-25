# プリミティブフィッティング 現行実装・ロジック詳細（詳説版）

最終更新: 2025-12-25  /  対象: `/home/murasaki01/primitive_fitting`

本書は **現行コード（main.py / primitives.py / gui_app.py）を読み取って整理**した実装解説です。
卒論での記述を想定し、
- 「どの処理がどのファイルにあるか」
- 「なぜそういう設計か」
- 「パラメータが何に効くか」
を分かりやすくまとめています。

参考:
- 仕様概要: `SPEC_OVERVIEW_JP.md`
- 操作方法: `README.md`

---

## 1. 目的と設計方針

### 1.1 目的
- 点群から **平面・円柱・階段（水平面群）**を抽出し、
  形状パラメータ（法線・半径・長さなど）を推定する。

### 1.2 設計方針
- **完全自動ではなく半自動**（ユーザがseed/ROIを指示）
- その後は **RANSAC + 連結成分抽出 + 反復リファイン**で自動推定
- 点群密度の差に強く、研究用の検証がしやすい構成を重視

---

## 2. コード構成（対応表）

| 役割 | ファイル | 主な関数/クラス |
|---|---|---|
| CLI / 実行フロー | `main.py` | `main()`, `run_*_mode` |
| 形状推定コア | `primitives.py` | `fit_plane`, `fit_cylinder`, `expand_*`, `probe_*` |
| GUI | `gui_app.py` | `PrimitiveFittingApp` |
| テスト | `tests/` | seed-expand / probe / stairs 検証 |

---

## 3. 点群の前処理（main.py / gui_app.py）

### 3.1 読み込み
- `load_point_cloud(filepath)`
  - `o3d.io.read_point_cloud` で読み込み
  - 空点群は例外

### 3.2 前処理 `preprocess_point_cloud`
1. **Voxel downsample**（`voxel_size`）
2. **統計的外れ値除去**（`nb_neighbors`, `std_ratio`）
3. **法線推定**（`normal_radius`）

#### 設計理由
- RANSACを安定させるためにノイズを減らす
- 円柱推定では法線が重要 → 法線推定を標準化

---

## 4. ROI / seed 選択（main.py）

### 4.1 ROISelector（適応半径）
- `select_roi_adaptive()`
  - `r_min → r_max` まで `r_step` で半径拡張
  - `min_points` を満たしたら確定

### 4.2 固定半径
- `--no-adaptive-roi` で固定半径を使用

#### 設計理由
- 点群密度の差を吸収し、最低限の点数を確保
- 完全自動より「狙いを指定させる」方が安定

---

## 5. 平面推定（primitives.py）

### 5.1 `fit_plane`（RANSAC）
- `segment_plane` の結果から
  - 法線を正規化
  - `nz >= 0` に揃える
  - 重心を平面に射影して代表点とする

### 5.2 平面パッチの生成
- GUI/CLI可視化用のメッシュ生成
- `patch_shape` により **凸包 / 矩形**を選択

#### 設計理由
- RANSACは外れ値に強い
- 代表点を平面上に拘束することで後段の処理が安定

---

## 6. 円柱推定（primitives.py）

### 6.1 `fit_cylinder` の流れ
1. ランダムサンプルから **SVDで軸方向推定**
2. 軸直交距離の中央値を半径候補
3. 閾値内の点をインライヤ
4. 法線があれば、半径方向と法線の整合性を条件に追加
5. 最良インライヤ集合を選択
6. 最終的な軸・半径・長さを再推定

### 6.2 軸の安定化
- `_axis_from_normals` で法線分布から軸方向を推定し再評価

### 6.3 長さ推定
- 軸方向投影の **外れ値を除去**して長さ算出

---

## 7. Seed‑Expand（平面）

### 7.1 `expand_plane_from_seed`
1. seed半径内の点 → 初期平面推定
2. max半径内で候補点抽出
3. 平面距離・法線一致で候補絞り込み
4. 連結成分抽出（seedに連結する成分のみ）
5. 反復リファイン
   - SVDで法線再推定
   - 自動しきい値調整（中央値+MAD）

### 7.2 面積・残差計算
- 面積/extentを算出
- residual median / p90 / p95 を保存

---

## 8. Seed‑Expand（円柱）

### 8.1 `expand_cylinder_from_seed`
1. seed内で初期円柱推定
2. max半径内で円柱表面候補抽出
3. 連結成分抽出で1本に限定
4. 再推定（軸/半径/長さ）

---

## 9. Cylinder Probe（円柱専用強化版）

### 9.1 `compute_cylinder_proxy_from_seed`
- seed半径を複数試行
- score = inlier数 / 残差 で最良候補を選択

### 9.2 `finalize_cylinder_from_proxy`
- 表面成分抽出
- 軸再推定（スナップ・正則化オプション）
- 長さ再推定

#### 設計理由
- 円柱はROIに他形状が混在しやすい
- 代理円柱を使うことで **誤推定を抑制**

---

## 10. Stairs Mode（階段抽出）

1. RANSACで平面を反復抽出
2. 水平面のみ残す
3. 高さで近い平面をマージ

設計理由:
- 階段は「高さの違う水平面群」として扱える

---

## 11. GUI実装（gui_app.py）

### 11.1 主要UI
- 右パネル：全て折りたたみ式（初期は閉じ）
- センサプロファイル / ROI / パラメータ / 出力 / 編集 / アウトライナー

### 11.2 操作
- **Shift+クリック**でseed指定
- 円柱は **`probe_cylinder_from_seed`** を使用

### 11.3 編集モード
- Shift+クリックで半径内削除
- 矩形削除は **2クリック方式**
  - Open3Dのドラッグ操作と競合するため
- マスク方式で削除（元点群は保持）

### 11.4 アウトライナー
- `[S]/[W]/[H]` 表示状態
- 表示/非表示, ワイヤ/ソリッド切替
- 色の手動変更

### 11.5 地面/天井制約
- 地面：RANSAC + 傾斜角制限
- 天井：Z上位点群からRANSAC
- 円柱の垂直制約：地面法線との角度判定

---

## 12. 出力形式（JSON）

- `fit_results.json`（平面・円柱）
- `seed_expand_results.json`
- `cyl_probe_results.json`
- `stairs_results.json`

いずれも **追記保存**で複数結果を保持。

---

## 13. 既知の制約

- Open3D SceneWidget の仕様により
  **ドラッグ操作を完全に上書きできない**
- そのため矩形削除は2クリック方式のみ

---

## 14. どのロジックがどこにあるか（早見表）

- 平面RANSAC: `primitives.py::fit_plane`
- 円柱RANSAC: `primitives.py::fit_cylinder`
- Seed‑expand平面: `primitives.py::expand_plane_from_seed`
- Seed‑expand円柱: `primitives.py::expand_cylinder_from_seed`
- Cylinder probe: `primitives.py::probe_cylinder_from_seed`
- Stairs抽出: `primitives.py::extract_multi_planes`
- CLI制御: `main.py`
- GUI制御: `gui_app.py`

---

必要なら、このドキュメントを卒論の「手法」章構成に沿って
**章立て（背景→関連研究→提案手法→実験→考察）**に合わせて整理します。
