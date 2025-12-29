# Primitive Fitting Tool - スライド用まとめ（研究室発表）

## 1. 目的 / 位置付け
- LiDAR 点群（XYZ）から **平面・円柱プリミティブ**を半自動で抽出し、寸法（法線・軸・半径・長さなど）を推定する。
- FAST-LIO / R3LIVE などで取得済みの点群を **オフラインで計測・寸法化**する用途。
- v0 研究用プロトタイプとして、UI 完全自動化より **再現可能な簡易ワークフロー**を優先。

## 2. 現在の構成（リポジトリ）
- `main.py`
  - CLI 引数処理、点群 I/O、前処理
  - ROI 選択（GUI）、プリミティブ選択
  - 可視化、結果 JSON 保存
  - 5つの動作モード（通常 / 階段 / seed-expand / 円柱プローブ / セッション）
- `primitives.py`
  - 平面/円柱フィット
  - 階段向け多平面抽出
  - seed-expand（領域拡張）アルゴリズム
  - 円柱プローブの最終化（軸補正/長さ再推定）
- `requirements.txt`
  - `open3d`, `numpy`
- `CYLINDER_PROBE_IMPROVEMENTS.md`
  - 円柱プローブ改善メモ（軸安定化/長さ推定/診断出力）

## 3. 実行環境 / 依存
- OS: Ubuntu (Linux Desktop)
- 言語: Python 3
- 主要ライブラリ: Open3D, NumPy
- GUI 要求: Open3D ビューワ、Tkinter（ファイル選択ダイアログ）

## 4. 全体フロー（通常モード）
1. 点群読み込み（PCD/PLY）
2. 前処理（ダウンサンプリング / 外れ値除去 / 法線推定）
3. 点群表示 → ROI をクリック選択
4. 平面 or 円柱 を選択
5. フィット結果可視化
6. JSON へ保存（追記）

```
Input PCD/PLY
   ↓
Preprocess (voxel / outlier / normals)
   ↓
ROI selection (GUI)
   ↓
Primitive fit (plane / cylinder)
   ↓
Visualization + JSON output
```

※ 階段/seed-expand/円柱プローブ/セッションは、同じ前処理 + クリック操作をベースに、抽出アルゴリズムと出力が分岐する。

## 5. アルゴリズム概要（理論）
### 5.1 平面フィット
- Open3D の RANSAC (`segment_plane`) を使用
- 出力: 法線ベクトル、平面上の点、インライヤ数
- 法線は +Z 方向優先で符号統一

### 5.2 円柱フィット
- ランダムサンプル + SVD で軸方向を推定（RANSAC）
- 半径は軸からの距離の中央値で推定
- 距離閾値によるインライヤ判定
- 法線がある場合は **法線と半径方向の整合**も条件にする
- 長さは軸方向への射影範囲（外れ値を除いた量子点）で推定

### 5.3 階段モード（多平面抽出）
- 反復 RANSAC で複数平面抽出
- **水平面フィルタ**（法線の Z 成分による傾き制限）
- **高さマージ**（近い高さの平面を統合）

### 5.4 Seed-Expand（種→拡張）
- ユーザが seed 点をクリック → seed 半径内で初期フィット
- seed から候補点を抽出し、**連結成分/BFS で領域成長**
- 1〜2 回の再フィットでモデルを更新
- 拡張後の面積/長さ/残差の統計を出力

### 5.5 円柱プローブ（インタラクティブ）
- 1 クリックで proxy 円柱を生成し、キー操作で半径/長さ/位置/軸を微調整
- 確定後、proxy を基に再抽出 → 最終円柱を再推定
- 軸のスナップ/正則化・長さ再計算などをオプション化

### 5.6 セッションモード（ワークスペース型）
- seed を連続クリック → plane/cylinder を自動判定して一発抽出
- 抽出物をシーンに保持し、まとめて JSON/メッシュ出力
- Undo/削除/保存を対話的に操作

## 6. ROI 選択（GUI）
- **適応半径モード**（標準）
  - r_min から開始し、min_points を満たすまで半径を拡大
- **固定半径モード**（legacy）
- シフト＋クリックで中心点指定

## 7. センサプロファイル
用途に応じたプリセットで前処理・閾値・ROI を調整。
- `default` / `mid70_map` / `mid70_dense` / `velodyne_map` / `mid70_stairs`
- 例: `mid70_stairs` は広い ROI と小さめ閾値で階段抽出向け

## 8. 出力形式
- **通常モード**: `fit_results.json`
  - planes: normal, point, inlier_count
  - cylinders: axis_point, axis_direction, radius, length, inlier_count
- **階段モード**: `stairs_results.json`
  - plane_count, height, inlier_count
- **seed-expand**: `seed_expand_results.json`
  - 成功/失敗、拡張点数、面積、パッチサイズ等
- **円柱プローブ**: `cyl_probe_results.json`
  - proxy/final の円柱パラメータ、残差統計、操作回数など
- **セッション**: `session.json`
  - objects[]（seed/params/quality/stop_reason）
- オプション: 平面/円柱の PLY/OBJ 出力、拡張点群/診断 PLY の出力

## 9. 既知の制約 / 想定
- 点群の RGB は使用しない（XYZ のみ）
- GUI 必須で完全自動ではない（Open3D + Tk）
- RANSAC ベースのため外れ値・密度に敏感
- モード併用不可（`session` は他モードと併用不可、`seed-expand` と `cyl-probe` は排他）
- v0 プロトタイプのため精度/速度はチューニング余地あり

## 10. 今後の拡張案
- 円柱フィットの精度向上（最小二乗/専用 RANSAC）
- RGB/強度を用いたセグメンテーション
- ROI 自動化（クラスタリング・検出補助）
- セッションの自動判定/品質指標の強化
- SLAM 系へのフィードバック（マップ更新・構造抽出）
