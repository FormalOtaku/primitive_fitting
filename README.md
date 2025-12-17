# Primitive Fitting Tool

LiDAR 点群に対して平面・円柱プリミティブをフィットする半自動ツール。

## Requirements

- Python 3.8+
- Ubuntu (Linux Desktop)
- Open3D が動作する環境（GUI が必要）

## Setup

```bash
# 仮想環境を作成（推奨）
python3 -m venv venv
source venv/bin/activate

# 依存パッケージをインストール
pip install -r requirements.txt
```

## Usage

### 基本的な使い方

```bash
python main.py --input <point_cloud.pcd>
```

### コマンドライン引数

| 引数 | 説明 | デフォルト |
|------|------|------------|
| `--input`, `-i` | 入力 PCD/PLY ファイルパス（必須） | - |
| `--voxel_size` | ダウンサンプリングのボクセルサイズ | 0.01 |
| `--output`, `-o` | 結果出力先 JSON ファイル | fit_results.json |
| `--roi_radius` | ROI 選択時の半径 | 0.2 |
| `--no_preprocess` | 前処理をスキップ | False |

### 実行例

```bash
# 基本実行
python main.py -i scan.pcd

# ボクセルサイズを調整
python main.py -i scan.pcd --voxel_size 0.02

# 前処理をスキップして実行
python main.py -i scan.ply --no_preprocess
```

## Workflow

1. 点群ファイルを読み込み、前処理（ダウンサンプリング、外れ値除去、法線推定）
2. Open3D ビューワで点群を表示
3. ROI 選択モードで、フィットしたい領域の中心点を Shift+クリック
4. プリミティブタイプ（平面 or 円柱）を選択
5. フィット結果を可視化し、JSON に保存
6. 繰り返しフィット可能

## Output

フィット結果は JSON 形式で保存されます：

```json
{
  "planes": [
    {
      "id": 0,
      "normal": [0.0, 0.0, 1.0],
      "point": [1.0, 2.0, 0.5],
      "inlier_count": 150
    }
  ],
  "cylinders": [
    {
      "id": 0,
      "axis_point": [0.0, 0.0, 0.0],
      "axis_direction": [0.0, 0.0, 1.0],
      "radius": 0.05,
      "length": 1.2,
      "inlier_count": 200
    }
  ]
}
```

## File Structure

```
primitive_fitting/
├── main.py          # CLI・メインロジック
├── primitives.py    # プリミティブフィット関数
├── requirements.txt # 依存パッケージ
├── README.md        # このファイル
└── SPEC.md          # 仕様書
```

## Stairs Mode (階段モード)

階段の段（踏面）や踊り場の平面を自動で複数枚抽出するモードです。

### 基本的な使い方

```bash
# 階段用プロファイルで実行（推奨）
python main.py -i scan.pcd --sensor-profile mid70_stairs --stairs-mode

# メッシュも出力する場合
python main.py -i scan.pcd --sensor-profile mid70_stairs --stairs-mode --export-mesh stairs_planes.ply
```

### 階段モードの動作

1. 点群を表示 → ビューワを閉じる
2. ROI選択モードで階段領域の中心を Shift+クリック
3. ROI内から複数の水平面を自動抽出
4. 結果を可視化（各平面をランダム色で表示）
5. JSON と PLY（オプション）に保存

### 階段モードオプション

| 引数 | 説明 | デフォルト |
|------|------|------------|
| `--stairs-mode` | 階段モードを有効化 | - |
| `--max-planes` | 抽出する最大平面数 | 20 |
| `--min-inliers` | 平面として認める最小点数 | 50 |
| `--stairs-ransac-n` | 平面RANSACのサンプル数 `ransac_n` | 3 |
| `--stairs-num-iterations` | 平面RANSACの反復回数 `num_iterations` | 1000 |
| `--max-tilt` | 水平面として認める最大傾斜角（度） | 15.0 |
| `--height-eps` | 平面マージの高さ許容値（m） | 0.03 |
| `--no-horizontal-filter` | 水平面フィルタを無効化 | False |
| `--no-height-merge` | 高さによる平面マージを無効化 | False |
| `--stairs-output` | 結果出力先 JSON ファイル | stairs_results.json |
| `--export-mesh` | 平面メッシュの出力先（PLY/OBJ） | - |

`distance_threshold` は `--plane-threshold`（全モード共通）で上書きできます。未指定の場合はセンサプロファイルの `plane_distance_threshold` を使用します（`mid70_stairs` のデフォルトは `0.025m`）。

### 関連するRANSACオプション（全モード共通）

| 引数 | 説明 | デフォルト |
|------|------|------------|
| `--plane-threshold` | plane RANSAC の `distance_threshold`（階段モードでも使用） | センサプロファイル値 |

### 実行例

```bash
# デフォルト設定で実行
python main.py -i ~/maps/staircase.pcd --sensor-profile mid70_stairs --stairs-mode

# 最大10枚の平面を抽出、傾斜20度まで許容
python main.py -i scan.pcd --stairs-mode --max-planes 10 --max-tilt 20.0

# 水平面フィルタを無効化して全平面を抽出
python main.py -i scan.pcd --stairs-mode --no-horizontal-filter

# メッシュを OBJ 形式で出力
python main.py -i scan.pcd --sensor-profile mid70_stairs --stairs-mode --export-mesh stairs.obj

# カスタム設定
python main.py -i scan.pcd --stairs-mode \
    --sensor-profile mid70_stairs \
    --max-planes 15 \
    --min-inliers 100 \
    --max-tilt 10.0 \
    --height-eps 0.02 \
    --stairs-output my_stairs.json \
    --export-mesh my_stairs.ply
```

### 出力形式

#### JSON (stairs_results.json)

```json
{
  "mode": "stairs",
  "plane_count": 5,
  "planes": [
    {
      "id": 0,
      "normal": [0.0, 0.0, 1.0],
      "point": [1.0, 2.0, 0.0],
      "height": 0.0,
      "inlier_count": 150
    },
    {
      "id": 1,
      "normal": [0.01, -0.02, 0.99],
      "point": [1.1, 2.1, 0.18],
      "height": 0.18,
      "inlier_count": 120
    }
  ]
}
```

#### PLY/OBJ メッシュ

`--export-mesh` を指定すると、検出した全平面のパッチメッシュを結合した 3D メッシュファイルを出力します。各平面は異なる色で着色されます。

- 各平面は「インライヤ点を平面に投影 → 2D凸包 → 三角形化（ファン）」でパッチ化します（点群形状に沿う）
- PLY は頂点カラーを保持します（OBJ は一般に頂点カラーを保持しません）

### センサプロファイル

| プロファイル | 説明 | 用途 |
|--------------|------|------|
| `default` | 一般的なデフォルト設定 | 汎用 |
| `mid70_map` | Livox Mid-70 FAST-LIO2 マップ用 | 通常のマップ |
| `mid70_dense` | Livox Mid-70 高密度スキャン用 | 近距離高密度 |
| `mid70_stairs` | Livox Mid-70 階段抽出用 | **階段モード推奨** |
| `velodyne_map` | Velodyne マップ用 | Velodyne系 |

`mid70_stairs` プロファイルは階段抽出に最適化された設定です：
- 広いROI範囲（r_max=3.0m）
- 適切な距離閾値（plane_distance_threshold=0.025）
- 十分な最小点数（min_points=200）

## Notes

- v0 は研究用プロトタイプ。`fit_plane` / `fit_cylinder` はスタブ実装
- 将来的に RANSAC の詳細実装を追加予定
