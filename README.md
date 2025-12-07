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

## Notes

- v0 は研究用プロトタイプ。`fit_plane` / `fit_cylinder` はスタブ実装
- 将来的に RANSAC の詳細実装を追加予定
