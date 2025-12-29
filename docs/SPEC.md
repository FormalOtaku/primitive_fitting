# Primitive Fitting Tool (B4 Research Prototype)

## Goal

LiDAR 点群（XYZのみ。色情報は使わない）を入力として、
ユーザが指定した「それっぽい領域」に対して、平面または円柱のプリミティブをフィットし、
その寸法（軸方向・半径・長さ、法線・位置など）を推定する半自動ツールを作る。

点群は、FAST-LIO や R3LIVE などの LIO 系 SLAM で取得済みとし、
本ツールはオフラインでのプリミティブ抽出・寸法推定のみを担当する。

## Environment

- OS: Ubuntu (Linux デスクトップ)
- Language: Python 3
- Libraries:
  - open3d
  - numpy
  - 標準ライブラリ: argparse, json, pathlib など

## Input

- コマンドライン引数 `--input` で PCD / PLY ファイルを読み込む。
- 点群には RGB が含まれていてもよいが、v0 では XYZ のみを利用する。

## Processing Pipeline

1. 前処理
   - VoxelGrid ダウンサンプリング（`--voxel_size` で指定）
   - 統計的外れ値除去（StatisticalOutlierRemoval 相当）
   - 法線推定（近傍点から）

2. ユーザによる ROI 選択（半自動）
   - Open3D のビューワを用いて点群を表示する。
   - ユーザが「プリミティブを当てたい領域」を指定できること。
     - 実装が簡単な方法でよい（例）:
       - 画面上で矩形選択し、その範囲の点群をクロップする
       - 1点クリック + 半径 r の近傍点群を ROI とする
   - ROI が決まったら、ユーザが「平面」か「円柱」を選択する。

3. プリミティブフィット
   - 平面:
     - ROI 内点群に RANSAC を適用して平面を推定。
     - パラメータ:
       - 法線ベクトル (nx, ny, nz)
       - 平面上の代表点 (px, py, pz)
       - インライヤ数
   - 円柱:
     - ROI 内点群（および法線）を用いて円柱モデルを推定。
     - 方法は RANSAC または最小二乗など、実装しやすいものでよい。
     - パラメータ:
       - 軸方向ベクトル (dx, dy, dz)
       - 軸上の一点 (ax, ay, az)
       - 半径 r
       - 長さ L（インライヤ点を軸に投影して算出）
       - インライヤ数

4. 可視化
   - Open3D のウィンドウ上で、フィットしたプリミティブをオーバーレイ表示する。
     - 平面: インライヤ点の色を変える、または半透明ポリゴンとして描画。
     - 円柱: 軸と円柱メッシュを描画。

5. 結果保存
   - 各フィット結果を JSON に追記保存する:
     - 例: `fit_results.json`
   - JSON 構造イメージ:

     ```json
     {
       "planes": [
         {
           "id": 0,
           "normal": [nx, ny, nz],
           "point": [px, py, pz],
           "inlier_count": 123
         }
       ],
       "cylinders": [
         {
           "id": 0,
           "axis_point": [ax, ay, az],
           "axis_direction": [dx, dy, dz],
           "radius": 0.05,
           "length": 1.2,
           "inlier_count": 456
         }
       ]
     }
     ```

   - 余裕があれば、プリミティブの簡単なメッシュを PLY / STL として出力する関数も用意する。

## Module Structure (Desired)

- `main.py`
  - CLI 引数処理
  - 点群読み込み / 前処理
  - ビューワ起動・ROI 選択・「平面 or 円柱」選択の管理
  - フィット結果の保存

- `primitives.py`
  - `fit_plane(points: np.ndarray) -> PlaneParam`
  - `fit_cylinder(points: np.ndarray, normals: np.ndarray | None) -> CylinderParam`
  - `PlaneParam` / `CylinderParam` は dataclass などで定義。

- `io_utils.py`（必要なら）
  - PCD/PLY の読み書き
  - JSON 保存・読み込み

## Notes

- v0 では「完璧なUI」や「完全自動」は目指さず、研究用プロトタイプとして動くことを優先する。
- 将来的に自作の円柱フィッタや、色（RGB）を使ったセグメンテーション、SLAM へのフィードバックなどに拡張しやすいように、関数・クラスのインターフェイスをシンプルに保つ。
