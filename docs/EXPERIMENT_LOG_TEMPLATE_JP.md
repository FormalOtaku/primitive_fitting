# 実験ログ テンプレート（卒論向け）

日付: YYYY-MM-DD
実験ID: exp-YYYY-MM-DD-XX
担当: （名前）

---

## 1. 目的
- 例: 円柱Probe方式の成功率を確認する

## 2. 使用バージョン
- ブランチ: `work/xxxx`
- コミット: `git rev-parse --short HEAD`
- タグ: `exp-YYYY-MM-DD`（作成したら記載）

## 3. データ
- データセット名:
- ファイルパス:
- 点数/密度の概要:

## 4. 実験条件
- モード: plane / cylinder / seed-expand / probe / stairs
- パラメータ:
  - voxel_size:
  - roi_r_min / roi_r_max / roi_r_step:
  - distance_threshold:
  - 直径/高さ入力:
  - 地面・天井制約:
  - その他:

## 5. 操作手順
- クリック回数:
- GUI操作メモ:

## 6. 結果
- 成功/失敗:
- 推定半径・長さ・法線:
- インライヤ数:
- 保存JSON:

## 7. 評価
- 真値との差:
- 成功率/失敗理由:
- 主観評価（使いやすさ）:

## 8. 考察・次の課題
- 何が効いたか
- 次に試すパラメータ

---

## 9. 付記（再現用コマンド）
```
# 例
./venv/bin/python main.py --gui-app --gui-font NotoSansCJK
```
