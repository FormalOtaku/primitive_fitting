# Runbook テンプレート

目的: （例: 円柱プローブ実験の再現）

---

## 0. 事前確認
- 実行環境: Ubuntu / GPU / Open3D
- 使用データ: path/to/pointcloud.ply

---

## 1. 実行手順

```bash
# RUN
./scripts/record_experiment.py --date YYYY-MM-DD
```

```bash
# RUN
./venv/bin/python main.py --gui-app --gui-font NotoSansCJK
```

---

## 2. 確認項目
- [ ] 円柱が正しく表示される
- [ ] JSONに結果が保存される

---

## 3. 出力
- 実験ログ: experiments/EXPERIMENT_LOG_YYYY-MM-DD-XX.md
- 結果JSON: fit_results.json
