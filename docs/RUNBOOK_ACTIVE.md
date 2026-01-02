# RUNBOOK_ACTIVE（最小テンプレ）

目的: （例: 円柱抽出の再現実験）

---

## 0. 事前確認（手動）
- [ ] 使用データのパスを確認
- [ ] GUI起動が必要か確認

---

## 1. 実行（手動）

- GUIを使う場合は、自分で起動してクリック操作を行う

例:
```
./venv/bin/python main.py --gui-app --gui-font NotoSansCJK
```

---

## 2. 後処理（ここだけ自動実行）

```bash
# RUN
# 実験ログ作成 + 結果をまとめてコミット + タグ + push
./scripts/record_experiment.py --commit-all --tag --push
```

---

## 3. 出力
- 実験ログ: experiments/EXPERIMENT_LOG_YYYY-MM-DD-XX.md
- 結果JSON: fit_results.json / seed_expand_results.json など

---

## 4. 注意
- 破壊的操作がある場合はコメントで明示すること
