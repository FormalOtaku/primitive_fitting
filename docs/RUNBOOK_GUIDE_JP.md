# Runbook運用ガイド（MDに従って実行）

最終更新: 2025-12-25

目的: 「〇〇MDに従って実行して」と言えば、
**MDの手順をそのまま実行できる**ようにするためのルール。

---

## 1. Runbookの書き方

- 実行したいコマンドは **bash のコードブロック**に書く
- 実際に実行させたいブロックは、先頭に **`# RUN`** を置く

例:
```bash
# RUN
./venv/bin/python main.py --gui-app --gui-font NotoSansCJK
```

---

## 2. 実行方法

```bash
./scripts/runbook_runner.py RUNBOOK_XXX.md
```

- 既定では `# RUN` が付いたブロックだけを実行
- `--all` を付けると、bash/shブロックをすべて実行
- `--dry-run` で実行せずに表示のみ

---

## 3. すべて自動（ログ作成→実行→コミット→タグ→push）

```bash
./scripts/runbook_runner.py RUNBOOK_XXX.md \
  --log --commit --tag --push
```

- `--log` : 実験ログを `experiments/` に自動生成
- `--commit` : 変更をコミット
- `--tag` : 実験IDでタグ付け
- `--push` : ブランチとタグをpush

※ `--commit` のみだと **ログだけコミット**されます。結果JSONも含めたい場合は `--commit-all` を使ってください。

---

## 4. ルール

- destructiveな操作（削除・上書き）は必ず **コメントで明示**
- 実験ログや出力ファイルのパスは **絶対パスまたは相対パス**で明確に

---

## 5. 例: 実験Runbook

```markdown
# Cylinder Probe 実験

## 実行
```bash
# RUN
./scripts/runbook_runner.py RUNBOOK_XXX.md --log --commit --tag
```

```bash
# RUN
./venv/bin/python main.py --gui-app --gui-font NotoSansCJK
```
```

---

必要なら、runbookテンプレや運用ルールを追加できます。
