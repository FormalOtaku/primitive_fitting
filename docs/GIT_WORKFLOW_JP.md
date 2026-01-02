# Git運用メモ（研究向け）

最終更新: 2025-12-25

目的: 研究開発で安全に作業し、卒論で「再現性」と「履歴管理」を確保するためのGit運用ルール。

---

## 1. 基本方針

- **mainは完成版**（常に動く状態）
- 作業は必ず **作業ブランチ** で行う
- 確認できたら **mainにマージ**

---

## 2. いつも使う流れ（最小構成）

```bash
# 1) mainを最新化
git checkout main
git pull origin main

# 2) 作業ブランチを作る
git checkout -b work/feature-name

# 3) 作業・コミット
git add .
git commit -m "やったこと"

# 4) 作業ブランチをpush
git push -u origin work/feature-name

# 5) mainにマージ
git checkout main
git pull origin main
git merge --no-ff work/feature-name

# 6) mainをpush
git push origin main
```

---

## 3. 大きめの変更（安全ルート）

- 直接 main に入れず、**統合ブランチ**を挟む

```bash
git checkout -b integration

git merge --no-ff work/feature-name
# テストや確認

git checkout main
git merge --no-ff integration
git push origin main
```

---

## 4. PR運用（GitHub）

**目的**: 差分の見える化とレビューの簡易化

基本フロー:
```bash
# 作業ブランチをpush
# GitHub上でPR作成
# レビュー・確認後にmerge
```

最低限のチェック:
- 変更点が期待通りか（差分）
- テストが通るか
- UI変更はスクリーンショットで確認

---

## 5. release / タグ運用

卒論では「実験時点のバージョン」を固定すると便利。

例:
```bash
# タグ作成（実験1）
git tag exp-2025-12-25
# タグをpush
git push origin exp-2025-12-25
```

推奨タグ名:
- `exp-YYYY-MM-DD`
- `v0.1`, `v0.2` など

---

## 6. 実験ログの自動作成・コミット

### ログ作成のみ
```bash
./scripts/create_experiment_log.py
```

### ログ作成 + コミット + タグ
```bash
./scripts/record_experiment.py
```

### ログ作成 + コミット + タグ + push
```bash
./scripts/record_experiment.py --push
```

---

## 7. Runbook（MDに従って実行）

```bash
./scripts/runbook_runner.py RUNBOOK_XXX.md
```

### すべて自動
```bash
./scripts/runbook_runner.py RUNBOOK_XXX.md --log --commit --tag --push
```

---

## 8. コンフリクトが出たとき

```bash
# rebase中なら
# 修正 → add → continue

git add <file>
git rebase --continue

# 途中で諦めるなら
git rebase --abort
```

---

## 9. よくある失敗を防ぐ

- **mainで作業しない**
- **push前に pull** する
- **大きな変更はPRでレビュー**

---

## 10. 卒論向けのポイント

- 履歴を残すことで「研究過程の再現性」を示せる
- 実験ごとにブランチを分けると整理しやすい
- コミットメッセージに「目的/結果」を書くと後で助かる

---

必要なら、テンプレート（実験ログやPRの本文）も作れます。
