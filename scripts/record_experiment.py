#!/usr/bin/env python3
"""Create an experiment log and optionally commit/tag/push it.

Usage:
  ./scripts/record_experiment.py
  ./scripts/record_experiment.py --date 2025-12-25 --push
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Optional, Tuple


def _run_git(args: list[str]) -> str:
    try:
        out = subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def _next_suffix(out_dir: Path, day: str) -> str:
    suffixes = []
    for path in out_dir.glob(f"EXPERIMENT_LOG_{day}-*.md"):
        stem = path.stem
        tail = stem.split("-")[-1]
        if tail.isdigit():
            suffixes.append(int(tail))
    next_num = max(suffixes, default=0) + 1
    return f"{next_num:02d}"


def _fill_template(text: str, day: str, exp_id: str, branch: str, commit: str) -> str:
    text = text.replace("YYYY-MM-DD", day)
    text = text.replace("exp-YYYY-MM-DD-XX", exp_id)

    def repl_line(prefix: str, value: str, content: str) -> str:
        lines = []
        for line in content.splitlines():
            if line.strip().startswith(prefix):
                lines.append(f"{prefix} `{value}`")
            else:
                lines.append(line)
        return "\n".join(lines)

    text = repl_line("- ブランチ:", branch, text)
    text = repl_line("- コミット:", commit, text)
    return text + ("\n" if not text.endswith("\n") else "")


def create_log(day: str, out_dir: Path, template_path: Path) -> Tuple[Path, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = _next_suffix(out_dir, day)
    exp_id = f"exp-{day}-{suffix}"
    out_path = out_dir / f"EXPERIMENT_LOG_{day}-{suffix}.md"

    template_text = template_path.read_text(encoding="utf-8")
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    commit = _run_git(["rev-parse", "--short", "HEAD"])
    filled = _fill_template(template_text, day, exp_id, branch, commit)
    out_path.write_text(filled, encoding="utf-8")
    return out_path, exp_id


def _git_tag_exists(tag: str) -> bool:
    try:
        subprocess.check_call(["git", "rev-parse", tag], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def _run(args: list[str]) -> None:
    subprocess.check_call(args)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Create experiment log and optionally commit/tag/push")
    parser.add_argument("--date", help="YYYY-MM-DD (default: today)")
    parser.add_argument("--outdir", default=str(root / "experiments"))
    parser.add_argument("--template", default=str(root / "docs" / "EXPERIMENT_LOG_TEMPLATE_JP.md"))
    parser.add_argument("--no-commit", action="store_true")
    parser.add_argument("--no-tag", action="store_true")
    parser.add_argument("--force-tag", action="store_true")
    parser.add_argument("--message", help="commit message (default: Add experiment log <id>)")
    parser.add_argument("--tag", help="tag name (default: exp-YYYY-MM-DD-XX)")
    parser.add_argument("--push", action="store_true", help="push branch and tag")
    parser.add_argument("--commit-all", action="store_true", help="git add -A before commit")
    parser.add_argument("--commit-path", action="append", default=[], help="extra path to git add")
    args = parser.parse_args()

    from datetime import date as _date
    day = args.date or _date.today().isoformat()
    out_dir = Path(args.outdir)
    template_path = Path(args.template)

    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    out_path, exp_id = create_log(day, out_dir, template_path)
    print(f"Created: {out_path}")
    print(f"Experiment ID: {exp_id}")

    if args.no_commit:
        return 0

    if args.commit_all:
        _run(["git", "add", "-A"])
    else:
        _run(["git", "add", str(out_path)])
        for extra in args.commit_path:
            _run(["git", "add", extra])
    message = args.message or f"Add experiment log {exp_id}"
    _run(["git", "commit", "-m", message])

    if not args.no_tag:
        tag = args.tag or exp_id
        if _git_tag_exists(tag) and not args.force_tag:
            raise RuntimeError(f"Tag already exists: {tag} (use --force-tag to overwrite)")
        if args.force_tag and _git_tag_exists(tag):
            _run(["git", "tag", "-f", tag])
        else:
            _run(["git", "tag", tag])

    if args.push:
        branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
        _run(["git", "push", "origin", branch])
        if not args.no_tag:
            tag = args.tag or exp_id
            _run(["git", "push", "origin", tag])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
