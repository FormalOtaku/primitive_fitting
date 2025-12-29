#!/usr/bin/env python3
"""Execute commands listed in a Markdown runbook.

Runbook format:
- Only executes fenced code blocks marked with ```bash or ```sh.
- By default, requires a line "# RUN" as the first non-empty line in the block.
- Use --all to run all bash/sh blocks without marker.
"""

from __future__ import annotations

import argparse
import subprocess
from datetime import date as _date
from pathlib import Path
from typing import List, Optional, Tuple


def _extract_blocks(text: str) -> List[List[str]]:
    blocks: List[List[str]] = []
    lines = text.splitlines()
    in_block = False
    block: List[str] = []
    block_lang = ""

    for line in lines:
        if not in_block and line.strip().startswith("```"):
            lang = line.strip()[3:].strip().lower()
            if lang in ("bash", "sh"):
                in_block = True
                block = []
                block_lang = lang
            continue
        if in_block and line.strip().startswith("```"):
            if block_lang in ("bash", "sh"):
                blocks.append(block)
            in_block = False
            block = []
            block_lang = ""
            continue
        if in_block:
            block.append(line)

    return blocks


def _should_run(block: List[str], allow_all: bool) -> bool:
    if allow_all:
        return True
    for line in block:
        if line.strip() == "":
            continue
        return line.strip() == "# RUN"
    return False


def _strip_run_marker(block: List[str]) -> List[str]:
    stripped = []
    removed = False
    for line in block:
        if not removed and line.strip() == "# RUN":
            removed = True
            continue
        stripped.append(line)
    return stripped


def _run_git(args: list[str]) -> str:
    try:
        out = subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def _call(args: list[str]) -> None:
    subprocess.check_call(args)


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


def _create_log(day: str, out_dir: Path, template_path: Path) -> Tuple[Path, str]:
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


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run commands from a Markdown runbook")
    parser.add_argument("path", help="Path to runbook .md")
    parser.add_argument("--all", action="store_true", help="Run all bash/sh blocks (no marker needed)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--stop-on-error", action="store_true", default=True)

    # Experiment log options
    parser.add_argument("--log", action="store_true", help="Create an experiment log before running")
    parser.add_argument("--date", help="YYYY-MM-DD for log (default: today)")
    parser.add_argument("--outdir", default=str(root / "experiments"))
    parser.add_argument("--template", default=str(root / "docs" / "EXPERIMENT_LOG_TEMPLATE_JP.md"))

    # Commit/tag/push options
    parser.add_argument("--commit", action="store_true", help="Commit after run")
    parser.add_argument("--commit-all", action="store_true", help="git add -A before commit")
    parser.add_argument("--commit-path", action="append", default=[], help="extra path to git add")
    parser.add_argument("--message", help="commit message")
    parser.add_argument("--tag", action="store_true", help="Create tag after commit")
    parser.add_argument("--tag-name", help="Override tag name")
    parser.add_argument("--force-tag", action="store_true", help="Overwrite existing tag")
    parser.add_argument("--push", action="store_true", help="Push branch and tag")
    args = parser.parse_args()

    path = Path(args.path)
    text = path.read_text(encoding="utf-8")
    blocks = _extract_blocks(text)

    if not blocks:
        print("No runnable blocks found.")
        return 1

    log_path: Optional[Path] = None
    exp_id: Optional[str] = None
    if args.log:
        day = args.date or _date.today().isoformat()
        out_dir = Path(args.outdir)
        template_path = Path(args.template)
        log_path, exp_id = _create_log(day, out_dir, template_path)
        print(f"Created log: {log_path}")
        print(f"Experiment ID: {exp_id}")

    ran = 0
    for block in blocks:
        if not _should_run(block, args.all):
            continue
        cmd_lines = _strip_run_marker(block)
        cmd = "\n".join(cmd_lines).strip()
        if not cmd:
            continue
        ran += 1
        print(f"\n[RUN] block {ran}\n{cmd}\n")
        if args.dry_run:
            continue
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"Command failed with code {result.returncode}")
            if args.stop_on_error:
                return result.returncode

    if ran == 0:
        print("No blocks matched run criteria.")
        return 1

    if args.commit:
        if args.commit_all:
            _call(["git", "add", "-A"])
        else:
            staged_any = False
            if log_path is not None:
                _call(["git", "add", str(log_path)])
                staged_any = True
            for extra in args.commit_path:
                _call(["git", "add", extra])
                staged_any = True
            if not staged_any:
                raise RuntimeError("No paths staged. Use --commit-all or --commit-path")

        message = args.message
        if message is None:
            tag_hint = f" {exp_id}" if exp_id else ""
            message = f"Run runbook {path.name}{tag_hint}"
        _call(["git", "commit", "-m", message])

        if args.tag:
            tag_name = args.tag_name or exp_id
            if not tag_name:
                raise RuntimeError("Tag name not specified and no experiment log created")
            exists = _run_git(["rev-parse", tag_name]) != "unknown"
            if exists and not args.force_tag:
                raise RuntimeError(f"Tag already exists: {tag_name} (use --force-tag)")
            if exists and args.force_tag:
                _call(["git", "tag", "-f", tag_name])
            else:
                _call(["git", "tag", tag_name])

        if args.push:
            branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
            _call(["git", "push", "origin", branch])
            if args.tag:
                tag_name = args.tag_name or exp_id
                _call(["git", "push", "origin", tag_name])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
