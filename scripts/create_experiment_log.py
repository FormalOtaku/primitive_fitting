#!/usr/bin/env python3
"""Create a new experiment log from the Japanese template.

- Fills date, experiment id, branch, commit hash automatically.
- Writes to ./experiments by default.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from datetime import date
from pathlib import Path


def _run_git(args: list[str]) -> str:
    try:
        out = subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def _next_suffix(out_dir: Path, day: str) -> str:
    pattern = re.compile(rf"EXPERIMENT_LOG_{re.escape(day)}-(\\d{{2}})\\.md$")
    suffixes = []
    for path in out_dir.glob(f"EXPERIMENT_LOG_{day}-*.md"):
        m = pattern.match(path.name)
        if m:
            try:
                suffixes.append(int(m.group(1)))
            except ValueError:
                pass
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


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Create a new experiment log from template")
    parser.add_argument("--date", help="YYYY-MM-DD (default: today)")
    parser.add_argument("--outdir", default=str(root / "experiments"))
    parser.add_argument("--template", default=str(root / "docs" / "EXPERIMENT_LOG_TEMPLATE_JP.md"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    day = args.date or date.today().isoformat()
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = _next_suffix(out_dir, day)
    exp_id = f"exp-{day}-{suffix}"
    out_path = out_dir / f"EXPERIMENT_LOG_{day}-{suffix}.md"

    template_path = Path(args.template)
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    template_text = template_path.read_text(encoding="utf-8")
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    commit = _run_git(["rev-parse", "--short", "HEAD"])

    filled = _fill_template(template_text, day, exp_id, branch, commit)

    if args.dry_run:
        print(filled)
        return 0

    out_path.write_text(filled, encoding="utf-8")
    print(f"Created: {out_path}")
    print(f"Experiment ID: {exp_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
