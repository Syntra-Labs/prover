#!/usr/bin/env python3
"""
Diff Size and Reviewability Linter.

Enforces strict reviewability budgets:
- Max 300 lines changed (insertions + deletions) per commit (--staged).
- Max 500 lines changed (insertions + deletions) per PR (--compare-base <base>).

Ensures all commits and PRs are small, focused, and manually auditable.
"""

import sys
import argparse
import subprocess
from pathlib import Path

# Configurable Default Budgets
DEFAULT_MAX_COMMIT_LINES = 300
DEFAULT_MAX_PR_LINES = 500

# Paths or patterns to exclude from line counting (e.g. submodule pointers, binary data)
EXCLUDED_PATTERNS = [
    "vendor/leandojo",
    "vendor/uor-r4",
    "reports/",
    "logs/",
    "scripts/patches/",
    ".lock",
    "lock.json",
]

def run_git(args: list[str]) -> str:
    res = subprocess.run(["git"] + args, capture_output=True, text=True, check=True)
    return res.stdout.strip()

def is_excluded(filepath: str) -> bool:
    for pat in EXCLUDED_PATTERNS:
        if pat in filepath:
            return True
    return False

def get_diff_stats(git_diff_args: list[str]) -> tuple[int, list[tuple[str, int, int]]]:
    """
    Parses `git diff --numstat` output.
    Returns (total_lines_changed, list_of_(file, ins, dels)).
    """
    try:
        output = run_git(["diff", "--numstat"] + git_diff_args)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to run git diff: {e.stderr}", file=sys.stderr)
        sys.exit(2)

    total_lines = 0
    file_records = []

    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        ins_str, del_str, filepath = parts[0], parts[1], parts[2]

        # Ignore binary files or excluded submodules
        if ins_str == "-" or del_str == "-":
            continue
        if is_excluded(filepath):
            continue

        ins = int(ins_str)
        dels = int(del_str)
        line_count = ins + dels
        total_lines += line_count
        file_records.append((filepath, ins, dels))

    return total_lines, file_records

def main():
    parser = argparse.ArgumentParser(description="Enforce reviewable diff size budgets.")
    parser.add_argument("--staged", action="store_true", help="Check staged changes for commit hook (max 300 lines).")
    parser.add_argument("--compare-base", type=str, default=None, help="Check branch diff against base (e.g., origin/leanDojo).")
    parser.add_argument("--max-lines", type=int, default=None, help="Override maximum lines allowed.")

    args = parser.parse_args()

    if not args.staged and not args.compare_base:
        parser.error("Must specify either --staged or --compare-base <base>.")

    if args.staged:
        mode_label = "Pre-Commit Staged Diff"
        limit = args.max_lines or DEFAULT_MAX_COMMIT_LINES
        diff_args = ["--cached"]
    else:
        mode_label = f"Pre-Push PR Diff (vs {args.compare_base})"
        limit = args.max_lines or DEFAULT_MAX_PR_LINES
        # Find common ancestor
        try:
            merge_base = run_git(["merge-base", "HEAD", args.compare_base])
            diff_args = [f"{merge_base}..HEAD"]
        except subprocess.CalledProcessError:
            diff_args = [f"{args.compare_base}...HEAD"]

    total_lines, files = get_diff_stats(diff_args)

    print("=" * 70)
    print(f"DIFF SIZE REVIEWABILITY AUDIT: {mode_label}")
    print("=" * 70)

    if not files:
        print("[PASS] No tracked text modifications detected.")
        print("=" * 70)
        sys.exit(0)

    print(f"{'Lines':<10} {'+Ins':<8} {'-Del':<8} {'File':<40}")
    print("-" * 70)
    for filepath, ins, dels in sorted(files, key=lambda x: x[1] + x[2], reverse=True):
        count = ins + dels
        print(f"{count:<10} +{ins:<7} -{dels:<7} {filepath:<40}")
    print("-" * 70)
    print(f"Total Modified Lines: {total_lines} / {limit} max allowed")

    if total_lines > limit:
        print("=" * 70)
        print(f"[REJECTED] Diff exceeds the maximum reviewability budget of {limit} lines!")
        print("=" * 70)
        print("Remediation:")
        print("  1. Split this change into smaller, atomic commits / PRs.")
        print("  2. Focus each PR on a single logical invariant, component, or task.")
        print("  3. (Emergency bypass only: `git commit --no-verify` or `git push --no-verify`)")
        print("=" * 70)
        sys.exit(1)

    print(f"[PASS] Diff size is within the reviewable budget ({total_lines} <= {limit}).")
    print("=" * 70)
    sys.exit(0)

if __name__ == "__main__":
    main()
