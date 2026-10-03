#!/usr/bin/env python3
"""
Build the one file you upload for this week's lab.

    python make_submission.py

It asks for your student number the first time and remembers it. It checks that
SUBMISSION.md is finished, runs setup_check.py, gathers your files, and writes a
single zip named after you.

    python make_submission.py --force      submit anyway, recorded as incomplete
    python make_submission.py --week 2     if it picks the wrong week
    python make_submission.py --dry-run    show what would go in, write nothing

Nothing is sent anywhere. This runs entirely on your machine and only writes a
file. Standard library only — no installs.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# Used only if the week cannot be worked out from the folder path.
DEFAULT_WEEK = 2

LAB_DIR = Path(__file__).resolve().parent
SUBMISSION_MD = LAB_DIR / "SUBMISSION.md"
SETUP_CHECK = LAB_DIR / "setup_check.py"
STUDENT_FILE = LAB_DIR / ".student_number"
EXTRA_DIR = LAB_DIR / "submission"
SETUP_OUTPUT = LAB_DIR / "setup_check_output.txt"

# What counts as your work: the section*_*.py / exercise_7_*.py files you wrote,
# any notebook, and the plots you saved. The exclusions below decide the rest.
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
WORK_SUFFIXES = IMAGE_SUFFIXES | {".py", ".ipynb"}
TOOL_SCRIPTS = {"make_submission.py", "setup_check.py"}

# Never collected: the provided data and figures, environments, caches, and
# anything this script produced on a previous run.
EXCLUDED_DIRS = {
    "figures", "__pycache__", ".git", ".venv", "venv", "env",
    ".ipynb_checkpoints", ".idea", ".vscode", "node_modules", ".pytest_cache",
    "data",
}
EXCLUDED_NAMES = {".student_number", "UCI_Credit_Card.csv"}
MAX_FILE_MB = 20


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def say(msg: str = "") -> None:
    print(msg, flush=True)


def stop(msg: str) -> "NoReturn":  # type: ignore[valid-type]
    say()
    say("=" * 68)
    say("  STOPPED")
    say("=" * 68)
    say()
    say(msg.rstrip())
    say()
    sys.exit(1)


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def is_excluded(path: Path) -> bool:
    if path.name in EXCLUDED_NAMES or path.name.startswith("~$"):
        return True
    rel = path.relative_to(LAB_DIR)
    return any(part in EXCLUDED_DIRS for part in rel.parts)


# --------------------------------------------------------------------------
# week, student number, completeness
# --------------------------------------------------------------------------

def detect_week(override: int | None) -> tuple[int, str]:
    """--week wins, then the folder path, then DEFAULT_WEEK."""
    if override is not None:
        return override, "you told it"

    patterns = (
        re.compile(r"topic-0*(\d+)"),
        re.compile(r"week-?0*(\d+)"),
        re.compile(r"wk-?0*(\d+)"),
    )
    for parent in [LAB_DIR, *LAB_DIR.parents]:
        name = parent.name.lower()
        for pat in patterns:
            m = pat.search(name)
            if m:
                week = int(m.group(1))
                if 0 <= week <= 13:
                    return week, f"from the folder name '{parent.name}'"
    return DEFAULT_WEEK, "the default for this lab"


def get_student_number() -> str:
    if STUDENT_FILE.exists():
        saved = STUDENT_FILE.read_text(encoding="utf-8").strip()
        if re.fullmatch(r"\d{5,10}", saved):
            return saved
        say(f"  The saved student number '{saved}' does not look right. Asking again.")

    say()
    say("  Your student number, digits only (e.g. 20012345).")
    say("  It is saved to .student_number so you are only asked once.")
    say()
    for _ in range(3):
        try:
            entered = input("  Student number: ").strip()
        except (EOFError, KeyboardInterrupt):
            stop("No student number given, so nothing was built.")
        entered = re.sub(r"[^0-9]", "", entered)
        if re.fullmatch(r"\d{5,10}", entered):
            STUDENT_FILE.write_text(entered + "\n", encoding="utf-8")
            say(f"  Saved. Delete {STUDENT_FILE.name} if you need to change it.")
            return entered
        say("  That is not 5 to 10 digits. Try again.")
    stop("Three tries, no valid student number. Nothing was built.")


def _tidy(text: str, limit: int = 58) -> str:
    text = re.sub(r"\s+", " ", text.strip().strip("*").strip())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "\u2026"


def find_todos(text: str) -> list[tuple[int, str]]:
    """Line numbers of remaining TODO markers, with the question each sits under.

    A TODO inside backticks is this file talking about the markers rather than
    being one, so it does not count.
    """
    heading = "(top of file)"
    pending: list[str] = []
    found = []

    for i, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()

        if pending:                                   # a bold question that wrapped
            pending.append(stripped)
            if stripped.endswith("**"):
                heading = _tidy(" ".join(pending))
                pending = []
        elif stripped.startswith("#"):
            heading = _tidy(stripped.lstrip("#"))
        elif stripped.startswith("**"):
            if stripped.endswith("**") and len(stripped) > 4:
                heading = _tidy(stripped)
            else:
                pending = [stripped]

        if "TODO" in re.sub(r"`[^`]*`", "", line):
            found.append((i, heading))

    return found


# --------------------------------------------------------------------------
# the steps
# --------------------------------------------------------------------------

def check_submission_md(force: bool) -> bool:
    if not SUBMISSION_MD.exists():
        stop(
            f"There is no SUBMISSION.md in {LAB_DIR}.\n\n"
            "It came with the lab files. Unzip them again, or run this script from\n"
            "the folder that has SUBMISSION.md in it."
        )

    text = SUBMISSION_MD.read_text(encoding="utf-8")
    todos = find_todos(text)

    if not todos:
        say("  SUBMISSION.md   complete, no TODO markers left")
        return True

    if not force:
        listing = "\n".join(f"    line {ln:>4}   {h}" for ln, h in todos[:15])
        more = f"\n    ... and {len(todos) - 15} more" if len(todos) > 15 else ""
        stop(
            f"SUBMISSION.md still has {len(todos)} TODO marker(s) in it, which means it\n"
            f"is not finished. Search the file for the word TODO:\n\n"
            f"{listing}{more}\n\n"
            "Answer those sections, delete each TODO, and run this again.\n\n"
            "If you need to submit it incomplete — and sometimes that is the right\n"
            "call — run:\n\n"
            "    python make_submission.py --force"
        )

    say(f"  SUBMISSION.md   {len(todos)} TODO left, submitting anyway (--force)")
    return False


def run_setup_check() -> bool:
    if not SETUP_CHECK.exists():
        say("  setup_check.py  not found, skipping the environment report")
        return False
    try:
        result = subprocess.run(
            [sys.executable, str(SETUP_CHECK)],
            cwd=LAB_DIR, capture_output=True, text=True, timeout=180,
        )
        output = (result.stdout or "") + (result.stderr or "")
        note = f"(exit code {result.returncode})"
    except subprocess.TimeoutExpired:
        output = "setup_check.py did not finish within 180 seconds."
        note = "(timed out)"
    except Exception as exc:  # noqa: BLE001 — never let this stop a submission
        output = f"setup_check.py could not be run: {exc!r}"
        note = "(failed to run)"

    limit = 50_000                      # a stack trace can carry a whole file with it
    if len(output) > limit:
        output = output[:limit] + f"\n\n... truncated, {len(output) - limit} more characters\n"

    header = (
        f"setup_check.py output\n"
        f"{datetime.now().astimezone():%Y-%m-%d %H:%M %Z}\n"
        f"{'-' * 60}\n"
    )
    SETUP_OUTPUT.write_text(header + output, encoding="utf-8")
    say(f"  setup_check.py  run, output captured {note}")
    return True


def collect_files() -> tuple[list[tuple[Path, str]], list[str]]:
    """Returns (files to zip as (source, name-inside-zip)), and skip messages."""
    picked: list[tuple[Path, str]] = []
    skipped: list[str] = []
    seen: set[Path] = set()

    def add(path: Path, arcname: str) -> None:
        if path in seen or not path.is_file() or is_excluded(path):
            return
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_FILE_MB:
            skipped.append(f"{arcname} — {size_mb:.0f} MB, over the {MAX_FILE_MB} MB limit")
            return
        seen.add(path)
        picked.append((path, arcname))

    add(SUBMISSION_MD, "SUBMISSION.md")

    # Your own work saved in the lab folder itself: scripts, notebooks, plots.
    for f in sorted(LAB_DIR.iterdir()):
        if not f.is_file() or f.name in TOOL_SCRIPTS:
            continue
        if f.name.startswith("submission-week") and f.suffix == ".zip":
            continue  # a zip from a previous run
        if f.suffix.lower() in WORK_SUFFIXES:
            add(f, f.name)

    # Anything you chose to include.
    if EXTRA_DIR.is_dir():
        for f in sorted(EXTRA_DIR.rglob("*")):
            if f.is_file():
                add(f, str(f.relative_to(LAB_DIR)).replace("\\", "/"))

    if SETUP_OUTPUT.exists():
        add(SETUP_OUTPUT, SETUP_OUTPUT.name)

    return picked, skipped


def build_manifest(student: str, week: int, complete: bool,
                   files: list[tuple[Path, str]]) -> dict:
    return {
        "student_number": student,
        "week": week,
        "module": "AI Principles and Practice (COMP-0987)",
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "built_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "complete": complete,
        "files": [name for _, name in files],
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "tool_version": "1.1",
    }


# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the single zip you upload for this week's lab.")
    parser.add_argument("--force", action="store_true",
                        help="build even if SUBMISSION.md is unfinished")
    parser.add_argument("--week", type=int, default=None,
                        help="set the week number explicitly")
    parser.add_argument("--dry-run", action="store_true",
                        help="show what would be included, write nothing")
    args = parser.parse_args()

    week, why = detect_week(args.week)

    say()
    say("=" * 68)
    say(f"  Week {week:02d} lab submission")
    say("=" * 68)
    say(f"  Folder          {LAB_DIR}")
    say(f"  Week            {week} ({why})")
    say()

    complete = check_submission_md(args.force)
    run_setup_check()

    files, skipped = collect_files()
    total = sum(p.stat().st_size for p, _ in files)

    say()
    say(f"  Collected {len(files)} file(s), {human(total)}:")
    for _, name in files:
        say(f"    {name}")
    say("    manifest.json")
    for note in skipped:
        say(f"  SKIPPED  {note}")
    work = [n for _, n in files
            if n.endswith((".py", ".ipynb")) and not n.startswith("submission/")]
    if not work:
        say()
        say("  No scripts or notebooks found in this folder. If your Section 1-7 code")
        say("  lives somewhere else, copy it here (or into a folder called 'submission')")
        say("  and run this again — the code is part of the hand-in.")

    if args.dry_run:
        saved = STUDENT_FILE.read_text(encoding="utf-8").strip() if STUDENT_FILE.exists() else ""
        student = saved or "<your-number>"
        say()
        say("  Dry run — nothing written. It would have made "
            f"submission-week{week:02d}-{student}.zip")
        say()
        return 0

    student = get_student_number()
    manifest = build_manifest(student, week, complete, files)
    out = LAB_DIR / f"submission-week{week:02d}-{student}.zip"

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for path, name in files:
            z.write(path, name)
        z.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")

    say()
    say("=" * 68)
    say("  DONE" + ("" if complete else "  (recorded as incomplete)"))
    say("=" * 68)
    say()
    say(f"  Your file    {out}")
    say(f"  Size         {human(out.stat().st_size)}")
    say()
    say("  Upload that one file. Not the folder, not the individual files.")
    say()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        say()
        say("  Cancelled. Nothing was written.")
        sys.exit(1)
