"""Copy shared modules into the task files that use them (PLAN.md 3.1, D6).

A push uploads only the task file, so on Kaggle a task can't import
randombench. Instead, a task file holds a copy of each module it needs,
between two marker lines:

    # --- begin randombench/prompts.py ---
    # --- end randombench/prompts.py ---

`python -m randombench.sync` replaces everything between each pair with the
module's current text, so the file run locally is the file pushed. To give
a task a module, add an empty pair of markers and run the sync. A test fails
when a copy is out of date.

A copy is the module's text, unchanged, so the module must stand alone: it
can't import randombench itself. Whether jupytext changes any line when the
CLI converts the file is checked by a test that runs the conversion.

Usage: python -m randombench.sync [task files]   (default: tasks/*.py)
"""

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

NOTE = "# Copied by `python -m randombench.sync` (PLAN.md D6). Edit the source, not this copy.\n"

_BEGIN = re.compile(r"^# --- begin (\S+) ---$", re.MULTILINE)
_REGION = re.compile(
    r"^# --- begin (?P<path>\S+) ---\n.*?^# --- end (?P=path) ---$",
    re.MULTILINE | re.DOTALL,
)


class SyncError(Exception):
    """A task file's markers or a copied module break the rules above."""


def imports_randombench(source: str) -> bool:
    """Whether the code imports randombench or any module in it."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names = [node.module or ""]
        else:
            continue
        if any(name.split(".")[0] == "randombench" for name in names):
            return True
    return False


def module_text(path: str) -> str:
    """Return a module's text for copying, which must end with a newline."""
    file = ROOT / path
    if not file.is_file():
        raise SyncError(f"{path}: no such file")
    text = file.read_text(encoding="utf-8")
    if imports_randombench(text):
        raise SyncError(f"{path} imports randombench, so its copy wouldn't run on Kaggle")
    return text if text.endswith("\n") else text + "\n"


def synced(task_text: str) -> str:
    """Return the task file's text with every copy replaced by its module's text."""
    regions = list(_REGION.finditer(task_text))
    if len(regions) != len(_BEGIN.findall(task_text)):
        raise SyncError("a begin marker has no matching end marker")
    paths = [m["path"] for m in regions]
    if len(paths) != len(set(paths)):
        raise SyncError(f"a module is copied twice: {paths}")

    def fill(m: re.Match) -> str:
        path = m["path"]
        return f"# --- begin {path} ---\n{NOTE}{module_text(path)}# --- end {path} ---"

    return _REGION.sub(fill, task_text)


def sync(task_file: Path) -> bool:
    """Bring a task file's copies up to date. Return whether anything changed."""
    task_file = Path(task_file)
    text = task_file.read_text(encoding="utf-8")
    new = synced(text)
    if new == text:
        return False
    task_file.write_text(new, encoding="utf-8", newline="\n")
    return True


def task_files() -> list[Path]:
    return sorted((ROOT / "tasks").glob("*.py"))


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    for task_file in [Path(a) for a in args] or task_files():
        print(f"{task_file}: {'updated' if sync(task_file) else 'up to date'}")


if __name__ == "__main__":
    main()
