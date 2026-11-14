import jupytext
import pytest

from randombench import sync
from randombench.sync import NOTE, SyncError, imports_randombench, synced, task_files

TASKS = task_files()


# The task files in tasks/


def test_there_are_task_files():
    assert TASKS


@pytest.mark.parametrize("task", TASKS, ids=lambda p: p.name)
def test_task_copies_are_up_to_date(task):
    text = task.read_text(encoding="utf-8")
    assert synced(text) == text, "a copy is out of date: run `python -m randombench.sync`"


@pytest.mark.parametrize("task", TASKS, ids=lambda p: p.name)
def test_tasks_dont_import_randombench(task):
    # On Kaggle there's no randombench to import (D6).
    assert not imports_randombench(task.read_text(encoding="utf-8"))


def code_after_conversion(text):
    """The non-blank code lines of the notebook the CLI makes from a task file
    (`_convert_py_to_notebook` in kaggle/api/kaggle_api_extended.py)."""
    notebook = jupytext.reads(text, fmt="py:percent")
    return [
        line
        for cell in notebook.cells
        if cell.cell_type == "code"
        for line in cell.source.splitlines()
        if line.strip()
    ]


def code_as_written(text):
    """The non-blank lines of a task file, without its cell markers. A
    `# %choose` comment is the one magic written on purpose (PLAN.md 1.2)."""
    lines = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("# %%"):
            continue
        lines.append(line[2:] if line.startswith("# %choose ") else line)
    return lines


@pytest.mark.parametrize("task", TASKS, ids=lambda p: p.name)
def test_conversion_to_a_notebook_changes_no_line(task):
    # jupytext turns "# %magic" into a magic and "# !cmd" into a shell
    # command, and "#%%" splits a cell, so a stray comment in a task or in
    # a copied module could run on Kaggle.
    text = task.read_text(encoding="utf-8")
    assert code_after_conversion(text) == code_as_written(text)


@pytest.mark.parametrize("comment", ["# !rm -rf data", "# %pip install x", "#%%"])
def test_the_conversion_check_catches_a_stray_comment(comment):
    text = f"# %%\nx = 1\n{comment}\ny = 2\n"
    assert code_after_conversion(text) != code_as_written(text)


def test_the_conversion_check_allows_choose():
    text = "# %%\nx = 1\n# %choose randombench-collect\n"
    assert code_after_conversion(text) == code_as_written(text)


# Syncing


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A stand-in repo root holding one module, mod.py."""
    (tmp_path / "mod.py").write_text("X = 1\n", encoding="utf-8")
    monkeypatch.setattr(sync, "ROOT", tmp_path)
    return tmp_path


EMPTY = "a = 0\n# --- begin mod.py ---\n# --- end mod.py ---\nb = 2\n"
FILLED = f"a = 0\n# --- begin mod.py ---\n{NOTE}X = 1\n# --- end mod.py ---\nb = 2\n"


def test_fills_an_empty_copy(root):
    assert synced(EMPTY) == FILLED


def test_replaces_a_stale_copy(root):
    stale = FILLED.replace("X = 1", "X = 0\nOLD = True")
    assert synced(stale) == FILLED


def test_leaves_a_current_copy_alone(root):
    assert synced(FILLED) == FILLED


def test_leaves_a_file_without_markers_alone(root):
    assert synced("a = 0\n") == "a = 0\n"


def test_sync_rewrites_the_file_only_when_stale(root):
    task = root / "task.py"
    task.write_text(EMPTY, encoding="utf-8")
    assert sync.sync(task) is True
    assert task.read_text(encoding="utf-8") == FILLED
    assert sync.sync(task) is False


def test_adds_a_missing_final_newline(root):
    (root / "mod.py").write_text("X = 1", encoding="utf-8")
    assert synced(EMPTY) == FILLED


def test_rejects_a_begin_marker_without_an_end(root):
    with pytest.raises(SyncError):
        synced("# --- begin mod.py ---\nX = 1\n")


def test_rejects_an_end_marker_for_another_module(root):
    with pytest.raises(SyncError):
        synced("# --- begin mod.py ---\n# --- end other.py ---\n")


def test_rejects_a_module_copied_twice(root):
    with pytest.raises(SyncError):
        synced(EMPTY + EMPTY)


def test_rejects_a_missing_module(root):
    with pytest.raises(SyncError):
        synced("# --- begin nope.py ---\n# --- end nope.py ---\n")


def test_rejects_a_module_that_imports_randombench(root):
    (root / "mod.py").write_text("from randombench.parsing import parse_reply\n", encoding="utf-8")
    with pytest.raises(SyncError):
        synced(EMPTY)


# Spotting imports


@pytest.mark.parametrize(
    "code",
    [
        "import randombench",
        "import randombench.prompts as p",
        "from randombench.prompts import PROMPTS",
        "from randombench import prompts",
        "def f():\n    import randombench.parsing",
    ],
)
def test_spots_an_import_of_randombench(code):
    assert imports_randombench(code)


@pytest.mark.parametrize(
    "code",
    [
        "import random",
        "import randombenchmark",
        "from .prompts import PROMPTS",
        "# It imports randombench, which a push wouldn't upload.",
        "s = 'from randombench import x'",
    ],
)
def test_ignores_what_isnt_an_import_of_randombench(code):
    assert not imports_randombench(code)


# The modules that task files copy


@pytest.mark.parametrize("module", ["prompts", "parsing", "checks"])
def test_the_shared_modules_can_be_copied(module):
    # A scoring task copies the parser and the checks (3.4, 3.6), so neither
    # may import randombench. module_text raises if one does.
    assert sync.module_text(f"randombench/{module}.py")
