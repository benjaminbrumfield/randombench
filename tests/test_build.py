"""The scoring task files (PLAN.md 4.5b), checked offline.

No model is reached. Each file runs with stand-in models that fail if
prompted, and its score must match `python -m randombench.score`.
"""

import ast
import runpy

import pytest
from slugify import slugify

import kaggle_benchmarks as kbench
from kaggle.api.kaggle_api_extended import KaggleApi
from kaggle_benchmarks import events
from kaggle_benchmarks.ui.console import ConsoleUI

from randombench.build import (
    DATA,
    SCORES_DIR,
    SCORING_TASKS,
    BuildError,
    ScoringTask,
    format_stored,
    render,
    stored_replies,
)
from randombench.checks import score_answers
from randombench.extract import load_draws
from randombench.score import score_draws

DRAWS = load_draws(DATA)
STORED = stored_replies(DRAWS)
EXPECTED = score_draws(DRAWS)  # what `python -m randombench.score` reports
IDS = [t.slug for t in SCORING_TASKS]


# The ten tasks


def test_one_task_per_score_in_score_answers_order():
    assert [t.score for t in SCORING_TASKS] == list(score_answers([1, 2]))


def test_the_titles_are_the_decided_ones():
    # Each title fixes a slug at its first push (D10), so a change here is a decision.
    assert [t.title for t in SCORING_TASKS] == [
        "RandomBench Score",
        "RandomBench Valid Answers",
        "RandomBench Uniformity",
        "RandomBench Primes",
        "RandomBench Multiples of 10",
        "RandomBench Multiples of 5",
        "RandomBench Favorite Number",
        "RandomBench Edges",
        "RandomBench Repeated Digits",
        "RandomBench Last Digit",
    ]


def test_each_slug_is_what_the_cli_makes_of_the_title():
    # The CLI matches a pushed task by slugifying its title.
    assert [t.slug for t in SCORING_TASKS] == [slugify(t.title) for t in SCORING_TASKS]
    assert len(set(IDS)) == len(IDS)


# The files


@pytest.mark.parametrize("task", SCORING_TASKS, ids=IDS)
def test_the_task_file_is_up_to_date(task):
    assert task.file.read_text(encoding="utf-8") == render(task, STORED), (
        "out of date: run `python -m randombench.build`"
    )


def test_tasks_scores_holds_only_the_ten():
    assert sorted(p.name for p in SCORES_DIR.iterdir()) == sorted(t.file.name for t in SCORING_TASKS)


@pytest.mark.parametrize("task", SCORING_TASKS, ids=IDS)
def test_the_cli_finds_exactly_this_tasks_title(task):
    text = task.file.read_text(encoding="utf-8")
    assert KaggleApi._get_task_names_from_file(text) == [task.title]
    KaggleApi._validate_task_in_file(task.slug, str(task.file), text)  # raises if not found


def test_render_rejects_a_title_that_would_break_its_string():
    with pytest.raises(BuildError):
        render(ScoringTask("primes", 'RandomBench "Primes"', "x"), STORED)


# The stored replies (D13)


def test_the_tally_accounts_for_every_draw():
    for model, s in STORED.items():
        draws = [d for d in DRAWS if d["model"] == model]
        assert sum(s["replies"].values()) + s["failed"] == len(draws)
        assert s["runs"] == sorted({d["run_id"] for d in draws})


def test_the_stored_replies_read_back_exactly():
    awkward = {
        "test/model": {
            "runs": ["9", "10"],
            "failed": 2,
            "replies": {"It's \"42\"\n": 3, "−47": 2, "42 \U0001F600": 1, "": 1, "\\d": 1},
        }
    }
    for stored in [STORED, awkward]:
        text = format_stored(stored)
        assert text.isascii()
        assert ast.literal_eval(text) == stored


# Running a task file with stand-in models


class StandIn(kbench.LLMChat):
    """A model the scoring tasks must never prompt."""

    def __init__(self, name: str):
        super().__init__(name=name)

    def invoke(self, *args, **kwargs):
        raise AssertionError(f"{self.name} was prompted")


def task_namespace(task, tmp_path, monkeypatch):
    """The task file's namespace, without its last two cells, which run the
    task and %choose it. Run files land in tmp_path.

    The local console UI is detached. With the -> tuple[float, float]
    annotation, it formats a False result as a pair and raises, after the
    run file is written (1.3). Kaggle's notebook UI doesn't: 1.3's creation
    run returned False and completed.
    """
    listeners = [x for x in events.manager.listeners if not isinstance(x, ConsoleUI)]
    monkeypatch.setattr(events.manager, "listeners", listeners)
    cells = task.file.read_text(encoding="utf-8").split("\n# %%\n")
    body = [c for c in cells if "score.run(" not in c and "# %choose" not in c]
    assert len(body) == len(cells) - 2, "expected the last two cells to run the task and %choose it"
    code = tmp_path / "score_cells.py"
    code.write_text("\n# %%\n".join(body), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return runpy.run_path(str(code))


@pytest.mark.parametrize("task", SCORING_TASKS, ids=IDS)
def test_each_stored_model_gets_the_score_and_interval_the_analysis_gives(task, tmp_path, monkeypatch):
    ns = task_namespace(task, tmp_path, monkeypatch)
    for expected in EXPECTED:
        run = ns["score"].run(StandIn(expected.model))
        # The same counts and the same seed give the same resamples, so it's exact.
        assert run.result == (expected.scores[task.score], expected.intervals[task.score]), expected.model


@pytest.mark.parametrize("task", SCORING_TASKS, ids=IDS)
def test_a_model_without_stored_replies_gets_false(task, tmp_path, monkeypatch, capsys):
    # Such as .env's LLM_DEFAULT in a local run (D12). The creation run's
    # model, Gemini 3.8 Flash, has stored replies since 4.7 (D21).
    ns = task_namespace(task, tmp_path, monkeypatch)
    assert ns["score"].run(StandIn("google/gemini-3-flash-preview")).result is False
    assert "no stored replies" in capsys.readouterr().out


@pytest.mark.parametrize("task", SCORING_TASKS, ids=IDS)
def test_a_score_of_none_gets_false(task, tmp_path, monkeypatch):
    # One valid reply in four: only the valid-answer score exists (3.4).
    ns = task_namespace(task, tmp_path, monkeypatch)
    one_valid = {"runs": ["1"], "failed": 0, "replies": {"42": 1, "forty-two": 3}}
    monkeypatch.setitem(ns["STORED"], "test/one-valid", one_valid)
    result = ns["score"].run(StandIn("test/one-valid")).result
    if task.score == "invalid_answers":
        assert result[0] == 0.25
    else:
        assert result is False
