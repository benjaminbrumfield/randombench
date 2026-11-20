"""Checking both Benchmarks against data/, offline.

No test reaches Kaggle: each builds the rows the API would return.
"""

import json

from randombench.build import SCORING_TASKS
from randombench.leaderboards import BENCHMARKS, check, format_check, local_results, read_rows

HEADLINE = "randombench-score"
SUB_SCORES = [t.slug for t in SCORING_TASKS if t.slug != HEADLINE]


def cell(task, value, interval, version=6):
    numeric = {}
    if value:
        numeric["value"] = value
    if interval:
        numeric["confidenceInterval"] = interval
    return {
        "benchmarkTaskName": task,
        "benchmarkTaskSlug": f"/benchmarks/tasks/benjaminbrumfield/{task}",
        "taskVersion": version,
        "result": {"numericResult": numeric},
    }


def row(slug, cells):
    return {"modelVersionName": slug, "modelVersionSlug": slug, "taskResults": cells}


def boards(results, version=6):
    """Both Benchmarks' rows, showing results ({slug: {task: (value, ±)}})."""
    return {
        board: [row(slug, [cell(t.slug, *by_task[t.slug], version) for t in tasks])
                for slug, by_task in results.items()]
        for board, tasks in BENCHMARKS.items()
    }


def results(value=0.5):
    return {t.slug: (value, 0.01) for t in SCORING_TASKS}


def test_reads_a_row_as_the_api_returns_it():
    # Trimmed from randombench-overall's row for GPT-5.4 on 2026-11-20.
    api_row = {
        "modelVersionName": "GPT-5.4",
        "modelVersionSlug": "gpt-5.4-2026-03-05",
        "taskResults": [
            {"benchmarkTaskName": "(Overall)", "taskVersion": 1,
             "result": {"numericResult": {"value": 0.1528134378126042}}},
            {"benchmarkTaskName": "RandomBench Score",
             "benchmarkTaskSlug": "/benchmarks/tasks/benjaminbrumfield/randombench-score", "taskVersion": 5,
             "result": {"numericResult": {"value": 0.1528134378126042, "confidenceInterval": 0.01903399424886918}}},
        ],
    }
    assert read_rows([api_row]) == {
        "gpt-5.4-2026-03-05": {HEADLINE: (5, (0.1528134378126042, 0.01903399424886918))}
    }


def test_a_score_of_zero_has_no_value_and_a_false_has_no_numeric_result():
    # Protobuf's JSON leaves out a field at its default (4.6).
    zero = cell("randombench-favorite-number", 0.0, 0.0)
    failed = {**cell("randombench-uniformity", 0.1, 0.01), "result": {}}
    assert read_rows([row("m", [zero, failed])]) == {
        "m": {"randombench-favorite-number": (6, (0.0, 0.0)), "randombench-uniformity": (6, None)}
    }


def test_the_benchmarks_hold_every_scoring_task_once():
    assert [t.slug for t in BENCHMARKS["randombench-overall"]] == [HEADLINE]
    assert [t.slug for t in BENCHMARKS["randombench"]] == SUB_SCORES


def test_matching_results_are_up_to_date():
    local = {"a": results(0.5), "b": results(0.25)}
    c = check(boards(local), local, {"a", "b"})
    assert c.matched == 2 * len(SCORING_TASKS)
    assert c.problems == 0
    assert c.versions == {t.slug: {6} for t in SCORING_TASKS}
    assert format_check(c).endswith("Up to date.")


def test_an_old_version_differs():
    local = {"a": results(0.5)}
    c = check(boards({"a": results(0.4)}, version=5), local, {"a"})
    assert c.matched == 0
    assert len(c.differ) == len(SCORING_TASKS)
    assert c.differ[0] == ("randombench-overall", "a", HEADLINE, (0.4, 0.01), (0.5, 0.01))
    assert "randombench: 1 models, 9 tasks at v5" in format_check(c)


def test_a_last_digit_difference_still_matches():
    local = {"a": results(0.5)}
    c = check(boards({"a": results(0.5 + 1e-15)}), local, {"a"})
    assert c.problems == 0


def test_a_false_matches_only_a_score_of_none():
    local = {"a": {**results(0.5), HEADLINE: None}}
    shown = boards({"a": results(0.5)})
    shown["randombench-overall"][0]["taskResults"][0]["result"] = {}
    assert check(shown, local, {"a"}).problems == 0
    assert len(check(boards({"a": results(0.5)}), local, {"a"}).differ) == 1


def test_a_listed_model_off_a_benchmark_is_missing():
    local = {"a": results(), "new": results()}
    c = check(boards({"a": results()}), local, {"a", "new"})
    assert c.missing == [("randombench-overall", "new", ""), ("randombench", "new", "")]
    assert "no row: add the model on the website" in format_check(c)


def test_a_task_with_no_result_is_missing():
    local = {"a": results()}
    shown = boards(local)
    del shown["randombench"][0]["taskResults"][-1]
    assert check(shown, local, {"a"}).missing == [("randombench", "a", SUB_SCORES[-1])]


def test_a_model_kaggle_dropped_is_no_problem():
    # Its draws stay in data/, but no run can score it, so it drops off a
    # Benchmark that moves to the new versions.
    local = {"a": results(), "gone": results()}
    c = check(boards({"a": results()}), local, {"a"})
    assert c.unlisted == ["gone"]
    assert c.problems == 0


def test_a_model_with_no_stored_replies_is_extra():
    local = {"a": results()}
    c = check(boards({"a": results(), "other": results()}), local, {"a", "other"})
    assert c.extra == [("randombench-overall", "other"), ("randombench", "other")]


def test_local_results_key_by_folder_and_skip_a_model_whose_calls_all_failed(tmp_path):
    def store(folder, model, replies):
        (tmp_path / folder).mkdir()
        draws = [{"model": model, "run_id": folder, "draw_index": i, "reply": r} for i, r in enumerate(replies)]
        (tmp_path / folder / "1.jsonl").write_text("".join(json.dumps(d) + "\n" for d in draws), encoding="utf-8")

    store("nano", "openai/gpt-5.4-nano-2026-03-17", ["42", "47", "73", "42"])
    store("unserved", "example/unserved", [None, None])
    local = local_results(tmp_path)
    assert list(local) == ["nano"]
    assert set(local["nano"]) == {t.slug for t in SCORING_TASKS}
    value, half_width = local["nano"][HEADLINE]
    assert 0 < value < 1 and half_width >= 0
