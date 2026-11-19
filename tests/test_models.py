"""Comparing Kaggle's model list with data/ and MODELS, offline.

No test reaches Kaggle: each builds the list the API would return.
"""

import json
from datetime import date

import pytest
from kagglesdk.benchmarks.types.benchmark_types import BenchmarkModel

from randombench.models import Listed, allowed_models, compare, format_report, listed_from_api, stored_models
from randombench.sync import ROOT

NANO = "openai/gpt-5.4-nano-2026-03-17"


def listed(slug, llm_name, proxy=True, name=None, released=None):
    return Listed(slug=slug, name=name or slug, llm_name=llm_name, proxy=proxy, released=released)


def test_reads_an_api_record():
    # Trimmed from the API's reply for GPT-6.2 Sol on 2026-11-18.
    record = {
        "displayName": "GPT-6.2 Sol",
        "slug": "gpt-6.2-sol",
        "version": {
            "slug": "gpt-6.2-sol",
            "isDefault": True,
            "allowModelProxy": True,
            "modelProxySlug": "openai/gpt-6.2-sol",
            "releaseDate": "2026-11-17T00:00:00.000Z",
        },
    }
    m = listed_from_api(BenchmarkModel.from_json(json.dumps(record)))
    assert m == Listed("gpt-6.2-sol", "GPT-6.2 Sol", "openai/gpt-6.2-sol", True, date(2026, 11, 17))


def test_a_missing_field_reads_as_unset():
    # Claude Haiku 5.5 and Gemini 4 Argon have no release date.
    record = {"displayName": "X", "version": {"slug": "x"}}
    m = listed_from_api(BenchmarkModel.from_json(json.dumps(record)))
    assert (m.llm_name, m.proxy, m.released) == ("", False, None)


def test_sorts_each_model_into_one_group():
    gemma = listed("gemma-4-26b-a4b-it", "google/gemma-4-26b-a4b")  # its slug isn't its llm.name
    nano = listed("gpt-5.4-nano-2026-03-17", NANO)
    sol = listed("gpt-6.2-sol", "openai/gpt-6.2-sol")
    default = listed("gemini-3.8-flash", "google/gemini-3.8-flash")
    moved = listed("grok-4.7", "xai/grok-4.7-1120")
    blocked = listed("gpt-6-luna", "", proxy=False)  # Kaggle gives it no llm.name
    stored = {
        "gemma-4-26b-a4b-it": "google/gemma-4-26b-a4b",
        "gpt-5.4-nano-2026-03-17": NANO,
        "gemini-3.8-flash": "google/gemini-3.8-flash",
        "grok-4.7": "xai/grok-4.7",
        "gpt-6-luna": "openai/gpt-6-luna",
        "gemini-2.5-pro": "google/gemini-2.5-pro",
    }
    allowed = ("google/gemma-4-26b-a4b", NANO, "xai/grok-4.7", "openai/gpt-6-luna",
               "google/gemini-2.5-pro", "openai/gpt-5.4-nan0")
    report = compare([gemma, nano, sol, default, moved, blocked], stored, allowed)

    assert report.covered == [gemma, nano, blocked]
    assert report.new == [sol]
    assert report.left_out == [default]
    assert report.renamed == [(moved, "xai/grok-4.7")]
    assert report.not_allowed == [blocked]
    # A stored model Kaggle dropped, and a MODELS entry that names no model.
    assert report.gone == [("gemini-2.5-pro", "google/gemini-2.5-pro"), ("", "openai/gpt-5.4-nan0")]


def test_a_new_model_already_in_models_is_still_new():
    # Added to MODELS, but its pilot hasn't run yet.
    sol = listed("gpt-6.2-sol", "openai/gpt-6.2-sol")
    report = compare([sol], {}, ("openai/gpt-6.2-sol",))
    assert (report.new, report.gone) == ([sol], [])
    assert "already in MODELS" in format_report(report)


def test_the_report_lists_every_group():
    sol = listed("gpt-6.2-sol", "openai/gpt-6.2-sol", name="GPT-6.2 Sol", released=date(2026, 11, 17))
    nano = listed("gpt-5.4-nano-2026-03-17", NANO)
    report = compare([sol, nano], {"gpt-5.4-nano-2026-03-17": NANO, "gemini-2.5-pro": "google/gemini-2.5-pro"},
                     (NANO, "google/gemini-2.5-pro"))
    text = format_report(report)
    assert text.startswith("Kaggle lists 2 models. RandomBench covers 1:")
    assert "  gpt-6.2-sol  GPT-6.2 Sol  openai/gpt-6.2-sol  released 2026-11-17" in text
    assert "  gemini-2.5-pro  google/gemini-2.5-pro  stored draws, in MODELS" in text
    for group in ("New (1)", "Gone (1)", "Left out (0)", "Renamed (0)", "Not allowed (0)"):
        assert group in text
    assert text.count("  none") == 3


def write_run(folder, run_id, *models):
    folder.mkdir(parents=True, exist_ok=True)
    lines = "".join(json.dumps({"model": m, "draw_index": i}) + "\n" for i, m in enumerate(models))
    (folder / f"{run_id}.jsonl").write_text(lines, encoding="utf-8")


def test_reads_each_folders_llm_name(tmp_path):
    write_run(tmp_path / "gemma-4-26b-a4b-it", 1, "google/gemma-4-26b-a4b", "google/gemma-4-26b-a4b")
    (tmp_path / "empty").mkdir()
    (tmp_path / "notes.txt").write_text("not a folder", encoding="utf-8")
    assert stored_models(tmp_path) == {"gemma-4-26b-a4b-it": "google/gemma-4-26b-a4b"}


def test_reads_the_newest_runs_llm_name(tmp_path):
    # After a rename, the newest run holds the name a run gets now. Run ids
    # compare as numbers, so 1000 is newer than 999.
    write_run(tmp_path / "grok-4.7", 999, "xai/grok-4.7")
    write_run(tmp_path / "grok-4.7", 1000, "xai/grok-4.7-1120")
    assert stored_models(tmp_path) == {"grok-4.7": "xai/grok-4.7-1120"}


def test_reads_the_stored_models():
    stored = stored_models(ROOT / "data")
    assert stored["gpt-5.4-nano-2026-03-17"] == NANO
    assert stored["gemma-4-26b-a4b-it"] == "google/gemma-4-26b-a4b"


def test_reads_models_without_running_the_task(tmp_path):
    task = tmp_path / "task.py"
    task.write_text('import kaggle_benchmarks_missing\nMODELS = (\n    "a/b",  # a comment\n    "c/d",\n)\n',
                    encoding="utf-8")
    assert allowed_models(task) == ("a/b", "c/d")


def test_reads_the_collection_tasks_models():
    models = allowed_models()
    assert NANO in models
    assert "google/gemini-3.8-flash" not in models  # D23


def test_a_task_without_models_is_an_error(tmp_path):
    task = tmp_path / "task.py"
    task.write_text("N_DRAWS = 300\n", encoding="utf-8")
    with pytest.raises(ValueError, match="doesn't assign MODELS"):
        allowed_models(task)
