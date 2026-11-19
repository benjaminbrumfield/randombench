"""Counting each model's replies and picking the next run's model, offline.

No test reaches Kaggle: each builds the list the API would return.
"""

import json
from collections import Counter

from randombench.models import Listed, allowed_models, stored_models
from randombench.sync import ROOT
from randombench.tally import format_tally, next_model, reply_counts, tally

NANO = "openai/gpt-5.4-nano-2026-03-17"
SOL = "openai/gpt-6.2-sol"
ASTRA = "openai/gpt-6.2-astra"


def listed(slug, llm_name, proxy=True):
    return Listed(slug=slug, name=slug, llm_name=llm_name, proxy=proxy, released=None)


def test_counts_replies_not_draws(tmp_path):
    folder = tmp_path / "grok-4.5-0708"
    folder.mkdir()
    draws = [{"model": "xai/grok-4.5", "run_id": "1", "draw_index": i, "reply": None} for i in range(3)]
    draws += [{"model": NANO, "run_id": "2", "draw_index": 0, "reply": "42"}]
    (folder / "1.jsonl").write_text("".join(json.dumps(d) + "\n" for d in draws), encoding="utf-8")
    assert reply_counts(tmp_path) == {"xai/grok-4.5": 0, NANO: 1}


def test_the_fewest_replies_go_first_and_ties_go_by_slug():
    models = [listed("gpt-6.2-sol", SOL), listed("gpt-6.2-astra", ASTRA), listed("nano", NANO)]
    stored = {"gpt-6.2-sol": SOL, "gpt-6.2-astra": ASTRA, "nano": NANO}
    rows = tally(models, stored, Counter({SOL: 24, ASTRA: 24, NANO: 624}), (NANO, SOL, ASTRA))
    assert [r.slug for r in rows] == ["gpt-6.2-astra", "gpt-6.2-sol", "nano"]
    assert next_model(rows).slug == "gpt-6.2-astra"


def test_a_new_model_in_models_goes_first():
    # Added to MODELS, with no draws stored yet.
    models = [listed("gpt-6.2-sol", SOL), listed("nano", NANO)]
    rows = tally(models, {"nano": NANO}, Counter({NANO: 624}), (NANO, SOL))
    assert (rows[0].slug, rows[0].replies, rows[0].why_not) == ("gpt-6.2-sol", 0, "")
    assert next_model(rows) is rows[0]


def test_skips_every_model_a_run_cant_draw_from():
    stored = {
        "gemini-3.8-flash": "google/gemini-3.8-flash",  # the server's default (D23)
        "gemini-2.5-pro": "google/gemini-2.5-pro",  # dropped by Kaggle
        "gpt-6-luna": "openai/gpt-6-luna",  # Model Proxy won't call it
        "grok-4.7": "xai/grok-4.7",  # renamed
        "nano": NANO,
    }
    models = [listed("gemini-3.8-flash", "google/gemini-3.8-flash"), listed("gpt-6-luna", "", proxy=False),
              listed("grok-4.7", "xai/grok-4.7-1120"), listed("nano", NANO)]
    replies = Counter({name: 10 for name in stored.values()}) | Counter({NANO: 600})
    allowed = ("google/gemini-2.5-pro", "openai/gpt-6-luna", "xai/grok-4.7", NANO)
    rows = {r.llm_name: r for r in tally(models, stored, replies, allowed)}
    assert rows["google/gemini-3.8-flash"].why_not == "not in MODELS"
    assert (rows["google/gemini-2.5-pro"].why_not, rows["google/gemini-2.5-pro"].slug) == ("Kaggle doesn't list it", "")
    assert rows["openai/gpt-6-luna"].why_not == "Model Proxy doesn't allow it"
    assert rows["xai/grok-4.7"].why_not == "Kaggle renamed it xai/grok-4.7-1120"
    # Nano has the most replies, but it's the only model a run can draw from.
    assert next_model(list(rows.values())).llm_name == NANO


def test_no_model_to_draw_from():
    rows = tally([], {"nano": NANO}, Counter({NANO: 5}), ())
    assert next_model(rows) is None
    assert format_tally(rows).endswith("Next: none")


def test_the_table():
    models = [listed("gpt-6.2-sol", SOL), listed("nano", NANO)]
    rows = tally(models, {"gpt-6.2-sol": SOL, "nano": NANO, "old": "google/gemini-2.5-pro"},
                 Counter({SOL: 24, NANO: 1624, "google/gemini-2.5-pro": 621}),
                 (SOL, NANO, "google/gemini-2.5-pro"))
    assert format_tally(rows).splitlines() == [
        "| # | Model | -m slug | Replies | Next run |",
        "|---:|---|---|---:|---|",
        "| 1 | GPT-6.2 Sol | gpt-6.2-sol | 24 | next |",
        "| 2 | Gemini 2.5 Pro | - | 621 | never: Kaggle doesn't list it |",
        "| 3 | GPT-5.4 nano | nano | 1,624 |  |",
        "",
        "2,269 replies from 3 models, 2 of which a run can draw from.",
        "Next: gpt-6.2-sol",
    ]


def test_tallies_the_stored_draws():
    # Every stored model, with Kaggle's list left empty: nothing can be drawn.
    data = ROOT / "data"
    rows = tally([], stored_models(data), reply_counts(data), allowed_models())
    assert {r.llm_name for r in rows} >= set(allowed_models())
    assert next(r for r in rows if r.llm_name == NANO).replies >= 600
    assert all(r.why_not for r in rows)
