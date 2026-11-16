"""The leaderboard page, docs/index.html (PLAN.md 5.3), checked offline."""

import math
import re
from datetime import datetime, timezone
from html.parser import HTMLParser

import pytest

from randombench.build import DATA, SCORING_TASKS
from randombench.extract import load_draws
from randombench.page import (
    CHECK_NAMES,
    DISPLAY_NAMES,
    LOWEST,
    PAGE,
    _answers_cell,
    _favorite_cell,
    average_shares,
    axis,
    favorites,
    newest,
    ranked,
    render_page,
    shade,
)
from randombench.score import score_draws

DRAWS = load_draws(DATA)
EXPECTED = score_draws(DRAWS)  # what `python -m randombench.score` reports

NANO = "openai/gpt-5.4-nano-2026-03-17"
GEMINI = "google/gemini-3.1-flash-lite-preview"
GROK_45 = "xai/grok-4.5-0708"


class Tables(HTMLParser):
    """Each table's body rows, keyed by the table's class. A row is its
    attributes and its cells, and a cell is its attributes and its text."""

    def __init__(self, text: str):
        super().__init__()
        self.tables = {}
        self.table = self.row = self.cell = None
        self.in_body = False
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "table":
            self.table = self.tables.setdefault(attrs["class"], [])
        elif tag == "tbody":
            self.in_body = True
        elif tag == "tr" and self.in_body:
            self.row = (attrs, [])
            self.table.append(self.row)
        elif tag in ("td", "th") and self.row is not None:
            self.cell = (attrs, [])
            self.row[1].append(self.cell)

    def handle_endtag(self, tag):
        if tag == "tbody":
            self.in_body, self.row = False, None
        elif tag in ("td", "th"):
            self.cell = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell[1].append(data)

    def rows(self, table: str) -> list[tuple[dict, list[tuple[dict, str]]]]:
        return [(attrs, [(a, "".join(parts)) for a, parts in cells]) for attrs, cells in self.tables[table]]


def shown(score, half) -> str:
    """A score as the page shows it."""
    return "—" if score is None else f"{score:.3f} ±{half:.3f}"


# The names and the scales


def test_every_model_in_data_has_a_display_name():
    assert {d["model"] for d in DRAWS} <= set(DISPLAY_NAMES)


def test_the_checks_are_named_as_their_scoring_tasks():
    assert [(key, f"RandomBench {name}") for key, name in CHECK_NAMES.items()] == [
        (t.score, t.title) for t in SCORING_TASKS[1:]
    ]


def test_the_lowest_scores_are_the_bias_tables():
    # PLAN.md's "Lowest" column, derived from the checks themselves.
    assert LOWEST == {
        "invalid_answers": 0, "uniformity": 0.01, "primes": 0.25, "multiples_of_10": 0.10,
        "multiples_of_5": 0.20, "favorite": 0, "edges": 0.20, "repeated_digits": 0.09, "last_digit": 0.10,
    }


def test_shading_runs_from_the_lowest_score_to_1():
    assert shade("primes", 0.25) == 0
    assert shade("primes", 0.625) == pytest.approx(0.5)
    assert shade("primes", 1.0) == 1
    assert shade("favorite", None) == 0


def test_uniformity_and_last_digit_are_shaded_on_a_log_scale():
    # A fair pick among 1, 10, or 100 values (D28).
    assert [shade("uniformity", k / 100) for k in (1, 10, 100)] == pytest.approx([0, 0.5, 1])
    assert shade("last_digit", math.sqrt(10) / 10) == pytest.approx(0.5)


def test_the_average_share_counts_each_model_the_same():
    few, many, none = [0] * 100, [0] * 100, [0] * 100
    few[42], many[47] = 2, 6
    shares = average_shares([few, many, none])  # a model with no valid answers is left out
    assert shares[42] == shares[47] == 0.5  # added up instead, 47 would weigh 3 times 42
    assert sum(shares) == pytest.approx(1)
    assert average_shares([none]) == [0.0] * 100


def test_the_axis_is_the_smallest_round_one_that_holds_every_interval():
    assert axis(0.119) == (0.04, 3)  # 0 to 0.12
    assert axis(0.12) == (0.04, 3)
    assert axis(0.121) == (0.05, 3)
    assert axis(1.0) == (0.25, 4)


# The committed page


def test_the_page_is_up_to_date():
    assert PAGE.read_text(encoding="utf-8") == render_page(EXPECTED, newest(DRAWS)), (
        "out of date: run `python -m randombench.build`"
    )


def test_every_number_on_the_page_is_the_scorers():
    tables = Tables(PAGE.read_text(encoding="utf-8"))
    by_model = {r.model: r for r in EXPECTED}
    order = [r.model for r in ranked(EXPECTED)]
    # Every model with a reply. Grok 4.5 has none (D24).
    assert len(order) == 36 and GROK_45 not in order

    ranking = tables.rows("rank")
    assert [attrs["data-model"] for attrs, _ in ranking] == order
    for rank, (attrs, cells) in enumerate(ranking, 1):
        r = by_model[attrs["data-model"]]
        name, provider = DISPLAY_NAMES[r.model]
        texts = [text for _, text in cells]
        assert texts[:4] == [str(rank), f"{name} {provider}", str(r.draws - r.failed),
                             shown(r.scores["headline"], r.intervals["headline"])]

    biases = tables.rows("biases")
    assert [attrs["data-model"] for attrs, _ in biases] == order
    for attrs, cells in biases:
        r = by_model[attrs["data-model"]]
        values, top = favorites(r.counts)
        assert cells[1][1] == f"{', '.join(map(str, values))} · {top:.0%}"
        assert values == [v for v, c in enumerate(r.counts) if c == max(r.counts)]
        assert [text for _, text in cells[2:]] == [shown(r.scores[k], r.intervals[k]) for k in CHECK_NAMES]
        # What the sort reads.
        assert [float(a["data-v"]) for a, _ in cells[2:]] == pytest.approx([r.scores[k] for k in CHECK_NAMES], abs=1e-6)


def test_the_summary_cards_are_the_scorers():
    text = PAGE.read_text(encoding="utf-8")
    models = ranked(EXPECTED)
    shares = average_shares([r.counts for r in models])
    assert re.findall(r'<dd class="value[^"]*">([^<]*)</dd>', text) == [
        str(len(models)),
        f"{sum(sum(r.counts) for r in models):,}",
        f'{models[0].scores["headline"]:.3f}',
        str(shares.index(max(shares))),
    ]


def test_the_headline_is_the_ranking():
    scores = [by.scores["headline"] for by in ranked(EXPECTED)]
    assert scores == sorted(scores, reverse=True)


def test_the_page_loads_nothing_from_other_sites():
    # Pages serves the one file, and it should load at once (5.3).
    text = PAGE.read_text(encoding="utf-8")
    for loader in ["<script src", 'rel="stylesheet"', "@import", "url(", "<img", "<iframe"]:
        assert loader not in text
    hrefs = [part.split('"', 1)[0] for part in text.split('href="')[1:]]
    allowed = ("https://github.com/benjaminbrumfield/randombench", "https://www.kaggle.com/benchmarks/benjaminbrumfield/")
    assert all(h == "data:," or h.startswith(allowed) for h in hrefs), hrefs


# Rendering other data


def test_a_model_with_no_replies_is_left_out_and_one_with_no_headline_goes_last():
    draws = [
        {"model": NANO, "reply": r} for r in ["42", "42", "37"]
    ] + [
        {"model": GEMINI, "reply": "forty-two"},  # one reply, invalid, so no headline
        {"model": GROK_45, "reply": None},  # a failed call
    ]
    tables = Tables(render_page(score_draws(draws), datetime(2026, 1, 2, tzinfo=timezone.utc)))
    ranking = tables.rows("rank")
    assert [attrs["data-model"] for attrs, _ in ranking] == [NANO, GEMINI]
    assert [text for _, text in ranking[1][1]][3] == "—"
    gemini = [text for _, text in tables.rows("biases")[1][1][1:]]
    assert gemini == ["—", "0.000 ±0.000"] + ["—"] * 8  # no favorite without a valid answer


def test_a_tie_for_favorite_shows_every_tied_value():
    counts = [0] * 100
    counts[42], counts[73], counts[7] = 3, 3, 1
    assert favorites(counts) == ([42, 73], 3 / 7)
    assert favorites([0] * 100) is None
    assert ">42, 73 <" in _favorite_cell(counts)


def test_the_answers_chart_draws_each_value_given_with_the_favorites_apart():
    counts = [0] * 100
    counts[42], counts[73], counts[7], counts[99] = 20, 20, 1, 5
    cell = _answers_cell(counts)
    paths = re.findall(r'<path( class="fav")? d="([^"]*)"', cell)
    bars = {bool(fav): re.findall(r"M(\d+) 20v-([0-9.]+)", d) for fav, d in paths}
    assert bars[True] == [("42", "20"), ("73", "20")]
    assert bars[False] == [("7", "2"), ("99", "5")]  # one answer is under the floor
    assert 'aria-label="Favorite 42, 73: 43% of its valid answers"' in cell
    assert _answers_cell([0] * 100) == '<td class="answers"></td>'  # no valid answers


def test_sorting_the_favorites_groups_them_in_ascending_order_largest_share_first():
    def key(favorite, count):
        counts = [0] * 100
        counts[favorite], counts[99] = count, 10 - count
        return float(re.search(r'data-v="([-0-9.]+)"', _favorite_cell(counts))[1])
    # The table sorts from the highest data-v down.
    assert sorted([(47, 9), (42, 6), (73, 8), (42, 9)], key=lambda f: -key(*f)) == [(42, 9), (42, 6), (47, 9), (73, 8)]


def test_the_page_is_dated_by_the_newest_draw():
    draws = [{"started_at": "2026-01-02T23:59:00Z"}, {"started_at": None}, {"started_at": "2025-12-31T00:00:00Z"}]
    assert newest(draws) == datetime(2026, 1, 2, 23, 59, tzinfo=timezone.utc)
    text = render_page(EXPECTED, newest(draws))
    assert 'Data as of <time datetime="2026-01-02">January 2, 2026</time>' in text
