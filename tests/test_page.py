"""The leaderboard page, docs/index.html (PLAN.md 5.3), checked offline."""

import json
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
    HEAT_DARK,
    HEAT_FLOOR,
    LOWEST,
    PAGE,
    _answers_cell,
    _distribution,
    _panel,
    _series,
    average_shares,
    axis,
    favorites,
    newest,
    panel_data,
    ranked,
    render_page,
    shade,
    tallest,
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


def pair(score, half) -> list[str]:
    """A score as the panel's data holds it."""
    return ["—", ""] if score is None else [f"{score:.3f}", f"±{half:.3f}"]


def panel_json(text: str) -> dict:
    """Every model's panel data, from the page's JSON."""
    raw = re.search(r'<script type="application/json" id="panel-data">(.*?)</script>', text, re.S)[1]
    assert "<" not in raw  # so nothing in it can end the script early
    return json.loads(raw)


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
    assert len(order) == 39 and GROK_45 not in order

    ranking = tables.rows("rank")
    assert [attrs["data-model"] for attrs, _ in ranking] == order
    for rank, (attrs, cells) in enumerate(ranking, 1):
        r = by_model[attrs["data-model"]]
        name, provider = DISPLAY_NAMES[r.model]
        texts = [text for _, text in cells]
        assert texts[:4] == [str(rank), f"{name} {provider}", str(r.draws - r.failed),
                             shown(r.scores["headline"], r.intervals["headline"])]
    # Each model's checks are in its panel data, which the next tests check.


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


def test_the_answers_chart_labels_its_five_tallest_bars():
    text = PAGE.read_text(encoding="utf-8")
    shares = average_shares([r.counts for r in ranked(EXPECTED)])
    by_share = sorted(range(100), key=lambda v: -shares[v])
    labels = re.findall(r'class="(label(?: minor)?)"[^>]*>(\d+)<small>([^<]*)</small>', text)
    assert {int(v) for kind, v, _ in labels if kind == "label"} == set(by_share[:3])
    assert {int(v) for kind, v, _ in labels if kind == "label minor"} == set(by_share[3:5])
    assert all(share == f"{shares[int(v)]:.1%}" for _, v, share in labels)


def test_the_panel_is_the_top_models():
    text = PAGE.read_text(encoding="utf-8")
    panel = text[text.index('<aside class="panel"'):text.index("</aside>")]
    r = ranked(EXPECTED)[0]
    name, provider = DISPLAY_NAMES[r.model]
    assert re.search(r"<h3[^>]*>(.*?)</h3>", panel)[1] == name and f"· {provider} ·" in panel
    big = re.sub(r"<[^>]+>", "", re.search(r'<p class="big">(.*?)</p>', panel)[1])
    assert big == shown(r.scores["headline"], r.intervals["headline"])
    checks = [re.sub(r"<[^>]+>", "", v) for v in re.findall(r'<dd class="num">(.*?)</dd>', panel)]
    assert checks == [shown(r.scores[k], r.intervals[k]) for k in CHECK_NAMES]


def test_every_models_panel_data_is_the_scorers():
    text = PAGE.read_text(encoding="utf-8")
    data = panel_json(text)
    assert data["dark"] == HEAT_DARK
    models = ranked(EXPECTED)
    assert [m["model"] for m in data["models"]] == [r.model for r in models]
    for rank, (m, r) in enumerate(zip(data["models"], models), 1):
        name, provider = DISPLAY_NAMES[r.model]
        assert m["name"] == name
        assert m["about"] == f"Rank {rank} of {len(models)} · {provider} · {r.draws - r.failed} replies"
        assert m["score"] == pair(r.scores["headline"], r.intervals["headline"])
        assert [c[:2] for c in m["checks"]] == [pair(r.scores[k], r.intervals[k]) for k in CHECK_NAMES]
        assert [c[2] for c in m["checks"]] == pytest.approx([shade(k, r.scores[k]) for k in CHECK_NAMES], abs=5e-4)
        assert m["fav"] == favorites(r.counts)[0]
        top = max(r.counts)
        assert m["t"] == pytest.approx([max(HEAT_FLOOR, math.sqrt(c / top)) if c else 0 for c in r.counts], abs=5e-4)
    # The panel the build writes is the first model's, as the script would show it.
    assert _panel(data["models"][0]) in text


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


def test_every_link_opens_in_a_new_tab():
    # Every link leaves the page, for GitHub or Kaggle, so the leaderboard stays open.
    links = re.findall(r"<a [^>]*>", PAGE.read_text(encoding="utf-8"))
    assert links and all('target="_blank"' in link for link in links), links


# Rendering other data


def test_a_model_with_no_replies_is_left_out_and_one_with_no_headline_goes_last():
    draws = [
        {"model": NANO, "reply": r} for r in ["42", "42", "37"]
    ] + [
        {"model": GEMINI, "reply": "forty-two"},  # one reply, invalid, so no headline
        {"model": GROK_45, "reply": None},  # a failed call
    ]
    text = render_page(score_draws(draws), datetime(2026, 1, 2, tzinfo=timezone.utc))
    ranking = Tables(text).rows("rank")
    assert [attrs["data-model"] for attrs, _ in ranking] == [NANO, GEMINI]
    assert [text for _, text in ranking[1][1]][3] == "—"
    gemini = panel_json(text)["models"][1]
    assert gemini["score"] == ["—", ""]
    assert [c[:2] for c in gemini["checks"]] == [["0.000", "±0.000"]] + [["—", ""]] * 8
    assert (gemini["fav"], gemini["key"]) == ([], "No valid answers yet.")  # no favorite without a valid answer
    assert gemini["t"] == [0] * 100


def test_a_tie_for_favorite_shows_every_tied_value():
    counts = [0] * 100
    counts[42], counts[73], counts[7] = 3, 3, 1
    assert favorites(counts) == ([42, 73], 3 / 7)
    assert favorites([0] * 100) is None
    assert 'aria-label="Favorite 42, 73: 43% of its valid answers"' in _answers_cell(counts)


def test_the_answers_chart_draws_each_value_given_on_a_square_root_scale_with_the_favorites_apart():
    counts = [0] * 100
    counts[42], counts[73], counts[7], counts[99], counts[3] = 200, 200, 1, 50, 18
    cell = _answers_cell(counts)
    paths = re.findall(r'<path( class="fav")? d="([^"]*)"', cell)
    bars = {bool(fav): re.findall(r"M(\d+) 100v-(\d+)h1V100z", d) for fav, d in paths}
    assert bars[True] == [("42", "100"), ("73", "100")]
    # A quarter of the favorite's count is half as tall, 18 of 200 is 30, and
    # one answer, 7 by the square root, is raised to the floor of 8.
    assert bars[False] == [("3", "30"), ("7", "8"), ("99", "50")]
    assert 'aria-label="Favorite 42, 73: 43% of its valid answers"' in cell
    assert _answers_cell([0] * 100) == '<td class="answers"></td>'  # no valid answers


def test_the_distribution_labels_its_tallest_bars_and_circles_values_never_given():
    shares = [0.0] * 100
    for value, share in {42: 0.4, 47: 0.2, 73: 0.15, 37: 0.1, 57: 0.05, 7: 0.05, 9: 0.05}.items():
        shares[value] = share
    chart = _distribution(shares)
    orange = re.search(r'<path class="fav" d="([^"]*)"', chart)[1]
    assert orange == "M42.1 100v-100h0.8V100zM47.1 100v-50h0.8V100zM73.1 100v-37.5h0.8V100z"
    assert re.findall(r"M(\d+)\.1", re.search(r'<path d="([^"]*)"', chart)[1]) == ["7", "9", "37", "57"]
    # 7 and 57 tie for fifth, and the lower value gets the label.
    labels = re.findall(r'class="(label(?: minor)?)"[^>]*>(\d+)<small>([^<]*)', chart)
    assert labels == [("label minor", "7", "5.0%"), ("label minor", "37", "10.0%"), ("label", "42", "40.0%"),
                      ("label", "47", "20.0%"), ("label", "73", "15.0%")]
    assert "The tallest are 42 (40.0%), 47 (20.0%), 73 (15.0%), 37 (10.0%), and 7 (5.0%)." in chart
    assert chart.count("<i ") == 93  # one circle for each value no model gave
    assert tallest([0.5, 0.5] + [0.0] * 98) == [0, 1]  # only values given get a label
    assert _distribution([0.0] * 100) == ""


def test_the_panel_shades_each_square_by_the_square_root_of_its_count_over_the_favorites():
    replies = ["42"] * 64 + ["37"] * 36 + ["47"] * 16 + ["7"]
    [r] = score_draws([{"model": NANO, "reply": x} for x in replies])
    d = panel_data(r, 1, 1)
    assert (d["fav"], d["t"][42], d["t"][37], d["t"][47], d["t"][0]) == ([42], 1, 0.75, 0.5, 0)
    assert '<i class="swatch"></i>Favorite <strong>42</strong>: 55% of its valid answers.' == d["key"]
    found = re.findall(r'<span(?: class="(\w+)")?(?: style="--t:([0-9.]+)")?>(\d+)</span>', _panel(d))
    squares = {int(value): (cls, t) for cls, t, value in found}
    assert len(squares) == 100
    assert squares[42] == ("fav", "")
    assert squares[37] == ("hi", "0.750")  # dark enough for --on-deep text
    assert squares[47] == ("", "0.500")  # a quarter of the favorite's count is half as dark
    assert squares[7] == ("", f"{HEAT_FLOOR:.3f}")  # 1 of 64 is 0.125 by the square root, raised to the floor
    assert squares[0] == ("none", "")


def test_a_series_reads_as_a_sentence():
    assert [_series(["a"]), _series(["a", "b"]), _series(["a", "b", "c"])] == ["a", "a and b", "a, b, and c"]


def test_the_page_is_dated_by_the_newest_draw():
    draws = [{"started_at": "2026-01-02T23:59:00Z"}, {"started_at": None}, {"started_at": "2025-12-31T00:00:00Z"}]
    assert newest(draws) == datetime(2026, 1, 2, 23, 59, tzinfo=timezone.utc)
    text = render_page(EXPECTED, newest(draws))
    assert 'Data as of <time datetime="2026-01-02">January 2, 2026</time>' in text
