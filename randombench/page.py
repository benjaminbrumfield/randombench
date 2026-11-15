"""Write the leaderboard page, docs/index.html (PLAN.md 5.3, D26, D27).

GitHub Pages serves it from main's docs/ folder. It shows the scores that
`python -m randombench.score` prints, with models in rows, which Kaggle's
Benchmarks can't show, and with each score's 95% interval, which they don't
show (4.13, D25). The Benchmarks stay the official leaderboard.

The page has two tables (D27). Ranking orders the models by the headline and
draws each one's interval on a shared scale. Biases holds each model's
favorite answer (D30), then the nine sub-scores for the same models in the
same order, each cell shaded by its score (D28). The template, page.html
beside this file, holds the page's text and style, and this fills in the
numbers. Below the tables, the findings are written by hand in the template,
and tests/test_findings.py checks each one against data/ (D29).

`python -m randombench.build` writes the page with the scoring tasks, and a
test fails when it's stale (D26). It's dated by the newest stored draw, not
by the build, so rebuilding unchanged data writes the same bytes.
"""

import math
import re
from datetime import datetime
from html import escape
from pathlib import Path

from randombench.checks import CHECKS, N_VALUES, counts_of
from randombench.score import ModelScores
from randombench.sync import ROOT

TEMPLATE = Path(__file__).with_name("page.html")
PAGE = ROOT / "docs" / "index.html"

PLACEHOLDERS = {"__DATE__", "__ISODATE__", "__COUNT__", "__AXIS__", "__STEPS__", "__TICKS__",
                "__RANKING__", "__HEADINGS__", "__BIASES__"}
_PLACEHOLDER = re.compile(r"__[A-Z]+__")

# Each model's name and provider, as the README names them. A test fails when
# a model in data/ has no entry.
DISPLAY_NAMES = {
    "anthropic/claude-haiku-4-5@20251001": ("Claude Haiku 4.5", "Anthropic"),
    "anthropic/claude-haiku-5-5": ("Claude Haiku 5.5", "Anthropic"),
    "anthropic/claude-sonnet-4-6@default": ("Claude Sonnet 4.6", "Anthropic"),
    "anthropic/claude-sonnet-5@default": ("Claude Sonnet 5", "Anthropic"),
    "anthropic/claude-sonnet-5-5@default": ("Claude Sonnet 5.5", "Anthropic"),
    "google/gemini-2.5-flash": ("Gemini 2.5 Flash", "Google"),
    "google/gemini-2.5-pro": ("Gemini 2.5 Pro", "Google"),
    "google/gemini-3-flash-preview": ("Gemini 3 Flash (preview)", "Google"),
    "google/gemini-3.1-flash-lite-preview": ("Gemini 3.1 Flash-Lite (preview)", "Google"),
    "google/gemini-3.1-pro-preview": ("Gemini 3.1 Pro (preview)", "Google"),
    "google/gemini-3.5-flash-lite": ("Gemini 3.5 Flash-Lite", "Google"),
    "google/gemini-3.6-flash": ("Gemini 3.6 Flash", "Google"),
    "google/gemini-3.8-flash": ("Gemini 3.8 Flash", "Google"),
    "google/gemini-4-argon": ("Gemini 4 Argon", "Google"),
    "google/gemma-4-26b-a4b": ("Gemma 4 26B A4B", "Google"),
    "openai/gpt-5.4-mini-2026-03-17": ("GPT-5.4 mini", "OpenAI"),
    "openai/gpt-5.4-nano-2026-03-17": ("GPT-5.4 nano", "OpenAI"),
    "openai/gpt-5.6-luna": ("GPT-5.6 Luna", "OpenAI"),
    "openai/gpt-5.6-terra": ("GPT-5.6 Terra", "OpenAI"),
    "openai/gpt-6-luna": ("GPT-6 Luna", "OpenAI"),
    "openai/gpt-oss-120b": ("gpt-oss-120b", "OpenAI"),
    "xai/grok-4.5-0708": ("Grok 4.5", "xAI"),  # no replies, so not on the page (D24)
    "xai/grok-4.6": ("Grok 4.6", "xAI"),
    "xai/grok-4.7": ("Grok 4.7", "xAI"),
}

# The nine sub-scores in score_answers' order, named as their scoring tasks
# are, without "RandomBench " (a test checks it).
CHECK_NAMES = {
    "invalid_answers": "Valid Answers",
    "uniformity": "Uniformity",
    "primes": "Primes",
    "multiples_of_10": "Multiples of 10",
    "multiples_of_5": "Multiples of 5",
    "favorite": "Favorite Number",
    "edges": "Edges",
    "repeated_digits": "Repeated Digits",
    "last_digit": "Last Digit",
}

# Each sub-score's lowest possible score, the "Lowest" column of PLAN.md's bias
# table. Every reply invalid scores 0 on Valid Answers. On every other check,
# the lowest is what a model scores when it always gives the same answer and
# that answer is the worst one, such as always a prime. The rounding drops
# floating-point noise.
LOWEST = {"invalid_answers": 0.0} | {
    name: round(min(float(check(counts_of([v, v]))) for v in range(N_VALUES)), 9)
    for name, check in CHECKS.items()
}

# These two score an effective number of values K, out of 100 or 10, so their
# cells are shaded on a log scale (D28): one value is unshaded, 10 of 100 is
# halfway, and every value is full. log K is the collision entropy, so the
# shade is the share of the most entropy possible.
LOG_SHADED = {"uniformity", "last_digit"}


def shade(name: str, score: float | None) -> float:
    """How strongly a Biases cell is shaded: 0 at its check's lowest score, 1 at a score of 1."""
    if score is None:
        return 0.0
    low = LOWEST[name]
    if name in LOG_SHADED:
        fraction = math.log(max(score, low) / low) / math.log(1 / low)
    else:
        fraction = (score - low) / (1 - low)
    return min(1.0, max(0.0, fraction))


def favorites(counts) -> tuple[list[int], float] | None:
    """The model's most common valid answer, or every one tied for most
    common, and the share of the valid answers each takes. None without any
    valid answers."""
    total, top = sum(counts), max(counts)
    if not total:
        return None
    return [value for value, count in enumerate(counts) if count == top], top / total


def axis(top: float) -> tuple[float, int]:
    """The Ranking scale: a round step, and how many steps reach top.

    Of the scales with at most 4 steps, it's the one that ends soonest, so the
    bars use as much of the width as they can. On a tie, the more steps the
    better.
    """
    scales = []
    for exponent in range(-4, 2):
        for mantissa in (1, 2, 2.5, 4, 5):
            step = mantissa * 10**exponent
            steps = max(1, math.ceil(top / step - 1e-9))
            if steps <= 4:
                scales.append((round(step * steps, 9), -steps, step))
    end, minus_steps, step = min(scales)
    return step, -minus_steps


def newest(draws: list[dict]) -> datetime:
    """When the newest stored draw started, which dates the page."""
    return max(datetime.fromisoformat(d["started_at"]) for d in draws if d.get("started_at"))


def _decimals(step: float) -> int:
    """How many decimals the step has, such as 2 for 0.04 and 3 for 0.025."""
    return len(f"{step:.6f}".rstrip("0").partition(".")[2])


def _short(value: float, decimals: int) -> str:
    """An axis label or a lowest score: 0 alone, anything else with its decimals."""
    return "0" if value == 0 else f"{value:.{decimals}f}"


def _number(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _plus_minus(half: float | None) -> str:
    return "" if half is None else f' <span class="pm">±{half:.3f}</span>'


def _favorite_cell(counts) -> str:
    """The Biases table's Favorite cell, such as 73 · 18%. It isn't a score, so it isn't shaded.

    The table sorts each column from the highest data-v down. Here that has to
    put the favorites in ascending order, the 42s before the 47s, and within
    each, the largest share first. A share is at most 1, so 100 minus the
    favorite, plus the share, does both.
    """
    found = favorites(counts)
    if found is None:
        return '<td class="num fav" data-v="-1">—</td>'
    values, share = found
    return (f'<td class="num fav" data-v="{N_VALUES - values[0] + share:.6f}">'
            f'{", ".join(str(v) for v in values)} <span class="pm">· {share:.0%}</span></td>')


def _model_cell(tag: str, model: str) -> str:
    name, provider = DISPLAY_NAMES[model]
    scope = ' scope="row"' if tag == "th" else ""
    return (f'<{tag}{scope} class="model"><span class="name">{escape(name)}</span>'
            f' <span class="prov">{escape(provider)}</span></{tag}>')


def ranked(results: list[ModelScores]) -> list[ModelScores]:
    """The models with at least one reply, highest headline first, and a headline of None last."""
    answered = [r for r in results if r.failed < r.draws]
    return sorted(answered, key=lambda r: (r.scores["headline"] is None, -(r.scores["headline"] or 0),
                                           DISPLAY_NAMES[r.model][0]))


def render_page(results: list[ModelScores], as_of: datetime) -> str:
    """Return the page's text for the models' scores, dated as_of."""
    models = ranked(results)
    tops = [r.scores["headline"] + r.intervals["headline"] for r in models if r.scores["headline"] is not None]
    step, steps = axis(max(tops, default=1.0))
    top = step * steps
    decimals = _decimals(step)

    def position(value: float) -> str:
        return f"{min(1.0, max(0.0, value / top)) * 100:.2f}%"

    ranking = []
    for rank, r in enumerate(models, 1):
        score, half = r.scores["headline"], r.intervals["headline"]
        bar = ""
        if score is not None:
            bar = (f'<div class="strip" style="--lo:{position(score - half)};--hi:{position(score + half)};'
                   f'--x:{position(score)}"><i class="ci"></i><b class="dot"></b></div>')
        ranking.append(
            f'<tr data-model="{escape(r.model)}"><td class="n">{rank}</td>{_model_cell("td", r.model)}'
            f'<td class="num replies">{r.draws - r.failed}</td>'
            f'<td class="num score">{_number(score)}{_plus_minus(half)}</td>'
            f'<td class="plot" aria-hidden="true">{bar}</td></tr>'
        )

    headings = '<th scope="col" class="num"><span class="h">Favorite</span><span class="low">and its share</span></th>'
    headings += "".join(
        f'<th scope="col" class="num"><span class="h">{name}</span>'
        f'<span class="low">lowest {_short(LOWEST[key], 2)}</span></th>'
        for key, name in CHECK_NAMES.items()
    )

    biases = []
    for rank, r in enumerate(models, 1):
        cells = [_favorite_cell(r.counts)]
        for key in CHECK_NAMES:
            score, half = r.scores[key], r.intervals[key]
            if score is None:
                cells.append('<td class="num cell" data-v="-1">—</td>')
            else:
                cells.append(f'<td class="num cell" data-v="{score:.6f}" style="--f:{shade(key, score):.3f}">'
                             f'{_number(score)}{_plus_minus(half)}</td>')
        biases.append(f'<tr data-model="{escape(r.model)}" data-rank="{rank}">'
                      f'{_model_cell("th", r.model)}{"".join(cells)}</tr>')

    ticks = "".join(f'<span class="tick" style="--x:{i / steps * 100:.2f}%">{_short(i * step, decimals)}</span>'
                    for i in range(steps + 1))

    text = TEMPLATE.read_text(encoding="utf-8")
    found = set(_PLACEHOLDER.findall(text))
    if found != PLACEHOLDERS:
        raise ValueError(f"the page template's placeholders are {sorted(found)}, not {sorted(PLACEHOLDERS)}")
    values = {
        "__DATE__": f"{as_of:%B} {as_of.day}, {as_of.year}",
        "__ISODATE__": f"{as_of:%Y-%m-%d}",
        "__COUNT__": str(len(models)),
        "__AXIS__": f"{top:.{decimals}f}",
        "__STEPS__": str(steps),
        "__TICKS__": ticks,
        "__HEADINGS__": headings,
        "__RANKING__": "\n".join(ranking),
        "__BIASES__": "\n".join(biases),
    }
    return _PLACEHOLDER.sub(lambda m: values[m.group()], text)
