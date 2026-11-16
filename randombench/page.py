"""Write the leaderboard page, docs/index.html (PLAN.md 5.3, D26).

GitHub Pages serves it from main's docs/ folder. It shows the scores that
`python -m randombench.score` prints, with models in rows, which Kaggle's
Benchmarks can't show, and with each score's 95% interval, which they don't
show (4.13, D25). The Benchmarks stay the official leaderboard.

Four cards open the page: how many models and valid answers there are, the
top score, and the most common answer, by its share averaged over the models.
This computes all four, so none is a claim written by hand. Below them, "What
the models answer" charts each value's share, averaged the same way, with
every model's favorite in orange.

Ranking orders the models by the headline, charts each one's answers from 0
to 99, with its favorite in orange, and draws each one's interval on a shared
scale. Beside it, a panel shows one model's answers on a 10 by 10 grid and its
nine checks as bars, each filled from its check's lowest score to 1, as D28
shaded the cells of the table it replaces. The build writes the
top model's panel, so the page reads without JavaScript, and every model's
panel data as JSON, which the page's script puts in the panel when a row is
selected.

The page used to have two tables (D27): Ranking, and Biases, which held every
model's nine checks and, from D30, its favorite, in a column that sorted. The
panel and the Answers column show both, so Biases is gone. What went with it
is comparing every model on one check at a glance; the panel shows one model
at a time.

The template, page.html beside this file, holds the page's text and style,
and this fills in the numbers. Below the Ranking, the findings are written by
hand in the template, and tests/test_findings.py checks each one against
data/ (D29).

`python -m randombench.build` writes the page with the scoring tasks, and a
test fails when it's stale (D26). It's dated by the newest stored draw, not
by the build, so rebuilding unchanged data writes the same bytes.
"""

import json
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

PLACEHOLDERS = {"__DATE__", "__ISODATE__", "__COUNT__", "__PROVIDERS__", "__ANSWERS__", "__BEST__",
                "__BESTMODEL__", "__COMMON__", "__COMMONSHARE__", "__DIST__", "__FAVORITES__",
                "__NEVER__", "__PANEL__", "__PANELS__", "__STEPS__", "__TICKS__",
                "__RANKING__"}
_PLACEHOLDER = re.compile(r"__[A-Z]+__")

# Each model's name and provider, as the README names them. A test fails when
# a model in data/ has no entry.
DISPLAY_NAMES = {
    "anthropic/claude-haiku-4-5@20251001": ("Claude Haiku 4.5", "Anthropic"),
    "anthropic/claude-haiku-5-5": ("Claude Haiku 5.5", "Anthropic"),
    "anthropic/claude-opus-4-5@20251101": ("Claude Opus 4.5", "Anthropic"),
    "anthropic/claude-opus-4-6@default": ("Claude Opus 4.6", "Anthropic"),
    "anthropic/claude-opus-4-7@default": ("Claude Opus 4.7", "Anthropic"),
    "anthropic/claude-opus-4-8@default": ("Claude Opus 4.8", "Anthropic"),
    "anthropic/claude-opus-5@default": ("Claude Opus 5", "Anthropic"),
    "anthropic/claude-opus-5-5@default": ("Claude Opus 5.5", "Anthropic"),
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
    "google/gemma-4-31b": ("Gemma 4 31B", "Google"),
    "openai/gpt-5.4-2026-03-05": ("GPT-5.4", "OpenAI"),
    "openai/gpt-5.4-mini-2026-03-17": ("GPT-5.4 mini", "OpenAI"),
    "openai/gpt-5.4-nano-2026-03-17": ("GPT-5.4 nano", "OpenAI"),
    "openai/gpt-5.5-2026-04-23": ("GPT-5.5", "OpenAI"),
    "openai/gpt-5.6-luna": ("GPT-5.6 Luna", "OpenAI"),
    "openai/gpt-5.6-sol": ("GPT-5.6 Sol", "OpenAI"),
    "openai/gpt-5.6-terra": ("GPT-5.6 Terra", "OpenAI"),
    "openai/gpt-6-astra": ("GPT-6 Astra", "OpenAI"),
    "openai/gpt-6-luna": ("GPT-6 Luna", "OpenAI"),
    "openai/gpt-6-sol": ("GPT-6 Sol", "OpenAI"),
    "openai/gpt-6.1-sol": ("GPT-6.1 Sol", "OpenAI"),
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
# bars are filled on a log scale (D28): one value is empty, 10 of 100 is
# halfway, and every value is full. log K is the collision entropy, so the
# fill is the share of the most entropy possible.
LOG_SHADED = {"uniformity", "last_digit"}


def shade(name: str, score: float | None) -> float:
    """How full a check's bar in the panel is: 0 at its check's lowest score, 1 at a score of 1."""
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


def average_shares(all_counts) -> list[float]:
    """Each value's share of a model's valid answers, averaged over the models that have any.

    Each model counts the same, however many replies it has. Adding up every
    model's answers instead would weigh a model with 624 replies nearly twice
    as much as one with 324, and how many replies a model has depends on how
    many runs it got, not on how it answers.
    """
    answered = [counts for counts in all_counts if sum(counts)]
    if not answered:
        return [0.0] * N_VALUES
    return [sum(counts[v] / sum(counts) for counts in answered) / len(answered) for v in range(N_VALUES)]


def _series(items: list[str]) -> str:
    """Items joined as in a sentence, with a serial comma: a, b, and c."""
    if len(items) < 3:
        return " and ".join(items)
    return ", ".join(items[:-1]) + ", and " + items[-1]


# The shortest a bar in "What the models answer" can be, out of 100, so that a
# value given once still shows.
MIN_DIST_BAR = 1


def _distribution(shares: list[float], favs: set[int]) -> str:
    """The chart under "What the models answer": a bar for each value's average share.

    The bars are one SVG, 100 units wide, that stretches to the page's width.
    The labels, the circles under the values no model gave, and the axis are
    HTML placed by percentage, so that they keep their shape as it stretches.
    Each value in favs, some model's favorite, is drawn and labeled in orange.
    """
    top = max(shares)
    if not top:
        return ""
    others, fav_bars, labels, never = [], [], [], []
    for value, share in enumerate(shares):
        x = f"--x:{value + 0.5:g}%"
        if not share:
            never.append(f'<i style="{x}"></i>')
            continue
        height = round(max(MIN_DIST_BAR, 100 * share / top), 2)
        bar = f"M{value + 0.1:g} 100v-{height:g}h0.8V100z"
        if value in favs:
            fav_bars.append(bar)
            labels.append(f'<span class="label" style="{x};--h:{height:g}%">{value}<small>{share:.1%}</small></span>')
        else:
            others.append(bar)
    paths = f'<path d="{"".join(others)}"/>' if others else ""
    paths += f'<path class="fav" d="{"".join(fav_bars)}"/>' if fav_bars else ""
    ticks = "".join(f'<span style="--x:{v + 0.5:g}%">{v}</span>' for v in range(0, N_VALUES, 10))
    return (f'      <div class="dist-plot" role="img" aria-label="A bar for each value from 0 to 99, '
            f'as tall as its average share of the answers">'
            f'<svg viewBox="0 0 {N_VALUES} 100" preserveAspectRatio="none">{paths}</svg>{"".join(labels)}</div>\n'
            f'      <div class="dist-base" aria-hidden="true">{"".join(never)}</div>\n'
            f'      <div class="dist-axis" aria-hidden="true">{ticks}</div>')


# The Ranking's Answers chart: its height in SVG units, and the shortest a bar
# can be, so that a value given once still shows.
BAR_HEIGHT = 20
MIN_BAR = 2


def _answers_cell(counts) -> str:
    """The Ranking's Answers cell: a bar for each value the model gave, 0 at the left and 99 at the right.

    A bar's height is in proportion to its count, up to the favorite's, with a
    floor of MIN_BAR. Only the values given are drawn, as one path for the
    favorites and one for the rest, which keeps 36 charts small. The SVG
    stretches to the cell's width, one unit per value.
    """
    found = favorites(counts)
    if found is None:
        return '<td class="answers"></td>'
    values, share = found
    top = max(counts)
    others, favs = [], []
    for value, count in enumerate(counts):
        if count:
            height = round(max(MIN_BAR, BAR_HEIGHT * count / top), 1)
            bar = f"M{value} {BAR_HEIGHT}v-{height:g}h1V{BAR_HEIGHT}z"
            (favs if value in values else others).append(bar)
    paths = f'<path d="{"".join(others)}"/>' if others else ""
    paths += f'<path class="fav" d="{"".join(favs)}"/>'
    label = f"Favorite {', '.join(str(v) for v in values)}: {share:.0%} of its valid answers"
    return (f'<td class="answers"><svg viewBox="0 0 {N_VALUES} {BAR_HEIGHT}" preserveAspectRatio="none" '
            f'role="img" aria-label="{label}">{paths}</svg></td>')


# The panel's squares are shaded by count over the favorite's, but never paler
# than HEAT_FLOOR, so that a value given once still shows. Above HEAT_DARK,
# a square's number turns to --on-deep to stay readable.
HEAT_FLOOR = 0.12
HEAT_DARK = 0.6


def _pair(score: float | None, half: float | None) -> list[str]:
    """A score and its interval as the page shows them, such as ["0.158", "±0.025"], or ["—", ""]."""
    return [_number(score), "" if half is None else f"±{half:.3f}"]


def _scored(value: str, pm: str) -> str:
    """A _pair in HTML, as _number and _plus_minus write it. The page's script writes it the same way."""
    return escape(value) + (f' <span class="pm">{escape(pm)}</span>' if pm else "")


def panel_data(r: ModelScores, rank: int, of: int) -> dict:
    """What the panel shows for one model, each part formatted as it's shown.

    The build writes the top model's panel from it, and the page carries every
    model's, as JSON, for its script to put in the panel when a row is
    selected. So every number and word in the panel is made here, once, and
    the script only puts them in place. That matters because JavaScript
    rounds an exact tie, such as 0.0625, the other way from Python.

    The squares run 0 to 99. Each one's t is its count over the favorite's,
    but at least HEAT_FLOOR, and 0 if the model never gave it. The favorites
    are listed apart, so a tie shows every one. Each check's bar is as full
    as shade() says (D28). The key is HTML; the rest is text.
    """
    name, provider = DISPLAY_NAMES[r.model]
    found = favorites(r.counts)
    top = max(r.counts)
    replies = r.draws - r.failed
    if found:
        values, share = found
        key = f"Favorite <strong>{', '.join(map(str, values))}</strong>: {share:.0%} of its valid answers."
    else:
        values, key = [], "No valid answers yet."
    return {
        "model": r.model,
        "about": f"Rank {rank} of {of} · {provider} · {replies} repl{'y' if replies == 1 else 'ies'}",
        "name": name,
        "score": _pair(r.scores["headline"], r.intervals["headline"]),
        "fav": values,
        "t": [round(max(HEAT_FLOOR, count / top), 3) if count else 0 for count in r.counts],
        "key": key,
        "checks": [_pair(r.scores[k], r.intervals[k]) + [round(shade(k, r.scores[k]), 3)] for k in CHECK_NAMES],
    }


def _panel(d: dict) -> str:
    """The panel beside the Ranking, from a model's panel_data: its score, its answers on a 10 by 10 grid, and its nine checks.

    A square darker than HEAT_DARK gets the class hi, for lighter text, as the
    page's script gives it.
    """
    squares = []
    for value, t in enumerate(d["t"]):
        if value in d["fav"]:
            squares.append(f'<span class="fav">{value}</span>')
        elif not t:
            squares.append(f'<span class="none">{value}</span>')
        else:
            dark = ' class="hi"' if t > HEAT_DARK else ""
            squares.append(f'<span{dark} style="--t:{t:.3f}">{value}</span>')
    checks = "".join(
        f'<div><dt>{title}<small>lowest {_short(LOWEST[k], 2)}</small></dt>'
        f'<dd aria-hidden="true"><span class="meter"><i style="--f:{f:.3f}"></i></span></dd>'
        f'<dd class="num">{_scored(value, pm)}</dd></div>'
        for (k, title), (value, pm, f) in zip(CHECK_NAMES.items(), d["checks"]))
    return (f'    <aside class="panel" id="panel" aria-label="Selected model">\n'
            f'      <p class="about">{escape(d["about"])}</p>\n'
            f'      <h3 aria-live="polite">{escape(d["name"])}</h3>\n'
            f'      <p class="big">{_scored(*d["score"])}</p>\n'
            f'      <div class="heat" role="img" aria-label="The answers on a 10 by 10 grid of 0 to 99">'
            f'{"".join(squares)}</div>\n'
            f'      <p class="key"><span class="fav-key">{d["key"]}</span> '
            f'Darker squares were answered more often, and dashed ones never.</p>\n'
            f'      <h4>Bias checks</h4>\n'
            f'      <dl class="checks">{checks}</dl>\n'
            f"      <p class=\"key\">Each bar runs from the check's lowest possible score, empty, to 1, full. "
            f"Uniformity and Last Digit are on a log scale, so that their low scores still differ: "
            f"a Uniformity score of 0.1 fills half its bar.</p>\n"
            f'    </aside>')


def _panels_json(panels: list[dict]) -> str:
    """Every model's panel_data, as the JSON the page's script reads, one model to a line.

    Each < is escaped, so that nothing in it can end the script early, as a
    </script> would. JSON.parse reads \\u003c back as <.
    """
    models = ",\n".join(json.dumps(p, ensure_ascii=False, separators=(",", ":")) for p in panels)
    return f'{{"dark":{HEAT_DARK},"models":[\n{models}\n]}}'.replace("<", "\\u003c")


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


def _model_cell(model: str) -> str:
    name, provider = DISPLAY_NAMES[model]
    return (f'<td class="model"><span class="name">{escape(name)}</span>'
            f' <span class="prov">{escape(provider)}</span></td>')


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
            f'<tr data-model="{escape(r.model)}"><td class="n">{rank}</td>{_model_cell(r.model)}'
            f'<td class="num replies">{r.draws - r.failed}</td>'
            f'<td class="num score">{_number(score)}{_plus_minus(half)}</td>'
            f'{_answers_cell(r.counts)}'
            f'<td class="plot" aria-hidden="true">{bar}</td></tr>'
        )

    ticks = "".join(f'<span class="tick" style="--x:{i / steps * 100:.2f}%">{_short(i * step, decimals)}</span>'
                    for i in range(steps + 1))

    # The summary cards. The most common answer is the value with the largest
    # average share, the lowest such value on a tie.
    best = models[0] if models and models[0].scores["headline"] is not None else None
    shares = average_shares([r.counts for r in models])
    common = max(range(N_VALUES), key=lambda v: shares[v]) if any(shares) else None

    # What the models answer. Its orange values are every model's favorite,
    # listed in the caption from the largest average share down.
    favs = {v for r in models if (found := favorites(r.counts)) for v in found[0]}
    never = sum(1 for share in shares if not share)

    panels = [panel_data(r, rank, len(models)) for rank, r in enumerate(models, 1)]

    text = TEMPLATE.read_text(encoding="utf-8")
    found = set(_PLACEHOLDER.findall(text))
    if found != PLACEHOLDERS:
        raise ValueError(f"the page template's placeholders are {sorted(found)}, not {sorted(PLACEHOLDERS)}")
    values = {
        "__DATE__": f"{as_of:%B} {as_of.day}, {as_of.year}",
        "__ISODATE__": f"{as_of:%Y-%m-%d}",
        "__COUNT__": str(len(models)),
        "__PROVIDERS__": str(len({DISPLAY_NAMES[r.model][1] for r in models})),
        "__ANSWERS__": f"{sum(sum(r.counts) for r in models):,}",
        "__BEST__": _number(best.scores["headline"] if best else None),
        "__BESTMODEL__": "no scores yet" if best is None else
                         f'{escape(DISPLAY_NAMES[best.model][0])}, ±{best.intervals["headline"]:.3f}',
        "__COMMON__": "—" if common is None else str(common),
        "__COMMONSHARE__": "—" if common is None else f"{shares[common]:.1%}",
        "__DIST__": _distribution(shares, favs),
        "__FAVORITES__": _series([f"{v} ({shares[v]:.1%})" for v in sorted(favs, key=lambda v: -shares[v])]),
        "__NEVER__": f"{never} value{'' if never == 1 else 's'}",
        "__PANEL__": _panel(panels[0]) if panels else "",
        "__PANELS__": _panels_json(panels),
        "__STEPS__": str(steps),
        "__TICKS__": ticks,
        "__RANKING__": "\n".join(ranking),
    }
    return _PLACEHOLDER.sub(lambda m: values[m.group()], text)
