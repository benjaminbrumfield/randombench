"""The findings on the leaderboard page (PLAN.md 5.5, D29), checked against data/.

The rest of the page is rebuilt from data/, but its findings are written by
hand in randombench/page.html, so new draws could leave them contradicting the
tables above them. So each finding has a test here, test_finding_ and its
data-finding, that checks what it claims. When new draws break one, rewrite
the finding first, then its test to match.
"""

import re
import statistics

import numpy as np
import pandas as pd

from randombench.build import DATA
from randombench.checks import MULTIPLE_OF_5, MULTIPLE_OF_10, PRIME, REPEATED_DIGITS
from randombench.extract import load_draws
from randombench.page import TEMPLATE, favorites, ranked
from randombench.parsing import parse_reply
from randombench.score import score_draws

DRAWS = load_draws(DATA)
MODELS = {r.model: r for r in ranked(score_draws(DRAWS))}  # every model with a reply, as ranked on the page
COUNTS = {model: np.array(r.counts) for model, r in MODELS.items()}
ALL = sum(COUNTS.values())  # every valid answer, pooled across the models
VALUES = np.arange(100)

GEMINI_25_FLASH = "google/gemini-2.5-flash"
GEMINI_25_PRO = "google/gemini-2.5-pro"
GEMINI_31_PRO = "google/gemini-3.1-pro-preview"
GEMINI_31_FLASH_LITE = "google/gemini-3.1-flash-lite-preview"
GPT_OSS = "openai/gpt-oss-120b"
GPT_56_LUNA = "openai/gpt-5.6-luna"
GPT_6_LUNA = "openai/gpt-6-luna"
GROK_46 = "xai/grok-4.6"
GROK_47 = "xai/grok-4.7"
HAIKU_45 = "anthropic/claude-haiku-4-5@20251001"
HAIKU_55 = "anthropic/claude-haiku-5-5"
SONNET_46 = "anthropic/claude-sonnet-4-6@default"
SONNET_5 = "anthropic/claude-sonnet-5@default"
SONNET_55 = "anthropic/claude-sonnet-5-5@default"


def share(counts: np.ndarray, mask: np.ndarray) -> float:
    """The share of the answers in counts that are in the mask's set."""
    return counts[mask].sum() / counts.sum()


def answered(mask: np.ndarray) -> set[str]:
    """The models that have given at least one answer in the mask's set."""
    return {model for model, counts in COUNTS.items() if counts[mask].any()}


def less_random(model: str, than: str) -> bool:
    """Whether model's headline interval lies wholly below than's, so the data can tell them apart."""
    def bounds(m):
        r = MODELS[m]
        return r.scores["headline"] - r.intervals["headline"], r.scores["headline"] + r.intervals["headline"]
    return bounds(model)[1] < bounds(than)[0]


def median_tokens(model: str) -> float:
    """The model's median output tokens over the calls that got a reply."""
    return statistics.median(d["output_tokens"] for d in DRAWS
                             if d["model"] == model and d["reply"] is not None and d["output_tokens"] is not None)


def test_every_finding_on_the_page_has_a_test_and_every_test_a_finding():
    on_page = set(re.findall(r'data-finding="([a-z-]+)"', TEMPLATE.read_text(encoding="utf-8")))
    tested = {name.removeprefix("test_finding_").replace("_", "-") for name in globals() if name.startswith("test_finding_")}
    assert on_page == tested


def test_finding_no_model_comes_close():
    # "as repetitive as a fair pick among fewer than a fifth of the 100 values"
    assert max(r.scores["uniformity"] for r in MODELS.values()) < 0.2


def test_finding_three_numbers_dominate():
    favorite = {model: set(favorites(counts)[0]) for model, counts in COUNTS.items()}
    assert favorite.pop(GPT_OSS) == {57}
    assert all(f <= {42, 47, 73} for f in favorite.values()), favorite
    assert ALL[[42, 47, 73]].sum() > ALL.sum() / 2  # most of all the answers
    assert COUNTS[GEMINI_31_FLASH_LITE][42] == COUNTS[GEMINI_31_FLASH_LITE].sum()
    for model in [SONNET_5, SONNET_55]:
        # "about 9 times in 10": Sonnet 5's is 89.96%, so "more than" would be wrong.
        assert 0.85 < COUNTS[model][47] / COUNTS[model].sum() < 0.95, model


def test_finding_newer_isnt_more_random():
    assert next(iter(MODELS)) == GEMINI_25_FLASH  # ranks first
    newer_geminis = [m for m in MODELS if m.startswith("google/gemini-") and not m.startswith("google/gemini-2.5")]
    assert len(newer_geminis) >= 7
    assert all(less_random(m, GEMINI_25_FLASH) for m in newer_geminis)
    assert less_random(GROK_47, GROK_46)
    assert less_random(GPT_6_LUNA, GPT_56_LUNA)
    assert less_random(SONNET_5, SONNET_46) and less_random(SONNET_55, SONNET_46)
    assert less_random(HAIKU_45, HAIKU_55)  # the other way


def test_finding_whole_regions_go_unused():
    assert share(ALL, (VALUES >= 20) & (VALUES <= 79)) > 0.95
    assert share(ALL, np.isin(VALUES % 10, [2, 3, 7])) > 0.95
    assert answered(MULTIPLE_OF_10) == set() and answered(VALUES < 10) == set()
    assert answered(VALUES > 89) == {GEMINI_25_FLASH}
    assert answered(REPEATED_DIGITS) == {GEMINI_25_FLASH}
    assert answered(MULTIPLE_OF_5) == {GEMINI_25_FLASH, GROK_46}
    # Every other model scores the same on the four checks.
    others = [r for model, r in MODELS.items() if model not in {GEMINI_25_FLASH, GROK_46}]
    for check in ["multiples_of_10", "multiples_of_5", "edges", "repeated_digits"]:
        assert len({r.scores[check] for r in others}) == 1, check


def test_finding_a_share_check_can_pass_by_accident():
    assert MODELS[GPT_OSS].scores["primes"] > 0.95  # close to perfect
    top_three = [int(v) for v in np.argsort(COUNTS[GPT_OSS])[::-1][:3]]
    assert top_three == [57, 42, 73]
    assert [v for v in top_three if PRIME[v]] == [73]
    for model, counts in COUNTS.items():
        if set(favorites(counts)[0]) & {47, 73}:
            assert share(counts, PRIME) > 0.5, model  # more than half the time
    for model in [SONNET_5, SONNET_55]:
        assert share(COUNTS[model], PRIME) > 0.95


def test_finding_nearly_every_reply_is_valid():
    invalid = [d for d in DRAWS if d["reply"] is not None and parse_reply(d["reply"]) is None]
    assert {d["model"] for d in invalid} == {GROK_46}
    assert all("\\confidence{" in d["reply"] for d in invalid)


def test_finding_reasoning_helps_a_little():
    tokens = {model: median_tokens(model) for model in MODELS}
    # Many reason: a two-digit answer takes a few tokens, so a median past 50 means reasoning.
    assert sum(t > 50 for t in tokens.values()) >= len(tokens) / 4
    assert all(tokens[m] > 200 for m in [GEMINI_25_PRO, GEMINI_31_PRO, GROK_46])
    assert all(tokens[m] <= 5 for m in list(MODELS)[-5:])  # the five least random
    assert all(t <= 5 for m, t in tokens.items() if m.startswith("anthropic/"))
    # More tokens tend to score higher: Spearman's correlation, the correlation of the ranks.
    ranks = pd.DataFrame({"tokens": tokens, "score": {m: r.scores["headline"] for m, r in MODELS.items()}}).rank()
    assert ranks["tokens"].corr(ranks["score"]) > 0.5
