"""The findings on the leaderboard page (PLAN.md 5.5, D29), checked against data/.

The rest of the page is rebuilt from data/, but its findings are written by
hand in randombench/page.html, so new draws could leave them contradicting the
numbers above them. So each finding has a test here, test_finding_ and its
data-finding, that checks what it claims. When new draws break one, rewrite
the finding first, then its test to match.
"""

import re
import statistics

import numpy as np
import pandas as pd
import pytest

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
GEMINI_4_NEON = "google/gemini-4-neon"
GPT_OSS = "openai/gpt-oss-120b"
GPT_54 = "openai/gpt-5.4-2026-03-05"
GPT_56_LUNA = "openai/gpt-5.6-luna"
GPT_56_TERRA = "openai/gpt-5.6-terra"
GPT_6_LUNA = "openai/gpt-6-luna"
GROK_46 = "xai/grok-4.6"
GROK_47 = "xai/grok-4.7"
HAIKU_45 = "anthropic/claude-haiku-4-5@20251001"
HAIKU_55 = "anthropic/claude-haiku-5-5"
OPUS_46 = "anthropic/claude-opus-4-6@default"
OPUS_48 = "anthropic/claude-opus-4-8@default"
OPUS_55 = "anthropic/claude-opus-5-5@default"
SONNET_46 = "anthropic/claude-sonnet-4-6@default"
SONNET_5 = "anthropic/claude-sonnet-5@default"
SONNET_55 = "anthropic/claude-sonnet-5-5@default"
RAFFLE = "raffle-66f5f2a6"  # the prompt_id of the raffle wording


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


def gpt_version(model: str) -> float | None:
    """The version number of a GPT model, such as 5.4 for GPT-5.4 mini, or None for any other model."""
    match = re.match(r"openai/gpt-(\d+(?:\.\d+)?)", model)
    return float(match[1]) if match else None


def invalid_replies(model: str) -> list[dict]:
    """The model's draws that got a reply the parser rejects."""
    return [d for d in DRAWS if d["model"] == model and d["reply"] is not None and parse_reply(d["reply"]) is None]


def test_every_finding_on_the_page_has_a_test_and_every_test_a_finding():
    on_page = set(re.findall(r'data-finding="([a-z-]+)"', TEMPLATE.read_text(encoding="utf-8")))
    tested = {name.removeprefix("test_finding_").replace("_", "-") for name in globals() if name.startswith("test_finding_")}
    assert on_page == tested


def test_finding_no_model_comes_close():
    # On the Ranking's Score: "Even the most random model scores less than 0.2,
    # and the top of its interval is less than 0.2 too." It's 0.158 ± 0.025 now.
    most_random = max(MODELS.values(), key=lambda r: r.scores["headline"])
    assert most_random.scores["headline"] + most_random.intervals["headline"] < 0.2


# GPT-6.2 Astra and GPT-6.2 Sol favor 63, and Gemini 4 Neon 83, which the
# finding's exceptions don't name. It waits to be rewritten until their
# replies catch up with the other models', as the 13 flagship models' pilots
# waited for their full runs. strict, so the test fails here if the finding
# comes true again before its mark is removed.
@pytest.mark.xfail(strict=True, reason="GPT-6.2 Astra, GPT-6.2 Sol, and Gemini 4 Neon break this finding: rewrite it once their replies catch up, then its test")
def test_finding_three_numbers_dominate():
    favorite = {model: set(favorites(counts)[0]) for model, counts in COUNTS.items()}
    assert ALL[[42, 47, 73]].sum() > ALL.sum() / 2  # most of all the answers
    assert favorite.pop(GPT_OSS) == {57}
    # "about equally often": within 10% of each other, whichever leads. They're 59 and 58 now.
    assert favorite.pop(GPT_54) <= {47, 57}
    assert abs(COUNTS[GPT_54][57] - COUNTS[GPT_54][47]) <= 0.1 * COUNTS[GPT_54][[47, 57]].max()
    assert all(f <= {42, 47, 73} for f in favorite.values()), favorite
    for model in [GEMINI_31_FLASH_LITE, OPUS_46]:
        assert COUNTS[model][42] == COUNTS[model].sum(), model
    for model in [SONNET_5, SONNET_55]:
        # "about 9 times in 10": Sonnet 5's is 89.96%, so "more than" would be wrong.
        assert 0.85 < COUNTS[model][47] / COUNTS[model].sum() < 0.95, model


def test_finding_newer_isnt_more_random():
    first, second = list(MODELS)[:2]
    assert first == GPT_54
    later_gpts = [m for m in MODELS if (gpt_version(m) or 0) > 5.4]  # GPT-5.5 to GPT-6.2
    assert len(later_gpts) >= 10 and max(map(gpt_version, later_gpts)) == 6.2
    assert all(less_random(m, GPT_54) for m in later_gpts)
    assert second == GEMINI_25_FLASH
    newer_geminis = [m for m in MODELS if m.startswith("google/gemini-") and not m.startswith("google/gemini-2.5")]
    assert len(newer_geminis) >= 7 and GEMINI_4_NEON in newer_geminis
    # Second, so ahead of each of them, and every interval is apart but Gemini 4 Neon's.
    assert all(less_random(m, GEMINI_25_FLASH) for m in newer_geminis if m != GEMINI_4_NEON)
    assert not less_random(GEMINI_4_NEON, GEMINI_25_FLASH)
    assert less_random(GROK_47, GROK_46)
    assert less_random(GPT_6_LUNA, GPT_56_LUNA)
    assert less_random(SONNET_5, SONNET_46) and less_random(SONNET_55, SONNET_46)
    # The other way
    assert less_random(HAIKU_45, HAIKU_55)
    claudes = [m for m in MODELS if m.startswith("anthropic/") and m != OPUS_55]
    assert len(claudes) >= 10
    assert all(less_random(m, OPUS_55) for m in claudes)


def test_finding_whole_regions_go_unused():
    assert share(ALL, (VALUES >= 20) & (VALUES <= 79)) > 0.95
    two_three_seven = np.isin(VALUES % 10, [2, 3, 7])
    assert share(ALL, two_three_seven) > 0.90
    # Gemini 4 Neon is the exception: the one model with fewer than half its answers ending in 2, 3, or 7.
    assert {m for m, counts in COUNTS.items() if share(counts, two_three_seven) < 0.5} == {GEMINI_4_NEON}
    assert share(COUNTS[GEMINI_4_NEON], two_three_seven) < 1 / 3
    neon_last_digits = np.bincount(VALUES % 10, weights=COUNTS[GEMINI_4_NEON])
    assert neon_last_digits.argmax() == 8 and (neon_last_digits < neon_last_digits[8]).sum() == 9
    trio = {GPT_54, GEMINI_25_FLASH, GROK_46}
    assert trio <= set(list(MODELS)[:4])  # "three of the four most random models"
    assert answered(VALUES > 89) == trio
    assert answered(REPEATED_DIGITS) == trio
    assert answered(VALUES < 10) == {GPT_54, GROK_46}
    assert answered(MULTIPLE_OF_10) == {GPT_54, GPT_56_TERRA}
    venturers = trio | {GPT_56_TERRA}
    assert answered(MULTIPLE_OF_5 | (VALUES < 10) | (VALUES > 89) | REPEATED_DIGITS) == venturers
    # Every other model scores the same on the four checks.
    others = [r for model, r in MODELS.items() if model not in venturers]
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
    replies = [d for d in DRAWS if d["reply"] is not None]
    assert sum(parse_reply(d["reply"]) is not None for d in replies) > 0.99 * len(replies)
    assert {model for model in MODELS if invalid_replies(model)} == {OPUS_48, GROK_46, OPUS_55}
    # Opus 4.8's are all raffle draws, and it declines nearly every raffle draw.
    declined = invalid_replies(OPUS_48)
    raffles = [d for d in replies if d["model"] == OPUS_48 and d["prompt_id"] == RAFFLE]
    assert all(d["prompt_id"] == RAFFLE for d in declined)
    assert len(declined) > 0.9 * len(raffles)
    assert all(d["reply"].startswith("I can't") for d in declined)  # "saying it can't draw a truly random number"
    better_ways = ("random.org", "randbetween", "randint")  # randint is Python's
    assert all(any(way in d["reply"].lower() for way in better_ways) for d in declined)
    # "A few offer a casual pick anyway", in bold, and "some warn" that it would favor 37, 42, or 73.
    assert 1 <= sum(bool(re.search(r"\*\*\d+\*\*", d["reply"])) for d in declined) <= 0.2 * len(declined)
    assert sum(bool(re.search(r"like (37|42|73)\b", d["reply"])) for d in declined) >= 5
    assert all("\\confidence{" in d["reply"] for d in invalid_replies(GROK_46))
    # Opus 5.5's two: an empty reply, and its reasoning written out before the number.
    empty, reasoned = sorted(invalid_replies(OPUS_55), key=lambda d: len(d["reply"]))
    assert empty["reply"].strip("\ufeff \n") == ""
    assert reasoned["reply"].rstrip().endswith("73") and len(reasoned["reply"]) > 50


def test_finding_reasoning_helps_a_little():
    tokens = {model: median_tokens(model) for model in MODELS}
    # A two-digit answer takes a few tokens, so a median past 50 means reasoning at length.
    reasoners = {m for m, t in tokens.items() if t > 50}
    assert len(reasoners) == 11
    assert {m for m, t in tokens.items() if t > 200} == {GEMINI_25_PRO, GEMINI_31_PRO, GROK_46}
    # More tokens tend to score higher: Spearman's correlation, the correlation of the ranks. It's 0.37 now.
    ranks = pd.DataFrame({"tokens": tokens, "score": {m: r.scores["headline"] for m, r in MODELS.items()}}).rank()
    assert ranks["tokens"].corr(ranks["score"]) > 0.3
    assert next(iter(MODELS)) == GPT_54 and tokens[GPT_54] == 4
    assert all(m.startswith(("google/", "xai/")) or m == GPT_OSS for m in reasoners), reasoners
    # "Still far from uniform": none scores above 0.15 on Uniformity. Gemini 4 Neon's 0.11 is the highest.
    assert all(MODELS[m].scores["uniformity"] <= 0.15 for m in reasoners)
