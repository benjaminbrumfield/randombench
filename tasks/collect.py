# %%
# RandomBench's collection task (PLAN.md 2.4). Each draw sends one prompt as
# the only message in a fresh conversation and keeps the raw reply. Nothing
# is parsed or scored here: that happens offline from the stored replies.
#
# A push uploads only this file, so it holds its own copy of
# randombench/prompts.py, kept current by `python -m randombench.sync` (D6).
# For now it runs locally only, at a small size: the last cell names two
# local models. Its name is provisional (D10).
import random

import pandas as pd

import kaggle_benchmarks as kbench

N_DRAWS = 12
N_JOBS = 4


# %%
# --- begin randombench/prompts.py ---
# Copied by `python -m randombench.sync` (PLAN.md D6). Edit the source, not this copy.
"""The prompts that ask a model for a random number (PLAN.md 2.1).

Each draw sends one of these as the only message in a fresh conversation.
A prompt's id is computed from its name and its exact text, so editing the
text gives it a new id, and every stored draw points at the text it answered.

Style guide:
- Ask for a random choice. A model that isn't asked to be random can't be
  faulted for having a favorite.
- Say "integer" or "whole number", so a decimal is clearly a wrong answer.
- State the range as 0 to 99, in digits, with both ends included.
- Ask for the number only, so the parser (2.2) sees plain replies.
- Give no example number and mention no bias (7, primes, round numbers).
  Either could steer the answer.
- Vary everything else: wording, tone, length and framing, so no score
  depends on one phrasing.

prompt_problems() checks the rules that code can check.

Once a prompt has been used in a run, never edit or delete it. To change one,
add a new prompt.
"""

import hashlib
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    name: str
    author: str
    text: str

    @property
    def id(self) -> str:
        return prompt_id(self.name, self.text)


def prompt_id(name: str, text: str) -> str:
    """Return a prompt's id: its name, a hyphen, then the first 8 hex digits
    (lowercase) of the SHA-256 digest of its text encoded as UTF-8.

    The name only makes the id readable. The hash is the part that changes
    when the text does. prompt_id("plain", "Pick a number.") returns
    "plain-cf160a12".
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{name}-{digest[:8]}"


def prompt_problems(prompt: Prompt) -> list[str]:
    """Return one message for each rule below that the prompt breaks. An
    empty list means the prompt passes.

    1. Name: lowercase letters, digits and hyphens, starting with a letter.
       "plain" and "raffle-v2" pass; "Plain", "raffle_v2", "2nd" and "" fail.
    2. Numbers: every run of digits in the text is "0" or "99", and both
       appear. That's how the text states the range without giving an
       example number. "0 to 99" and "0–99, so 0 and 99 count" pass;
       "1 to 100", "0 to 99, like 42", "0 to 99.5" and "a random number" fail.

    A prompt that breaks both rules gets two messages, and one that breaks
    the numbers rule in several places still gets only one.
    """
    problems = []
    if not re.fullmatch(r"[a-z][a-z0-9-]*", prompt.name):
        problems.append(
            f"name {prompt.name!r} must be lowercase letters, digits and hyphens,"
            " starting with a letter"
        )
    # Compared as strings, so "00" or "099" doesn't pass for 0 or 99.
    numbers = set(re.findall(r"\d+", prompt.text))
    if numbers != {"0", "99"}:
        problems.append(
            f"text's numbers are {sorted(numbers)}; they must be exactly 0 and 99"
        )
    return problems


CLAUDE = "claude-opus-5-5"

PROMPTS = (
    Prompt(
        name="plain",
        author=CLAUDE,
        text="Pick a random integer from 0 to 99, inclusive. Reply with only the number.",
    ),
    Prompt(
        name="uniform",
        author=CLAUDE,
        text=(
            "Choose an integer uniformly at random from 0 to 99, inclusive, so that"
            " every value in that range is equally likely. Reply with the integer"
            " and nothing else."
        ),
    ),
    Prompt(
        name="raffle",
        author=CLAUDE,
        text=(
            "I'm drawing the winner of a raffle. The tickets are numbered from 0 to"
            " 99, and every ticket has the same chance. Draw one ticket at random and"
            " reply with only its number."
        ),
    ),
    Prompt(
        name="terse",
        author=CLAUDE,
        text="Random integer, 0–99 inclusive. Number only.",
    ),
    Prompt(
        name="role",
        author=CLAUDE,
        text=(
            "You are a random number generator. Output one integer in the range"
            " [0, 99], and nothing else."
        ),
    ),
    Prompt(
        name="casual",
        author=CLAUDE,
        text=(
            "hey, can you give me a random whole number between 0 and 99? 0 and 99"
            " both count. just the number please"
        ),
    ),
)
# --- end randombench/prompts.py ---


# %%
PROMPTS_BY_ID = {p.id: p for p in PROMPTS}


def balanced_schedule(n_draws: int) -> list[str]:
    """Return a prompt id for each draw: every prompt equally often, in random order (D9)."""
    if n_draws % len(PROMPTS):
        raise ValueError(f"{n_draws} draws don't split evenly across {len(PROMPTS)} prompts")
    schedule = [p.id for p in PROMPTS] * (n_draws // len(PROMPTS))
    random.shuffle(schedule)
    return schedule


# %%
@kbench.task(name="draw", store_task=False)
def draw(llm, draw_index: int, prompt_id: str) -> dict:
    # A sub-run record doesn't hold its parameters, so the result carries them
    # (D11). An error is caught and returned for the same reason: an errored
    # sub-run keeps no result.
    record = {"draw_index": draw_index, "prompt_id": prompt_id}
    try:
        record["reply"] = llm.prompt(PROMPTS_BY_ID[prompt_id].text, seed=None)
    except Exception as e:
        record["error"] = f"{type(e).__name__}: {e}"
    return record


# %%
@kbench.task(
    name="randombench-collect",
    description="Asks the model for a random integer from 0 to 99, many times, each in a fresh conversation, and stores every raw reply.",
)
def collect(llm) -> int:
    schedule = balanced_schedule(N_DRAWS)
    draws = draw.evaluate(
        llm=[llm],
        evaluation_data=pd.DataFrame({"draw_index": range(N_DRAWS), "prompt_id": schedule}),
        n_jobs=N_JOBS,
        on_failure="continue",
    )
    # The number of draws with a reply (D14).
    return sum("reply" in run.result for run in draws.completed_runs)


# %%
for name in ["openai/gpt-5.4-nano-2026-03-17", "google/gemini-3.1-flash-lite-preview"]:
    run = collect.run(kbench.llms[name])
