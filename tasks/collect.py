# %%
# RandomBench's collection task (PLAN.md 2.4). Each draw sends one prompt as
# the only message in a fresh conversation and keeps the raw reply. Nothing
# is parsed or scored here: that happens offline from the stored replies.
#
# A push uploads only this file, so it holds its own copy of
# randombench/prompts.py, kept current by `python -m randombench.sync` (D6).
#
# It makes draws only for the models in MODELS (D12). Any other model's run
# prompts nothing and returns 0. That includes a push's creation run, which
# uses the server's default model, Gemini 3.8 Flash: its draws are done, so
# it's no longer listed (D23). It also includes a local run, whose kbench.llm
# is LLM_DEFAULT from .env, a model the server doesn't offer.
#
# Each run prints what the pilot has to learn about the server (4.1), and a
# listed model's run also prints how many HTTP requests its draws made. The
# printed lines are in `kaggle b t log`.
import importlib.metadata
import os
import random
import threading

import pandas as pd

import kaggle_benchmarks as kbench

# The full runs (4.3, 4.8, 4.11) made 600 draws, and the pilots (4.2, 4.7,
# 4.10) 24. In 4.11, the four costliest models made 96, to stay well inside
# the daily quota. Version 8 piloted the 13 flagship models with 24 draws
# each. Since version 9, each run goes to the model with the fewest replies.
# Version 9's runs made 96, and version 10's 300. Version 11 piloted GPT-6.2
# Astra and GPT-6.2 Sol with 24 draws each. Its runs on the two Gemini 2.5
# models, which Kaggle had dropped by 2026-11-18, never started.
#
# Since version 12, each run makes a random number of draws, from MIN_DRAWS to
# MAX_DRAWS, both included, chosen when the run starts. Each draw's prompt is
# chosen at random too, on its own, so a run's prompts can come up unequally
# often. Until then, a run's draws split evenly across the six prompts (D9).
# Versions 12 and 13 drew 50 to 150 times. Since version 14, it's 100 to 300,
# so each batch spends more of the day's quota and fewer batches run.
MIN_DRAWS, MAX_DRAWS = 100, 300
# Draws made at once. Model Proxy holds each call's maximum possible cost
# against the daily quota until it finishes (4.11), and a flagship model's
# hold can be several dollars, so runs since the flagship pilots make one at
# a time. The earlier runs used 4.
N_JOBS = 1

# The llm.name of each model, from the server's LLMS_AVAILABLE. All but the
# flagship models are in PLAN.md's model table. Kaggle runs and local runs use
# the same names. Any other
# model makes no draws, so an id that's wrong here costs a wasted run, not
# unwanted calls. Gemini 3.8 Flash, the server's default model, is in the
# table but not here, so a push prompts nothing (D23).
MODELS = (
    "openai/gpt-5.4-nano-2026-03-17",
    "google/gemini-3.1-flash-lite-preview",
    "google/gemini-3.5-flash-lite",
    "anthropic/claude-haiku-4-5@20251001",
    "google/gemma-4-26b-a4b",
    "openai/gpt-oss-120b",
    # Added in 4.10 (D23).
    # Kaggle no longer listed the two Gemini 2.5 models on 2026-11-18. They
    # stay here until we decide how to retire them.
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "google/gemini-3-flash-preview",
    "google/gemini-3.1-pro-preview",
    "google/gemini-3.6-flash",
    "google/gemini-4-argon",
    "xai/grok-4.6",
    "xai/grok-4.7",
    "openai/gpt-5.4-mini-2026-03-17",
    "openai/gpt-5.6-luna",
    "openai/gpt-5.6-terra",
    "openai/gpt-6-luna",
    "anthropic/claude-haiku-5-5",
    "anthropic/claude-sonnet-4-6@default",
    "anthropic/claude-sonnet-5@default",
    "anthropic/claude-sonnet-5-5@default",
    # The flagship models, added after 4.11: every model the server listed
    # and Model Proxy served, but Gemini 3.8 Flash (D23).
    "google/gemma-4-31b",
    "anthropic/claude-opus-4-5@20251101",
    "anthropic/claude-opus-4-6@default",
    "anthropic/claude-opus-4-7@default",
    "anthropic/claude-opus-4-8@default",
    "anthropic/claude-opus-5@default",
    "anthropic/claude-opus-5-5@default",
    "openai/gpt-5.4-2026-03-05",
    "openai/gpt-5.5-2026-04-23",
    "openai/gpt-5.6-sol",
    "openai/gpt-6-sol",
    "openai/gpt-6.1-sol",
    "openai/gpt-6-astra",
    # Released 2026-11-17, and piloted in version 11.
    "openai/gpt-6.2-astra",
    "openai/gpt-6.2-sol",
    # Listed by Kaggle on 2026-11-19, with no release date given.
    "google/gemini-4-neon",
)


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
- Vary everything else: wording, tone, length, and framing, so no score
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

    1. Name: lowercase letters, digits, and hyphens, starting with a letter.
       "plain" and "raffle-v2" pass; "Plain", "raffle_v2", "2nd", and "" fail.
    2. Numbers: every run of digits in the text is "0" or "99", and both
       appear. That's how the text states the range without giving an
       example number. "0 to 99" and "0–99, so 0 and 99 count" pass;
       "1 to 100", "0 to 99, like 42", "0 to 99.5", and "a random number" fail.

    A prompt that breaks both rules gets two messages, and one that breaks
    the numbers rule in several places still gets only one.
    """
    problems = []
    if not re.fullmatch(r"[a-z][a-z0-9-]*", prompt.name):
        problems.append(
            f"name {prompt.name!r} must be lowercase letters, digits, and hyphens,"
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


def random_schedule(n_draws: int) -> list[str]:
    """Return a prompt id for each draw, each chosen at random, independently
    of the others. Before version 12, a balanced shuffle gave every prompt
    the same number of draws (D9)."""
    return random.choices([p.id for p in PROMPTS], k=n_draws)


# %%
def report_environment() -> None:
    """Print the server facts that 1.3's environment dump left open (PLAN.md 4.1)."""
    # Local runs use 0.6.2, and extract.py expects its run-file shapes.
    print(f"kaggle_benchmarks {importlib.metadata.version('kaggle_benchmarks')}")
    # 0.6.2 never reads this variable. If the server's version does, its runs
    # may not use the provider's default reasoning (D1).
    effort = os.environ.get("KAGGLE_AGENT_LLM_REASONING_EFFORT")
    print(f"KAGGLE_AGENT_LLM_REASONING_EFFORT={effort!r}")


class RequestCounter:
    """Counts the HTTP requests a model's client sends, retries included.

    The OpenAI client retries a failed request on its own, and a retry leaves
    no trace in the run file (2.4). An httpx request hook sees every request
    (2.4b). It reaches into a private attribute, so a model without one, such
    as a stand-in or a client from a different library version, isn't counted.
    """

    def __init__(self, llm):
        self.count = 0
        self._lock = threading.Lock()
        try:
            self._hooks = llm.client._client.event_hooks["request"]
        except (AttributeError, KeyError, TypeError):
            self._hooks = None

    @property
    def counting(self) -> bool:
        return self._hooks is not None

    def _hook(self, request) -> None:
        with self._lock:  # draws run in N_JOBS threads
            self.count += 1

    def __enter__(self):
        if self.counting:
            self._hooks.append(self._hook)
        return self

    def __exit__(self, *exc) -> None:
        if self.counting:
            self._hooks.remove(self._hook)


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
    # The slug, randombench-draws, is permanent (D10). The title can change
    # only to another that slugifies the same way.
    name="RandomBench Draws",
    description="Asks the model for a random integer from 0 to 99, many times, each in a fresh conversation, and stores every raw reply.",
)
def collect(llm) -> int:
    # Printed in every run, the creation run included, so a push alone shows
    # the server's facts without prompting a model.
    report_environment()
    if llm.name not in MODELS:
        # A push's creation run, or a model RandomBench doesn't cover (D12).
        print(f"{llm.name} isn't one of RandomBench's models: no draws")
        return 0
    n_draws = random.randint(MIN_DRAWS, MAX_DRAWS)
    schedule = random_schedule(n_draws)
    with RequestCounter(llm) as requests:
        draws = draw.evaluate(
            llm=[llm],
            evaluation_data=pd.DataFrame({"draw_index": range(n_draws), "prompt_id": schedule}),
            n_jobs=N_JOBS,
            on_failure="continue",
        )
    if requests.counting:
        print(f"HTTP requests: {requests.count} for {n_draws} draws, so {requests.count - n_draws} were retries")
    else:
        print("HTTP requests: not counted, because the model's client has no httpx hooks")
    # The number of draws with a reply (D14).
    return sum("reply" in run.result for run in draws.completed_runs)


# %%
# On Kaggle, kbench.llm is the -m model, or the server's default in a push's
# creation run. Locally it's LLM_DEFAULT. Assigned, so the notebook doesn't
# display the run.
run = collect.run(kbench.llm)

# %%
# Deletes every .task.json and .run.json except this task's own and its latest
# run, so the download holds the parent run file, not one file per draw (1.2).
# jupytext turns this comment into the magic when the CLI converts the file,
# and it stays a comment when the file runs as a script. It names the task's
# variable, because the title has a space.
# %choose collect
