# %%
# RandomBench's collection task (PLAN.md 2.4). Each draw sends one prompt as
# the only message in a fresh conversation and keeps the raw reply. Nothing
# is parsed or scored here: that happens offline from the stored replies.
#
# For now it runs locally only, at a small size. It imports randombench,
# which a push wouldn't upload (D6), and its name is provisional (D10).
import random

import pandas as pd

import kaggle_benchmarks as kbench

from randombench.prompts import PROMPTS

N_DRAWS = 12
N_JOBS = 4

PROMPTS_BY_ID = {p.id: p for p in PROMPTS}


# %%
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
