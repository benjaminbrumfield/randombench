# %%
# Phase 1 probe (PLAN.md steps 1.1 and 1.2). Throwaway: the "draws" come from
# Python's random module and the model passed in is never prompted, so running
# this spends no Model Proxy quota. It shows how kaggle_benchmarks records a
# 600-row sub-task evaluation, including rows that fail.
import random

import pandas as pd

import kaggle_benchmarks as kbench

N_DRAWS = 600
# Rows that raise on purpose, to see how failures are recorded: 6 of 600.
FAIL_AT = {37, 137, 237, 337, 437, 537}


# %%
@kbench.task(name="fake-draw", store_task=False)
def fake_draw(llm, draw_index: int) -> dict:
    if draw_index in FAIL_AT:
        raise RuntimeError(f"injected failure at draw {draw_index}")
    return {"value": random.randint(0, 99)}


# %%
@kbench.task(
    name="randombench-fake-draws",
    description="Phase 1 probe: fake draws from Python's random module. Never prompts the model.",
)
def fake_draws(llm) -> dict:
    draws = fake_draw.evaluate(
        llm=[llm],
        evaluation_data=pd.DataFrame({"draw_index": range(N_DRAWS)}),
        n_jobs=8,
        on_failure="continue",
    )
    # Made-up sub-scores: this probe is about how results are recorded and
    # displayed, not about scoring. The two counts are real.
    return {
        "headline": 0.5,
        "invalid": 0.9,
        "uniformity": 0.8,
        "primes": 0.7,
        "draws_completed": len(draws.completed_runs),
        "draws_errored": len(draws.errored_runs),
    }


fake_draws.run(kbench.llm)
