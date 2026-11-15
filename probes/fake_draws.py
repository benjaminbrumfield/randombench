# %%
# Phase 1 probe (PLAN.md steps 1.1, 1.1b, and 1.2). Throwaway: the "draws" come
# from Python's random module and no model is ever prompted, so running this
# spends no Model Proxy quota. It shows how kaggle_benchmarks records a
# 600-row sub-task evaluation, including rows that fail.
import os
import random

import pandas as pd

import kaggle_benchmarks as kbench

N_DRAWS = 600
# Rows that raise on purpose, to see how failures are recorded: 6 of 600.
FAIL_AT = {37, 137, 237, 337, 437, 537}


# %%
class PythonRandom(kbench.LLMChat):
    """Stand-in model that labels the run with where its draws come from.

    The library labels a run with the first LLMChat passed to the task, so
    passing kbench.llm would credit these draws to LLM_DEFAULT (PLAN.md 1.1b).
    It can't answer prompts: one sent here is a bug, so it fails loudly.
    """

    def __init__(self):
        super().__init__(name="python-random")

    def invoke(self, *args, **kwargs):
        raise RuntimeError("python-random is a stand-in model and can't be prompted")


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


# Kaggle rejects a run labelled with a model it doesn't serve: the python-random
# run completed, then the task failed with "Not found" (PLAN.md 1.2). On Kaggle
# the run carries the -m model's label instead, though that model is never
# prompted. KAGGLE_KERNEL_RUN_TYPE is the signal the library itself uses to
# detect a Kaggle batch run (_config.py).
ON_KAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "").lower() == "batch"
fake_draws.run(kbench.llm if ON_KAGGLE else PythonRandom())

# %%
# Deletes every .task.json and .run.json except this task's own and its latest
# run. Without it, Kaggle took one of the 600 draws as the task (PLAN.md 1.2).
# When the CLI converts this file to a notebook, jupytext uncomments the magic
# below, so it runs on Kaggle and stays a comment when run as a script.
# %choose randombench-fake-draws
