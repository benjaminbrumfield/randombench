# %%
# Phase 1 probe (PLAN.md step 1.2): fake_draws.py with one change. Draw 300
# kills the whole process, to see what a run that dies partway keeps. Like
# fake_draws.py, it never prompts a model and spends no Model Proxy quota.
# A push uploads only this file, so it repeats fake_draws.py instead of
# importing it.
import os
import random

import pandas as pd

import kaggle_benchmarks as kbench

N_DRAWS = 600
# Rows that raise on purpose, to see how failures are recorded: 6 of 600.
FAIL_AT = {37, 137, 237, 337, 437, 537}
# The draw that kills the process. Draws run on threads, so os._exit stops
# every draw still in flight, the way a timeout or a killed kernel would. A
# raise wouldn't do: on_failure="continue" would record it as one failed draw.
CRASH_AT = 300


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
    if draw_index == CRASH_AT:
        os._exit(1)
    if draw_index in FAIL_AT:
        raise RuntimeError(f"injected failure at draw {draw_index}")
    return {"value": random.randint(0, 99)}


# %%
@kbench.task(
    name="randombench-fake-draws-crash",
    description="Phase 1 probe: fake draws that kill the process at draw 300. Never prompts the model.",
)
def fake_draws_crash(llm) -> dict:
    draws = fake_draw.evaluate(
        llm=[llm],
        evaluation_data=pd.DataFrame({"draw_index": range(N_DRAWS)}),
        n_jobs=8,
        on_failure="continue",
    )
    # Never reached: the process dies at draw 300.
    return {
        "headline": 0.5,
        "draws_completed": len(draws.completed_runs),
        "draws_errored": len(draws.errored_runs),
    }


# Kaggle rejects a run labelled with a model it doesn't serve (PLAN.md 1.2), so
# on Kaggle the run carries the -m model's label; that model is never prompted.
ON_KAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "").lower() == "batch"
fake_draws_crash.run(kbench.llm if ON_KAGGLE else PythonRandom())
