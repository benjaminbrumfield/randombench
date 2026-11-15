# %%
# Probe for PLAN.md 4.13. Throwaway: a task whose value is another task's
# 95% half-width, the way ArXivMath shows its "95% CI (±%)" column. Added to
# the scratch Benchmark with fake_extra.py, it also gives that Benchmark more
# tasks than models, to see whether that changes the leaderboard's layout.
# It never prompts a model, so running it spends no Model Proxy quota.
import os

import kaggle_benchmarks as kbench

# Made-up half-widths, different per model so a table or chart shows which is
# which. Any other model, such as a creation run's, gets the default.
HALF_WIDTHS = {"openai/gpt-5.4-nano-2026-03-17": 0.05, "google/gemini-3.8-flash": 0.03}
DEFAULT = 0.04


class StandIn(kbench.LLMChat):
    """The model for a local run. The task never prompts it."""

    def __init__(self):
        super().__init__(name="stand-in")

    def invoke(self, *args, **kwargs):
        raise RuntimeError("the stand-in model can't be prompted")


# %%
@kbench.task(
    name="randombench-fake-ci",
    description="Probe: a made-up 95% half-width returned as a task's value, the way ArXivMath shows its CI column. Never prompts the model.",
)
def fake_ci(llm) -> float:
    return HALF_WIDTHS.get(llm.name, DEFAULT)


# On Kaggle the run carries the -m model's label, or the server's default in a
# push's creation run. Neither is prompted. Assigned, so the notebook doesn't
# display the run.
ON_KAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "").lower() == "batch"
run = fake_ci.run(kbench.llm if ON_KAGGLE else StandIn())
