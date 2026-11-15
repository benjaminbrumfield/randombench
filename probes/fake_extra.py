# %%
# Probe for PLAN.md 4.13. Throwaway: a third made-up score, so the scratch
# Benchmark has more tasks than models, to see whether that changes the
# leaderboard's layout. It never prompts a model, so running it spends no
# Model Proxy quota.
import os

import kaggle_benchmarks as kbench

# Made-up scores, different per model so a table or chart shows which is
# which. Any other model, such as a creation run's, gets the default.
SCORES = {"openai/gpt-5.4-nano-2026-03-17": 0.30, "google/gemini-3.8-flash": 0.60}
DEFAULT = 0.50


class StandIn(kbench.LLMChat):
    """The model for a local run. The task never prompts it."""

    def __init__(self):
        super().__init__(name="stand-in")

    def invoke(self, *args, **kwargs):
        raise RuntimeError("the stand-in model can't be prompted")


# %%
@kbench.task(
    name="randombench-fake-extra",
    description="Probe: a made-up score, so the scratch Benchmark has more tasks than models. Never prompts the model.",
)
def fake_extra(llm) -> float:
    return SCORES.get(llm.name, DEFAULT)


# On Kaggle the run carries the -m model's label, or the server's default in a
# push's creation run. Neither is prompted. Assigned, so the notebook doesn't
# display the run.
ON_KAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "").lower() == "batch"
run = fake_extra.run(kbench.llm if ON_KAGGLE else StandIn())
