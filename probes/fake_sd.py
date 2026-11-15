# %%
# Probe for PLAN.md 4.13. Throwaway: does a result's standard deviation make
# Kaggle show its interval? kaggle_benchmarks writes only a value and a
# confidence interval for a (value, ±) result. The NumericResult type also has
# a standardDeviation, which a Benchmark's "with confidence interval" overall
# score needs (4.9). The library can't write one: its bundled protobuf type
# for a result has only value and confidence_interval, and parsing drops any
# other field. So this task wraps the library's private file writer and adds
# the field to the JSON afterwards. It never prompts a model, so running it
# spends no Model Proxy quota.
import json
import os
from pathlib import Path

import kaggle_benchmarks as kbench
from kaggle_benchmarks.kaggle import client

# Every model gets the same made-up result. The standard deviation is the
# interval divided by 2.5, not 1.96, so the ± a page shows tells which number
# it read: 0.05 is the interval, 0.0392 is 1.96 standard deviations, and 0.02
# is the standard deviation itself.
VALUE = 0.40
INTERVAL = 0.05
RATIO = 2.5

_save_proto = client.save_proto


def _save_with_standard_deviation(message, path):
    _save_proto(message, path)
    if not str(path).endswith(".run.json"):
        return
    run = json.loads(Path(path).read_text(encoding="utf-8"))
    for result in run.get("results", []):
        numeric = result.get("numericResult")
        if numeric and "confidenceInterval" in numeric:
            numeric["standardDeviation"] = numeric["confidenceInterval"] / RATIO
    Path(path).write_text(json.dumps(run, indent=2), encoding="utf-8")


# store_run looks save_proto up by name in its own module, in 0.6.2 and in the
# server's 0.6.3, so replacing it there takes effect.
client.save_proto = _save_with_standard_deviation


class StandIn(kbench.LLMChat):
    """The model for a local run. The task never prompts it."""

    def __init__(self):
        super().__init__(name="stand-in")

    def invoke(self, *args, **kwargs):
        raise RuntimeError("the stand-in model can't be prompted")


# %%
@kbench.task(
    name="randombench-fake-sd",
    description="Probe: a made-up (value, ±) result with a standard deviation added, to see whether Kaggle shows intervals. Never prompts the model.",
)
def fake_sd(llm) -> tuple[float, float]:
    return VALUE, INTERVAL


# On Kaggle the run carries the -m model's label, or the server's default in a
# push's creation run. Neither is prompted. Assigned, so the notebook doesn't
# display the run.
ON_KAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "").lower() == "batch"
run = fake_sd.run(kbench.llm if ON_KAGGLE else StandIn())
