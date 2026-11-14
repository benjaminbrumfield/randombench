# %%
# Phase 2 probe for D16 (PLAN.md 2.4b). Spends 12 Model Proxy calls on Gemini
# 3.1 Flash-Lite. In 2.4 it answered 42 to all 12 draws, with no temperature
# sent. This sends temperature=1.0 explicitly, with the same balanced schedule
# of prompts. If the answers spread out, Model Proxy's own default temperature
# is lower than the provider's.
#
# It also records the body of every HTTP request it sends, to show that the
# temperature goes out and how many requests the 12 draws really take. Only
# bodies are kept: the API key is in the headers.
#
# Local only: it imports randombench. Run it with the repo root on PYTHONPATH.
import json
import random

import pandas as pd

import kaggle_benchmarks as kbench
from kaggle_benchmarks.kaggle import model_proxy

from randombench.prompts import PROMPTS

MODEL = "google/gemini-3.1-flash-lite-preview"
TEMPERATURE = 1.0
N_DRAWS = 12

PROMPTS_BY_ID = {p.id: p for p in PROMPTS}

# The library sets support_temperature=False by default (model_proxy.py,
# line 102). Passing it here overrides that setdefault.
llm = model_proxy.ModelProxy(model=MODEL, api="openai", support_temperature=True)

SENT = []
llm.client._client.event_hooks["request"].append(
    lambda request: SENT.append(json.loads(request.content))
)


# %%
@kbench.task(name="draw", store_task=False)
def draw(llm, draw_index: int, prompt_id: str) -> dict:
    record = {"draw_index": draw_index, "prompt_id": prompt_id}
    try:
        record["reply"] = llm.prompt(
            PROMPTS_BY_ID[prompt_id].text, seed=None, temperature=TEMPERATURE
        )
    except Exception as e:
        record["error"] = f"{type(e).__name__}: {e}"
    return record


@kbench.task(
    name="randombench-temperature",
    description="D16 probe: 12 draws with temperature=1.0 sent explicitly.",
)
def temperature_probe(llm) -> dict:
    schedule = [p.id for p in PROMPTS] * (N_DRAWS // len(PROMPTS))
    random.shuffle(schedule)
    draws = draw.evaluate(
        llm=[llm],
        evaluation_data=pd.DataFrame({"draw_index": range(N_DRAWS), "prompt_id": schedule}),
        n_jobs=4,
        on_failure="continue",
    )
    replies = [run.result.get("reply") for run in draws.completed_runs]
    errors = [run.result["error"] for run in draws.completed_runs if "error" in run.result]
    return {"replies": ",".join(r for r in replies if r is not None), "errors": len(errors)}


# %%
run = temperature_probe.run(llm)

with open("sent.json", "w", encoding="utf-8") as f:
    json.dump(SENT, f, indent=1, ensure_ascii=False)
print(f"HTTP requests sent: {len(SENT)}")
