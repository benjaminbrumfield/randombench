# %%
# Phase 2 probe (PLAN.md step 2.3). Spends 2 Model Proxy calls: one draw on
# each local model, with seed=None and each provider's default reasoning (D1).
# It shows whether a null seed is accepted, and where the raw reply, the
# reasoning trace and the token usage land in the run file.
#
# Local only: it imports randombench, which a push wouldn't upload. Run it
# with the repo root on PYTHONPATH.
import kaggle_benchmarks as kbench

from randombench.parsing import parse_reply
from randombench.prompts import PROMPTS

PROMPT = PROMPTS[0]  # plain
MODELS = ["openai/gpt-5.4-nano-2026-03-17", "google/gemini-3.1-flash-lite-preview"]


# %%
@kbench.task(
    name="randombench-one-draw",
    description="PLAN.md 2.3 probe: one real draw with seed=None.",
)
def one_draw(llm) -> dict:
    reply = llm.prompt(PROMPT.text, seed=None)
    return {
        "prompt_id": PROMPT.id,
        "reply": reply,
        "value": parse_reply(reply),
        "reasoning": kbench.last_reasoning_traces(),
    }


# %%
for name in MODELS:
    run = one_draw.run(kbench.llms[name])
