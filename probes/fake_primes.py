# %%
# Phase 1 probe (PLAN.md step 1.3). Throwaway: a scoring task. It holds
# made-up stored draws for two models, picks the draws of the model it's run
# with, and returns their share of primes with a 95% confidence interval. It
# never prompts a model, so running it spends no Model Proxy quota.
import math
import os

import kaggle_benchmarks as kbench

# %%
# Made-up stored draws, written into the task file the way the real scoring
# tasks might carry them (D13). They're keyed by the model's name as the task
# sees it. On Kaggle, -m gpt-5.4-nano-2026-03-17 arrives as
# "openai/gpt-5.4-nano-2026-03-17", the label its 1.2 run carried. Local runs
# use the python-random stand-in. The nano draws are exactly 40 percent
# primes, and the python-random ones are uniform, so the two scores differ.
STORED_DRAWS = {
    "openai/gpt-5.4-nano-2026-03-17": [
        43, 7, 11, 78, 42, 40, 82, 29, 91, 38, 20, 70, 37, 43, 87, 77, 9, 11, 4, 63,
        71, 6, 79, 99, 90, 76, 26, 17, 65, 61, 62, 75, 89, 91, 9, 74, 52, 68, 67, 10,
        48, 48, 11, 9, 88, 11, 39, 27, 15, 17, 39, 71, 57, 97, 98, 31, 13, 65, 25, 28,
        78, 17, 87, 51, 47, 44, 48, 90, 85, 28, 19, 73, 13, 79, 82, 41, 21, 47, 99, 46,
        83, 67, 26, 25, 6, 69, 25, 67, 13, 58, 88, 12, 19, 16, 94, 58, 55, 28, 59, 39,
        58, 92, 11, 5, 83, 25, 50, 1, 59, 18, 41, 92, 45, 43, 59, 79, 10, 19, 98, 95,
        72, 3, 93, 90, 28, 29, 45, 51, 13, 63, 23, 77, 69, 15, 54, 52, 64, 11, 11, 19,
        44, 3, 26, 67, 29, 62, 73, 96, 73, 3, 29, 16, 73, 86, 66, 24, 18, 27, 25, 48,
        87, 39, 90, 0, 84, 17, 17, 55, 85, 73, 20, 12, 22, 89, 43, 67, 37, 34, 8, 99,
        7, 30, 84, 73, 6, 21, 5, 60, 79, 26, 24, 68, 88, 77, 9, 43, 73, 23, 11, 42,
        50, 31, 31, 89, 92, 11, 60, 41, 64, 71, 23, 75, 32, 43, 79, 46, 51, 92, 59, 58,
        23, 18, 76, 29, 71, 63, 43, 56, 28, 47, 35, 23, 26, 37, 35, 2, 20, 21, 61, 37,
        85, 86, 76, 30, 23, 50, 73, 44, 95, 49, 13, 10, 75, 97, 37, 92, 1, 65, 92, 19,
        97, 71, 19, 94, 31, 49, 74, 6, 19, 59, 22, 23, 62, 21, 0, 11, 62, 89, 47, 10,
        49, 41, 51, 27, 12, 5, 84, 72, 83, 72, 13, 44, 10, 6, 72, 30, 53, 73, 4, 85,
        59, 79, 33, 37, 28, 69, 55, 40, 97, 32, 5, 21, 0, 3, 16, 7, 61, 54, 37, 16,
        47, 77, 60, 98, 23, 67, 19, 79, 54, 46, 54, 0, 38, 98, 19, 13, 83, 3, 74, 10,
        29, 71, 89, 89, 75, 93, 47, 67, 20, 35, 46, 37, 53, 83, 23, 49, 78, 0, 21, 81,
        2, 33, 2, 91, 84, 83, 45, 10, 40, 72, 73, 67, 66, 33, 70, 10, 61, 48, 90, 98,
        20, 54, 92, 45, 3, 2, 98, 13, 55, 62, 76, 5, 95, 71, 31, 94, 39, 54, 83, 53,
        18, 47, 41, 92, 56, 4, 99, 84, 61, 65, 78, 38, 24, 28, 89, 40, 31, 89, 91, 64,
        9, 96, 27, 43, 11, 85, 38, 2, 12, 22, 6, 57, 37, 30, 35, 93, 13, 54, 20, 69,
        91, 73, 28, 81, 54, 16, 75, 3, 89, 61, 57, 62, 5, 74, 5, 0, 29, 37, 23, 17,
        30, 25, 19, 47, 53, 53, 5, 17, 18, 78, 28, 23, 1, 75, 98, 88, 68, 10, 52, 61,
        22, 17, 7, 99, 4, 73, 75, 89, 66, 88, 81, 75, 71, 11, 53, 75, 61, 79, 1, 1,
        29, 77, 45, 81, 71, 3, 68, 67, 11, 81, 32, 6, 61, 62, 19, 61, 66, 43, 84, 18,
        35, 91, 96, 11, 17, 79, 65, 59, 13, 83, 33, 23, 76, 29, 20, 43, 10, 3, 32, 37,
        12, 70, 2, 83, 3, 26, 28, 10, 43, 37, 14, 53, 18, 76, 73, 24, 23, 83, 8, 53,
        94, 47, 11, 98, 84, 4, 17, 52, 74, 39, 43, 14, 31, 86, 93, 95, 74, 95, 89, 97,
        67, 37, 66, 45, 86, 61, 27, 26, 47, 42, 10, 68, 43, 27, 29, 21, 30, 79, 55, 43,
    ],
    "python-random": [
        66, 54, 19, 81, 78, 18, 5, 41, 18, 82, 43, 13, 88, 71, 75, 90, 65, 57, 43, 65,
        72, 46, 0, 26, 1, 74, 66, 95, 53, 64, 38, 49, 19, 32, 65, 4, 6, 74, 52, 90,
        28, 41, 20, 14, 7, 14, 78, 92, 60, 88, 22, 86, 16, 52, 20, 54, 73, 33, 90, 27,
        88, 95, 67, 65, 99, 83, 0, 5, 52, 28, 87, 59, 91, 92, 30, 96, 56, 35, 84, 67,
        37, 26, 85, 62, 7, 52, 81, 91, 34, 65, 14, 85, 45, 84, 41, 82, 57, 57, 30, 40,
        16, 42, 89, 99, 29, 83, 33, 96, 24, 40, 49, 22, 43, 68, 43, 59, 83, 4, 59, 50,
        47, 90, 50, 72, 15, 76, 62, 5, 15, 12, 67, 70, 89, 63, 12, 29, 5, 11, 80, 58,
        11, 58, 54, 69, 35, 15, 3, 71, 27, 43, 63, 80, 5, 69, 39, 89, 42, 15, 77, 34,
        22, 75, 16, 69, 71, 51, 27, 16, 22, 73, 93, 88, 22, 32, 64, 74, 71, 66, 63, 25,
        40, 33, 57, 79, 14, 45, 12, 46, 7, 30, 42, 31, 3, 86, 74, 14, 43, 26, 3, 67,
        68, 82, 1, 36, 6, 14, 31, 22, 5, 56, 18, 42, 86, 8, 68, 89, 11, 92, 82, 49,
        43, 73, 27, 70, 43, 53, 20, 69, 6, 38, 45, 89, 28, 63, 0, 74, 6, 16, 57, 13,
        83, 11, 9, 32, 26, 47, 86, 61, 60, 38, 19, 43, 82, 64, 16, 97, 85, 82, 44, 68,
        12, 57, 86, 13, 58, 81, 14, 46, 68, 6, 59, 74, 62, 34, 95, 73, 84, 2, 31, 32,
        6, 61, 93, 61, 59, 90, 38, 29, 35, 65, 93, 5, 36, 58, 19, 40, 79, 44, 75, 29,
        2, 46, 65, 25, 16, 86, 26, 89, 64, 14, 15, 35, 77, 69, 26, 95, 10, 69, 62, 35,
        75, 38, 18, 53, 48, 76, 91, 48, 38, 17, 14, 42, 9, 33, 1, 20, 71, 61, 77, 68,
        15, 60, 26, 65, 27, 49, 18, 46, 31, 92, 21, 14, 61, 0, 49, 46, 63, 35, 31, 68,
        30, 6, 33, 65, 22, 5, 87, 28, 19, 49, 72, 86, 42, 61, 58, 66, 95, 92, 67, 29,
        57, 32, 71, 99, 18, 18, 38, 36, 72, 68, 10, 8, 22, 64, 18, 78, 34, 37, 77, 33,
        83, 39, 97, 66, 94, 60, 65, 48, 19, 6, 79, 99, 46, 11, 59, 71, 44, 52, 16, 36,
        74, 96, 40, 48, 40, 65, 64, 48, 93, 87, 12, 58, 36, 83, 99, 80, 77, 56, 87, 72,
        56, 30, 0, 76, 79, 1, 22, 77, 8, 35, 89, 20, 95, 19, 17, 21, 87, 98, 10, 6,
        26, 6, 50, 73, 57, 84, 42, 46, 48, 3, 24, 50, 50, 78, 21, 83, 49, 44, 82, 14,
        88, 81, 20, 10, 47, 87, 56, 54, 76, 50, 67, 75, 17, 28, 26, 55, 42, 11, 14, 98,
        38, 93, 4, 28, 64, 75, 78, 4, 9, 27, 78, 32, 2, 76, 23, 74, 48, 69, 85, 20,
        64, 95, 49, 61, 47, 42, 57, 5, 88, 77, 56, 13, 74, 86, 15, 9, 86, 5, 39, 74,
        96, 74, 48, 12, 6, 92, 83, 42, 50, 54, 75, 13, 36, 58, 42, 65, 99, 64, 57, 52,
        75, 47, 66, 6, 99, 36, 30, 74, 41, 43, 87, 64, 22, 62, 53, 2, 75, 42, 77, 27,
        28, 19, 44, 61, 2, 59, 19, 34, 89, 41, 51, 64, 17, 67, 89, 27, 34, 95, 12, 87,
    ],
}

PRIMES = {n for n in range(2, 100) if all(n % d for d in range(2, math.isqrt(n) + 1))}


# %%
class PythonRandom(kbench.LLMChat):
    """Stand-in model that labels a local run (PLAN.md 1.1b).

    It can't answer prompts: one sent here is a bug, so it fails loudly.
    """

    def __init__(self):
        super().__init__(name="python-random")

    def invoke(self, *args, **kwargs):
        raise RuntimeError("python-random is a stand-in model and can't be prompted")


# %%
@kbench.task(
    name="randombench-fake-primes",
    description="Phase 1 probe: share of primes in made-up stored draws, with a 95% confidence interval. Never prompts the model.",
)
def fake_primes(llm) -> tuple[float, float]:
    draws = STORED_DRAWS.get(llm.name)
    if draws is None:
        # The creation run lands here: every push runs the task once against
        # the server's default model, which has no stored draws (D12). Raising
        # failed the push (version 1), so it returns False: a result, but not
        # a score. The library only warns that it doesn't match the annotation.
        return False
    n = len(draws)
    share = sum(d in PRIMES for d in draws) / n
    # 95% interval for a share, by the normal approximation:
    # share ± 1.96 × √(share × (1 − share) / n). Kaggle takes the ± half-width.
    half_width = 1.96 * math.sqrt(share * (1 - share) / n)
    return share, half_width


ON_KAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "").lower() == "batch"
if ON_KAGGLE:
    # Names only, never values: some hold credentials. Comparing the creation
    # run's list with the -m run's shows whether the environment marks a
    # creation run (D12).
    print("Environment variables:", " ".join(sorted(os.environ)))
    # These four hold model names, not credentials, so their values print.
    for name in ["LLM_DEFAULT", "LLM_DEFAULT_EVAL", "LLMS_AVAILABLE", "KAGGLE_AGENT_LLM"]:
        print(f"{name}={os.environ.get(name)!r}")
# Assigned so the notebook doesn't display the run. The display formats a
# tuple[float, float] result as a pair and would crash on False, after the
# run file is written. The local console UI has the same flaw.
run = fake_primes.run(kbench.llm if ON_KAGGLE else PythonRandom())
