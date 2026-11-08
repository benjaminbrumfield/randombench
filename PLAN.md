# RandomBench plan

This is how RandomBench gets built, one small step at a time. Each step is one commit and comes with something to learn. Steps marked **quota** make LLM calls through Kaggle's Model Proxy. They wait for an explicit go-ahead, and each one states its call count up front.

## Goal

Measure how well LLMs generate random integers in [0, 99]. Each model answers hundreds of independent single-turn prompts, and its answers are checked for known biases: favoring primes, round numbers, and so on. Each check yields a sub-score in [0, 1], and a headline score combines them.

## Decided

- **Range:** integers 0–99.
- **Calls:** single-turn, in a fresh conversation each time. No tools, and the model never sees its earlier answers.
- **Prompts:** a fixed set of at least six, stored in the repo, each with a stable id and an author. Each draw uses a prompt picked at random.
- **Volume:** the target is 600 draws per model per run. Failed calls are recorded, not dropped, and a run that stops early keeps what it collected.
- **Raw data:** every raw response is stored. Parsed values and scores are always recomputed from the raw text, so a new bias check or a new parsing rule never requires calling the models again.
- **Invalid answers:** the rate of invalid answers is its own sub-score.
- **Temperature:** the provider's default. This was not our choice: kaggle-benchmarks 0.6.2 never sends temperature through Model Proxy (`kaggle_benchmarks/kaggle/model_proxy.py`, line 101).
- **Seed:** `seed=None`, so providers sample freshly, as in normal API use. The library's default seed is 0, and it is sent to every non-Google model. A backend that honors seeds could then return the same number every time, which would look like a bias. It's unverified whether every backend accepts a null seed. If one rejects it, the fallback is a distinct seed on every call.
- **Models:**
  - gpt-5.4-nano
  - gemini-3.1-flash-lite-preview
  - gemini-3.5-flash-lite
  - claude-haiku-4-5
  - gemma-4-26b-a4b-it
  - gpt-oss-120b

  Local development uses the first two, which are available both locally and on Kaggle.

## Bias checks (planned, not implemented)

A perfect generator gives each value probability 1/100, so the expected rates are:

| Check | What it measures | Expected from a perfect generator |
|---|---|---|
| Invalid answers | Share of replies that aren't a single integer in [0, 99] | 0% |
| Uniformity | How far the whole distribution is from uniform | Calibrated (see below) |
| Primes | Share of answers that are prime (2, 3, 5, …, 97: 25 values) | 25% |
| Round numbers | Share of multiples of 10, and separately of 5 | 10% (20%) |
| Favorite number | Share of answers taken by the single most common value | Calibrated |
| Edges | Share of answers in 0–9 or 90–99 | 20% |
| Repeated digits | Share of 11, 22, …, 99 | 9% |
| Last digit | How far the last-digit frequencies are from 10% each (7 is the usual suspect) | 10% each |

"Calibrated" means even a true random generator won't hit the ideal value exactly in 600 draws. We'll measure what Python's generator scores at the same sample size and compare models against that.

Some checks overlap. For example, 7, 17, 37, 47, 67 and 97 are all primes ending in 7. This matters when combining sub-scores into the headline.

## Steps

### Phase 0: groundwork

- [x] **0.1** Commit the environment fixes: the `hishel` pin (`91ef7a4`) and `PYTHONUTF8=1` in `CLAUDE.md` (`38cfc5f`).
- [x] **0.2** Commit this plan (`7c8488d`).

### Phase 1: learn the platform without spending quota

- [ ] **1.1 Fake-draw task, run locally.** This is a throwaway task whose "draws" come from Python's `random`, not from an LLM:
  - A sub-task runs 600 times through `.evaluate(on_failure="continue")`, with a few failures injected on purpose.
  - A parent task returns a dict of made-up sub-scores.

  *Learn:* what a `.run.json` contains, where sub-task results end up, and how failures are recorded. It makes no LLM calls.
- [ ] **1.2 Fake-draw task, on Kaggle.** Push and run the same task, plus a variant that crashes halfway through. `kaggle b t run` needs a model, but the task never calls it. Because this is a server run, it waits for a go-ahead anyway.

  *Learn:*
  - how a dict result appears on the leaderboard (settles D2)
  - what `kaggle b t download` returns
  - whether a run that crashes partway keeps anything (informs D3)

### Phase 2: from one call to a dataset

- [ ] **2.1 Prompts.** Add the six or more prompts, each with a stable id and an author. Rule: editing a prompt's text gives it a new id, so every stored draw points at the exact text it answered.
- [ ] **2.2 Parser and tests.** Write a function that turns a raw reply into an integer or "invalid", with unit tests on sample replies: `47`, `**47**`, `Sure! 47`, `47 or 52`, `100`, `3.5` and a refusal. No quota. Settles D4.
- [ ] **2.3 One real draw.** **Quota: 2 calls**, one per local model, with `seed=None`.

  *Learn:* whether a null seed is accepted, and where the raw reply, the reasoning trace and the token usage appear in the run file.
- [ ] **2.4 Small collection run.** **Quota: 24 calls**, 12 on each local model. This is the real collection task at a small size. Settles D9.

  *Learn:* how parallel calls and retries behave, and whether each draw's record is complete.
- [ ] **2.5 Extract and store.** Write a script that turns run files into per-draw records under `data/` and merges runs without duplicates. Each record holds:
  - model, run id, task version and timestamp
  - prompt id and author
  - raw reply, reasoning trace and token usage
  - the error, for failed calls

  No quota. Settles D5.

### Phase 3: scoring, entirely offline

These steps use only stored draws and simulated random numbers. No quota.

- [ ] **3.1 Share code between the task and the analysis.** Settles D6.
- [ ] **3.2 Calibration harness.** Score Python's random generator many times at N = 600, and at smaller N for runs that stop early, to see what a perfect generator scores.
- [ ] **3.3 Score mapping.** Decide together how a measured bias becomes a sub-score in [0, 1]. Base it on how big the bias is (effect size), not on p-values. A p-value gets more extreme as N grows, so more draws would make every model look worse. Settles D7.
- [ ] **3.4 Bias checks,** one commit each, in the order of the table above.
- [ ] **3.5 Headline score.** Decide together. Harmonic mean is the starting candidate. Watch for two problems: it drops to 0 if any single sub-score is 0, and overlapping checks penalize the same bias twice. Settles D8.
- [ ] **3.6 Wire the scores into the task's result.**

### Phase 4: real runs on Kaggle

- [ ] **4.1 Pilot.** **Quota: 120 calls**, 20 on each of the six models.

  *Learn:* whether all six accept a null seed, how many tokens a draw costs (reasoning models may think at length), and the invalid-answer rates. This is the go/no-go for full runs.
- [ ] **4.2 Full runs.** **Quota: 3,600 calls**, 600 on each of the six models. Download, extract and commit the data.
- [ ] **4.3 Top-ups** for any model short of 600, if D3 chooses to accumulate draws across runs.

### Phase 5: release

- [ ] **5.1 README.** Covers the method, the results, and the limitations, such as no temperature control and reliance on provider defaults.
- [ ] **5.2 Publish.** `kaggle b t publish` makes the task public and can't be undone from the CLI, so it needs an explicit go-ahead.

## Open decisions

| ID | Question | Leaning | Settled in |
|---|---|---|---|
| D1 | Reasoning: use each model's default, or force it on or off? | The provider default. Not every model can turn reasoning off: gpt-oss-120b offers only low, medium and high. Forcing a setting would treat models unevenly either way. The default also matches how people use each model, and how we already handle temperature. Reasoning traces and token counts are stored for later analysis. Watch the cost in the pilot. | Before 2.3 |
| D2 | Leaderboard: one task that returns a dict of sub-scores plus the headline, or one task per sub-score? | One dict task, if Kaggle displays it well | 1.2 |
| D3 | Runs that stop early: score each run on what it collected and merge all draws offline? Or have later runs top up to 600 from stored draws, through an attached Kaggle dataset? | Start with self-contained runs, and revisit after seeing real failures | 1.2, 4.1 |
| D4 | How strict is parsing? | Lenient on formatting (bold, punctuation), strict on content (exactly one integer) | 2.2 |
| D5 | Data layout: one committed file per run under `data/`? How large do reasoning traces make it? | — | 2.5 |
| D6 | A push uploads a single `.py` file, but the offline analysis needs the same parsing and scoring code. Options: a source module plus a script that copies it into the pushed task file; the task file as the only copy; or code in an attached Kaggle dataset. | A source module plus a copy script, with a test that the pushed file is up to date | 3.1 |
| D7 | How does a measured bias become a sub-score in [0, 1]? | Based on effect size | 3.3 |
| D8 | How is the headline score computed? | Harmonic mean, still to be decided | 3.5 |
| D9 | Prompt assignment: pick a prompt independently at random for each draw, or shuffle a balanced schedule with each prompt used exactly 100 times? | A balanced shuffle. The order is still random, and equal counts make prompts easier to compare. | 2.4 |
