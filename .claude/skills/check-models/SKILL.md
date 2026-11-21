---
name: check-models
description: >
  Check whether Kaggle Benchmarks has added or dropped models, and whether
  RandomBench can use a new one. Use when asked about new models on Kaggle,
  whether a model can join RandomBench, or before planning new draws. It only
  reads, and spends no Model Proxy quota.
---

# Check Kaggle's models

This answers three questions. Which models does Kaggle list that RandomBench doesn't cover yet? Could each one be used? Which covered models has Kaggle dropped? It only reads. Adding a model spends quota, so the skill ends with a proposal and waits for a yes.

## 1. Run the comparison

```bash
source .venv/Scripts/activate
python -m randombench.models
```

It asks Kaggle's API for the list behind `kaggle b t models`, and compares it with `data/` and `MODELS` in [tasks/collect.py](../../../tasks/collect.py). If it fails with an authentication error, the CLI's 12-hour token has expired. Get a fresh one in the same command: `export KAGGLE_API_TOKEN="$(kaggle auth print-access-token)" && python -m randombench.models`.

## 2. Read the report

RandomBench covers every model Kaggle lists except the server's default model (D23) and models that Model Proxy doesn't serve (D24). That's the rule `MODELS` has followed since the flagship models were added.

**New.** The candidates. A new model can be used if it has a Model Proxy name (the third column) and isn't also under Not allowed. That name is its exact `MODELS` entry, not a guess. On 2026-11-18, all 35 stored models' records held the name Kaggle gives. The report can't confirm the rest, which only a pilot shows:
- Model Proxy serves it. It can allow a model whose every call gets a 404 (D24).
- It accepts `seed=None`, and its replies parse ("Seed" in PLAN.md).
- What a call costs.

"Already in MODELS" means the model was added, but no draws are stored: a pilot is pending, or its run wasn't extracted.

**Gone.** Kaggle no longer lists it. Its draws stay in `data/`, so it stays on the page and in the findings, which are built from `data/` alone. It's off the Benchmarks, though. `kaggle b t run -m` refuses a model Kaggle doesn't list, and refuses the whole batch with it (the update-leaderboards skill's step 4), so no run can score it, and it dropped off at the first update after Kaggle stopped listing it. `python -m randombench.leaderboards` lists it under Not on Kaggle. A row with "(no data/ folder)" is a `MODELS` entry that names no listed model: a typo, or a model Kaggle dropped before its first run. A run on it makes no draws (D12).

Whether to take a gone model out of `MODELS` is the user's decision: raise it, but don't edit anything. Tell them what it changes:
- Nothing anyone can see. `data/`, the page, the findings, the scoring tasks, and the Benchmarks don't read `MODELS`.
- What happens if Kaggle lists it again under the same name. Still in `MODELS`, it goes straight back into the collect-draws loop, first if it has the fewest replies. Taken out, it shows here under Left out, as news, and needs the user's yes like a new model.
- `MODELS` follows its rule again: every model Kaggle lists, less the server's default and the models Model Proxy won't call.
- The cost is small. It's an edit to `MODELS` and its comment, and to the count in `tests/test_collect.py`, and the collect-draws skill then pushes a new `randombench-draws` version, which spends no quota. The tally gives the model's reason as "not in MODELS" instead of "Kaggle doesn't list it", and this report's Gone row drops "in MODELS".

**Left out.** By design, this holds exactly one model: `gemini-3.8-flash`, the server's default (D23). Any other model here is news. Its draws are stored, but `MODELS` doesn't list it.

**Renamed.** Kaggle now gives a slug a different `llm.name`. `MODELS` must hold the new name, or a run makes no draws (D12). Scoring, the scoring tasks, and the page also key draws by `llm.name` (PLAN.md 1.3). So new draws under the new name would count as a separate model. Raise this with the user before any run.

**Not allowed.** Model Proxy won't call it. A new model here can't be used. A covered model here can't make more draws, but its stored draws still count.

## 3. Tell the user

Report briefly. For each new model, give its slug, its Model Proxy name, its release date, and whether it looks usable. Then report anything in Gone, Renamed, or Not allowed, and anything unexpected in Left out. Skip groups that hold only what's expected.

## 4. Propose, then stop

For each usable new model, propose this path, with its call count. Each run makes a random number of draws, from 100 to 300 (`MIN_DRAWS` and `MAX_DRAWS` in `tasks/collect.py`), so there's no separate pilot size:

1. **First run, 100 to 300 calls per model, plus any retries.** Add the Model Proxy name to `MODELS`, push `randombench-draws`, and run it with `-m <slug>`. Run at most two collection runs at once (CLAUDE.md).
2. **Check its draws** after `python -m randombench.extract`. A 404 means Model Proxy doesn't serve the model (D24). A refused call still counts as one request, so the log's retry count can't show it (4.11).
3. **More runs.** Each costs another 100 to 300 calls. The first run's stored costs give an estimate. Check `kaggle b quota` first.

The collect-draws skill can do all three: it asks about each usable new model, and a model it adds goes first, with no replies yet.

Then wait for the user's yes before you edit, push, or run anything (CLAUDE.md, "Ask before any LLM call").

The report can't rule out one risk. A push's creation run uses the server's default model, and the API doesn't say which model that is. If Kaggle made a model in `MODELS` its default, every push would draw 100 to 300 times from it. `PYTHONUTF8=1 kaggle b t status randombench-draws` shows the current version's creation run. It's the earliest run, starts about a minute after "Created", and ends in the same second it starts. Name its model in the proposal, and say that a push spends quota if the default has since changed to a model in `MODELS`.

Once a new model's draws are stored, the follow-up step touches:
- `MODELS` and its comments in `tasks/collect.py`, and the count in `tests/test_collect.py`.
- `DISPLAY_NAMES` and `RELEASED` in `randombench/page.py`. A test fails when a stored model has no entry in either. `RELEASED`'s dates come from the web, not from Kaggle.
- `python -m randombench.build`. Afterwards, `tests/test_findings.py` may flag a finding to rewrite (D29).
- The update-leaderboards skill, which pushes the rebuilt scoring tasks and runs each one on every model, the new one included. Scoring tasks never prompt a model. The user then adds the new model to both Benchmarks on the website.

## Never

- Edit, push, run, publish, or delete anything as part of this check without the user's yes.
- Treat Model Proxy allowing a model as proof that the model works (D24).
- Add the server's default model to `MODELS` (D23).
