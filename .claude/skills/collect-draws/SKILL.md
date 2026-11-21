---
name: collect-draws
description: >
  Collect more draws for RandomBench in a loop. It checks Kaggle for new
  models, then runs one batch of 100 to 300 draws on the model with the fewest
  replies, reports how that batch's answers fall and every model's count, and
  repeats until every model a run can draw from has reached the reply cap or
  less than $4 of the day's quota is left. Use when asked to collect or top up
  draws, or to spend the day's quota. It spends Model Proxy quota: invoking it is the user's yes
  for its runs.
---

# Collect draws

Each run of `randombench-draws` makes a random number of draws, from 100 to 300 (`MIN_DRAWS` and `MAX_DRAWS` in [tasks/collect.py](../../../tasks/collect.py)). This skill runs one batch at a time, always on the model with the fewest replies, so the models even out over time. A model with 800 replies or more (`CAP` in [randombench/tally.py](../../../randombench/tally.py)) gets no more batches, so once every model a run can draw from has reached the cap the loop ends, however much quota is left.

CLAUDE.md says to ask before any LLM call. The user invoking this skill is that yes, for as many runs as the loop below makes. Adding a model is not covered, so step 0 asks.

Start every shell command with `source .venv/Scripts/activate`, because shell state doesn't persist between calls. If a `kaggle` command prints "Authentication required", the CLI's 12-hour token has expired: prefix the command with `export KAGGLE_API_TOKEN="$(kaggle auth print-access-token)" &&`. Fetch a token only then. On 2026-11-21 one fetch was enough: the commands after it signed in without the prefix, while fetching a token before every command got "429 Too Many Requests" from Kaggle.

## Before the loop: no missed runs, and Kaggle runs the committed task

1. If `git status --short` shows uncommitted changes, stop and ask. A push would upload work that isn't committed yet, and the user's commit after the loop would mix it in with this step's changes.
2. Collect any run the last loop didn't, before any push. A run can outlast the loop's 90-minute wait: on 2026-11-20, Grok 4.6's ran 7.5 hours under xAI's rate limits, and the loop stopped without it. `PYTHONUTF8=1 kaggle b t download randombench-draws -o results` fetches each model's latest run on the current version, skipping those already in `results/`, so it takes seconds. A push makes a new current version, so after one, the download would miss the old version's runs. Then run `python -m randombench.extract results/randombench-draws`. If `git status --short data/` shows a new file, it's a run the last loop missed. Report it as step 4 of the loop does, labeled as recovered rather than as a batch, before batch 1. If `kaggle b t status randombench-draws` shows a run that hasn't finished, stop and tell the user: a batch on that model would run beside it.
3. Read the current version's `Created` time (UTC) from `PYTHONUTF8=1 kaggle b t status randombench-draws`. If `git log --since="<Created>Z" --oneline -- tasks/collect.py` lists a commit, the version is older than the file, so push it: `PYTHONUTF8=1 kaggle b t push randombench-draws -f tasks/collect.py --wait`. An unneeded push does no harm: it spends no quota and makes a new version.
4. After any push, check its creation run, which is the earliest run in `kaggle b t status randombench-draws`. It should be `gemini-3.8-flash`, the server's default model, which `MODELS` leaves out (D23). If it's a model in `MODELS`, the push drew 100 to 300 times from it. Stop and tell the user.

## 0. New models, once at the start

Run `python -m randombench.models`, and read its report as the check-models skill does ([its step 2](../check-models/SKILL.md)).

- **Usable new model** (it has a Model Proxy name and isn't under Not allowed): ask the user with AskUserQuestion whether to add it. Give its name, slug, and release date. Say that its first batch makes 100 to 300 calls, and that this batch is its pilot. If the user says yes:
  - Add its Model Proxy name to `MODELS` with a comment giving its release date, and update the count in `tests/test_collect.py`.
  - Add it to `DISPLAY_NAMES` in `randombench/page.py`, with Kaggle's display name and its maker, and to `RELEASED` there, with its release date. Look the date up on the web, by the rule in the comment above `RELEASED`. Don't take Kaggle's, which can be missing or off by a day.
  - Run `python -m pytest`, then push as above. This is the one push of uncommitted work: the commit after the loop holds the change. It has no replies, so the first batch goes to it.
- **Renamed**: tell the user before running anything. New draws under the new name would count as a separate model. The table below marks the model as never drawn until `MODELS` changes.
- **Gone**: mention it. The table already skips it. Taking it out of `MODELS` is the user's decision, and [check-models](../check-models/SKILL.md) says what that changes: nothing on the page or the Benchmarks.

## The loop

Number the batches from 1. For each one:

1. **Check the quota.** Run `kaggle b quota --format json`. If the Daily or the Monthly `remaining` is under $4, the loop ends. Otherwise note the daily `used`. One 300-draw batch on the costliest model, Gemini 4 Neon, would cost about $0.82 at 2026-11-19 prices, so a batch can't overrun the floor.
2. **Pick.** Run `PYTHONUTF8=1 python -m randombench.tally`. Its last line, `Next: <slug>`, names the model the run goes to: the one with the fewest replies among the models a run can draw from, if it's under the cap. The table marks a model at the cap `done`. `Next: none` ends the loop.

   From batch 2 on, the previous batch's step 4 has already run both commands, so use its output for steps 1 and 2 instead of running them again.
3. **Run one batch.** Run `PYTHONUTF8=1 kaggle b t run randombench-draws -m <slug> --wait 5400` with `run_in_background: true`, and wait for the notification. Don't pipe it through `tail` or `head`, which hold its output until it ends. Then:
   - Check that `PYTHONUTF8=1 kaggle b t status randombench-draws -m <slug>` shows the new run as Completed. If the wait timed out, or the run errored, stop the loop and report its status. Don't start the run again: a second `run` makes a second batch. A run that's still going when the wait times out is collected by step 2 of "Before the loop", the next time this skill runs. If the `run` command printed "Authentication required" instead, the token expired while it waited, and only its polling failed. Check the status with a fresh token: if the run is Completed, carry on as usual.
   - `PYTHONUTF8=1 kaggle b t download randombench-draws -m <slug> -o results`
   - `python -m randombench.extract results/randombench-draws`. It prints one line per run it finds: `<slug> <run_id> (v<version>): <n> draws, <e> errors -> <file>`. The new run is the one with the highest run id for `<slug>`. Extracting older runs again rewrites the same files, so nothing is counted twice.
   - Errors are refused calls. If every draw errored, Model Proxy may not serve the model (D24), or the quota ran out. Stop the loop and tell the user, after step 4's report. Taking a model out of `MODELS` is the user's decision.
4. **Report the batch.** Run these three in one command: `PYTHONUTF8=1 python -m randombench.batch <file>`, with the new run's file from extract's line, then `kaggle b quota --format json`, then `PYTHONUTF8=1 python -m randombench.tally`.

   The user can't see command output, so the report must be a message you write yourself. Write it before any tool call of the next batch, and after the last batch too. Never skip it or fold it into a later message: it's how the user follows the loop. It holds, in this order:
   - One line: `Batch <k>: <model>, <n> draws, <e> errors, $<spent>. $<left> of today's quota left.` The spend is the change in the daily `used` since step 1. If the daily quota refilled during the batch (its `refillAt` moved a day), that change is wrong. Sum the run's `input_cost_nanodollars` and `output_cost_nanodollars` from its file instead, as for a recovered run.
   - The batch command's output, pasted as it is. It gives the batch's summary, how often each value came up, the answers by decade, by last digit, and by prompt, and the batch's RandomBench Score beside the model's over all its runs.
   - Two or three sentences on what stands out, each backed by a number from the output: a favorite that takes a large share, decades or last digits the model crowds or avoids, a prompt that answers unlike the others, and whether this batch's score and favorite agree with the model's earlier runs.
   - The tally's table, pasted as it is.

   Then go back to step 1, using this step's quota and tally.

Run one batch at a time. CLAUDE.md allows two collection runs at once, but the loop needs each batch's count before it picks the next model.

## After the loop

1. `python -m randombench.build` rebuilds the scoring tasks and the page from `data/`.
2. `python -m pytest`. A failing `test_finding_` means the new draws broke a finding on the page: rewrite the finding in `randombench/page.html`, then its test (D29). A finding marked `xfail` is waiting for more draws. If the models it waits for have caught up with the others, rewrite it and remove the mark. Quote each old and new wording in the report, so the user can check them.
3. End with the one-sentence commit message. Don't stage or commit: the user does both.

Report the batches run, the draws and replies added per model, the day's spend, the quota left, and why the loop ended: the quota floor, every model a run can draw from at the cap, or a run that failed. Say that the rebuilt scoring tasks aren't pushed, so the Benchmarks don't show the new draws yet. The update-leaderboards skill updates them, once the user has committed. It spends no quota, but takes about an hour. The page publishes when `main` is pushed.

## Never

- Run a model that isn't the tally's `Next`, or two batches at once.
- Add a model to `MODELS` without the user's yes in step 0.
- Add the server's default model to `MODELS` (D23).
- Start a batch with less than $4 of the daily or monthly quota left.
- Push the scoring tasks, publish anything, or stage or commit anything.
