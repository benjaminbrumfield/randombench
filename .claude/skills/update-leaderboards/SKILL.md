---
name: update-leaderboards
description: >
  Bring both Kaggle Benchmarks, RandomBench and RandomBench Biases, up to date
  with data/. It pushes each scoring task whose file changed since Kaggle's
  version, runs the new versions on every model, asks the user to move both
  Benchmarks to them on the website, and checks every result against data/.
  Use after collecting draws, or when asked to update the leaderboards or push
  the scores to Kaggle. Scoring tasks never prompt a model, so it spends no
  Model Proxy quota.
---

# Update the leaderboards

Both Benchmarks show results from the ten scoring tasks, `tasks/scores/<slug>.py`, which embed the stored replies (D13). Run with `-m <slug>`, a scoring task scores that model's stored replies and never prompts it. New draws reach the Benchmarks in four steps:

1. Push each scoring task as a new version. A push's creation run scores only the server's default model.
2. Run each new version on every other model.
3. The user moves each Benchmark's task rows to the new versions on the website. A Benchmark pins each row to one version, and the CLI can't move it (4.9, 4.12). The runs come first, because "Update and Overwrite" replaces the row's results with the new version's runs, and a model without one would show a blank.
4. Check every result against `data/` with `python -m randombench.leaderboards`.

None of this spends quota (4.8, 4.12). It changes no file in the repo, so there's nothing to commit. It takes about an hour, nearly all of it in step 2: version 5's runs went from 21:21 to 22:30 UTC on 2026-11-19.

Start every shell command with `source .venv/Scripts/activate`, because shell state doesn't persist between calls. If a `kaggle` command prints "Authentication required", or `python -m randombench.models` or `python -m randombench.leaderboards` fails to sign in, the CLI's 12-hour token has expired: prefix the command with `export KAGGLE_API_TOKEN="$(kaggle auth print-access-token)" &&`.

## 1. Start from committed, tested files

1. If `git status --short` shows changes, stop and ask. A push would upload files that aren't committed.
2. Run `python -m pytest`. If a test fails, stop and report it. A scoring task that `python -m randombench.build` would change fails a test (D20), so a pass means the files hold every stored draw.
3. Note the daily `used` from `kaggle b quota --format json`.

## 2. Find the tasks to push

For each of the ten slugs, the files in `tasks/scores/`, read the current version and its `Created` time (UTC) from `PYTHONUTF8=1 kaggle b t status <slug> | head -4`. One command can loop over all ten. If `git log --since="<Created>Z" --oneline -- tasks/scores/<slug>.py` lists a commit, the file changed after that version, so push it. Otherwise Kaggle already has it.

The ten files embed the same replies, so new draws change all ten.

## 3. Push

Start every push at once: one Bash call per task, all in one message, each with `run_in_background: true`.

```bash
PYTHONUTF8=1 kaggle b t push <slug> -f tasks/scores/<slug>.py --wait
```

Ten pushes at once all completed in 4.8 and 4.12. Wait for every notification. Then check that `PYTHONUTF8=1 kaggle b t status <slug>` shows a version one higher than step 2's, Completed, with a single run: the creation run. It scores the server's default model, `gemini-3.8-flash` so far (D23). If a push failed or its version Errored, stop and report its output.

A new version of a public task is public too, so nothing needs publishing.

## 4. Run the new versions on every model

**Which models.** Run `PYTHONUTF8=1 python -m randombench.models`. The runs go to every folder in `data/` except:
- the creation run's model, which the push already scored;
- the models under Gone. Kaggle no longer lists them, and the CLI refuses a whole batch if one of its models isn't listed. Their draws stay in `data/`, but they drop off the Benchmarks in step 5.

If the report lists a model under Renamed, stop and tell the user. A scoring task finds replies by the `llm.name` its run gets, so the renamed model's run would find none and return False, which shows as FAIL. Models under New have no draws, so there's nothing to score.

**Batches.** Split the list, in `ls data` order, into batches of at most 11. In 4.6, 54 runs started at once, and the open-weight models' runs never began. In 4.12, batches of 11 all completed.

**Runs.** One task at a time. For each task, start one Bash call with `run_in_background: true` that chains its batches with `;`, not `&&`. A batch with an errored run exits 1, and `&&` would skip every batch after it. Start the chain with a fresh token. On 2026-11-20 the 12-hour token expired during favorite-number's second batch, and its last two batches never started. Fetch it once per task, not once per batch: fetching it before every command got a "429 Too Many Requests" from Kaggle.

```bash
export KAGGLE_API_TOKEN="$(kaggle auth print-access-token)"; \
PYTHONUTF8=1 kaggle b t run <slug> -m <a> -m <b> ... --wait 900; \
PYTHONUTF8=1 kaggle b t run <slug> -m <l> -m <m> ... --wait 900; ...
```

Each batch takes about a minute and a half, so a task takes about 6 minutes. Wait for the notification, then check that `PYTHONUTF8=1 kaggle b t status <slug>` shows a Completed run for every model on the list, plus the creation run. If a model has none, because its run errored, timed out, or never started, run it again on its own, once: `kaggle b t run <slug> -m <model> --wait 900`. A rerun costs nothing. If it still has no Completed run, stop and report the status.

After each task, write one line, so the user can follow the hour: `Task <k> of <n>: <slug> v<version>, <runs> runs completed.`

The runs don't need downloading. Step 6 reads their results from the Benchmarks.

**A finished task with runs missing.** If step 2 found nothing to push for a task, check its status anyway, and run any model on the list that has no Completed run on the current version. That finishes an update that stopped partway.

**Quota.** After the last task, read `kaggle b quota --format json` again. The daily `used` should equal step 1's. If it rose, say so. Scoring runs spend nothing (4.12), so something else did, perhaps a collect-draws loop in another session.

## 5. The user moves both Benchmarks to the new versions

First run `PYTHONUTF8=1 python -m randombench.leaderboards`. Until the Benchmarks move, Differ lists every model whose draws changed, which is expected. Note each model under Missing with "no row". It's new since the last update, and needs adding to the Benchmarks.

Then ask the user with AskUserQuestion. The dialog may be all they see, so put the steps in the question itself:
- On RandomBench, https://www.kaggle.com/benchmarks/benjaminbrumfield/randombench-overall, the RandomBench Score row's version label is red. Click it, then "Update and Overwrite".
- On RandomBench Biases, https://www.kaggle.com/benchmarks/benjaminbrumfield/randombench, do the same for each of the nine rows.
- Only if a model is new: add it to both with Add Models, by its display name. Moving a row to a new version doesn't add models (4.12).

Offer two options: done, so check the Benchmarks; or not now, so end with the Benchmarks still showing the old versions, and say so in the report.

## 6. Check

Run `PYTHONUTF8=1 python -m randombench.leaderboards`. It reads both Benchmarks and compares each model's result for each task, value and interval, with what `data/` gives. Its first two lines give each Benchmark's model count and the versions its tasks are pinned to, which should be step 3's. Its last line is "Up to date." or a count of problems:
- **Differ**, with a task still on its old version: the user hasn't moved that row. Ask them to, then check again.
- **Differ**, with every task on its new version: Kaggle scored different draws from those in `data/`. Stop and report it. Kaggle's results have matched exactly since 4.6.
- **Missing**, "no row": the model needs adding to that Benchmark on the website. Missing for one task: that model has no run on the version the row shows. Run it, then check again.
- **Extra**: a model on a Benchmark with no stored replies. Report it. Removing it is the user's call.
- **Not on Kaggle** is no problem. Kaggle no longer lists these models, so they're off the Benchmarks.

## Report

- The tasks pushed, each with its old and new version, the runs completed, and any reruns.
- The daily quota used before and after. They should match.
- The check's output, pasted as it is. If it's up to date, its first three lines, Not on Kaggle, and its last line are enough.
- That there's nothing to commit. The leaderboard page on GitHub Pages is separate: it publishes when `main` is pushed.

## Never

- Run or push `randombench-draws`. A run prompts a model, and the draws task isn't on either Benchmark. The collect-draws skill pushes it when `tasks/collect.py` changes.
- Edit `tasks/scores/`, which `python -m randombench.build` generates (D20), or push files that aren't committed.
- Publish a task, edit a Benchmark (only the user does that, on the website), or stage or commit anything.
