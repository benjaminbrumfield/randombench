# RandomBench

RandomBench measures how well LLMs generate random integers from 0 to 99 and scores their answers for biases, such as favoring primes or round numbers. It's built with [`kaggle_benchmarks`](https://github.com/Kaggle/kaggle-benchmarks) and runs on Kaggle Benchmarks.

[PLAN.md](PLAN.md) holds the design, the decisions and the build steps. Read it before starting work, and record decisions there.

## How we work

- **One step at a time.** Each PLAN.md step is one small commit. This is a learning project, so explain what each step shows, and raise open decisions instead of settling them quietly.
- **Ask before any LLM call.** Local task runs, probe scripts and `kaggle b t run` all spend Model Proxy quota. Pushing `randombench-draws` doesn't: its creation run uses Gemini 3.8 Flash, which `MODELS` leaves out (PLAN.md D23). Adding Gemini 3.8 Flash back would make every push spend `N_DRAWS` calls. Say which models and how many calls, then wait for a yes. Read-only commands such as `kaggle b t models` are fine.

## Writing tasks

The `kaggle-benchmarks` skill covers the general rules. On top of those:

- Keep the slug in `@kbench.task(name=...)` stable, so a push makes a new version rather than a new task.
- A push uploads only the task file, so everything the task needs on Kaggle (prompts, parsing) must be in it. A task never imports `randombench`. It holds a copy of each module it uses: put an empty `# --- begin randombench/<module>.py ---` / `# --- end randombench/<module>.py ---` pair where the module goes, then run `python -m randombench.sync` (PLAN.md D6).
- The ten scoring tasks in `tasks/scores/` are generated (PLAN.md D20). Never edit them. Edit `tasks/templates/score.py`, or `SCORING_TASKS` in `randombench/build.py`, and run `python -m randombench.build`. The files embed the stored replies in `data/`, so rebuild after extracting new runs too.
- Never write a comment that starts with `# %` or `# !`, or a bare `#%%`. When the CLI converts the file to a notebook, jupytext turns these into magics, shell commands or cell breaks. `# %choose <task variable>`, such as `# %choose collect`, is the one deliberate exception, and a test checks every task file.
- Pass `seed=None` to `llm.prompt()`. The default seed, 0, could make a model give the same answer every time.
- Don't pass `schema=`. Every model replies in plain text, and our own parser reads it (PLAN.md, "Replies").

## Commands

```bash
# Setup, once (Git Bash, Python 3.11+)
python -m venv .venv && source .venv/Scripts/activate && pip install -r requirements-dev.txt
kaggle auth login && kaggle b init -y

# Run the tests. They're offline and spend no quota.
python -m pytest

# Run locally. Ask first if the task calls a model.
set -a && source .env && set +a && PYTHONUTF8=1 python <task>.py

# tasks/collect.py prompts only the models in its MODELS list. .env's
# LLM_DEFAULT is ibm/granite-4.0-h-small, which the server doesn't offer, so it
# can't be on the list and a plain local run makes no calls (PLAN.md D23). A
# test fails if it's on the list. `kaggle b init` resets it to
# google/gemini-3-flash-preview, which is on the list, so set it again after
# init. kaggle_benchmarks loads .env by itself on import, over the shell's own
# variables, so LLM_DEFAULT can't be overridden on the command line (PLAN.md
# 4.5b). To draw from a listed model locally, change LLM_DEFAULT in .env (ask
# first: it spends quota), and change it back afterwards.

# Refresh the copies of shared modules in task files (tests fail if stale)
python -m randombench.sync

# Rebuild the scoring task files, tasks/scores/<slug>.py, from the template
# and data/ (tests fail if stale). They never prompt a model.
python -m randombench.build

# Push, then run on Kaggle. Ask first: the run calls models. A push's creation
# run uses Gemini 3.8 Flash, the server's default model, which MODELS leaves
# out, so pushing prompts nothing (PLAN.md D23). The scoring tasks never
# prompt a model. Model Proxy holds each call's maximum possible cost (up to
# $1.54 for GPT-5.6 Terra) against the daily quota until the call finishes.
# So run at most two collection runs at once, and check the extracted draws
# for errors: a refused call is still one request, so the log's retry count
# can't show it (PLAN.md 4.11).
kaggle b t push <slug> -f <task>.py --wait
kaggle b t run <slug> -m <model> --wait

# Check a run and download its results into results/ (gitignored)
kaggle b t status <slug>
kaggle b t download <slug> -o results

# Turn a task's downloaded runs into data/<model>/<run_id>.jsonl (committed)
python -m randombench.extract results/<slug>

# Score every model's stored draws, offline
python -m randombench.score

# Read a Benchmark's results at full precision, and the AI quota used. Both are
# read-only. Benchmarks are created and edited only on the website (PLAN.md 4.9).
# randombench-overall holds the headline, and randombench holds the nine
# sub-scores as RandomBench Biases (PLAN.md D15).
kaggle b leaderboard benjaminbrumfield/randombench-overall -s --format json
kaggle b leaderboard benjaminbrumfield/randombench -s --format json
kaggle b quota

# Make the task public. This can't be undone from the CLI, so ask first.
kaggle b t publish <slug>
```

- `PYTHONUTF8=1` is needed on Windows. Without it, the console UI's emoji crash the default cp1252 encoding.
- Local runs reach only the models in `LLMS_AVAILABLE` in `.env`.
- The API key in `.env` is short-lived; refresh it with `kaggle b auth -y`. Never commit `.env`.

## Skills

`.claude/skills/` holds two vendored skills, `kaggle-benchmarks` and `kaggle-cli`. They're unmodified upstream copies pinned in [SOURCES.md](.claude/skills/SOURCES.md), so don't edit them. Where `kaggle-benchmarks` points at `src/kaggle_benchmarks/`, read the installed package instead: `python -c "import kaggle_benchmarks; print(kaggle_benchmarks.__file__)"`.
