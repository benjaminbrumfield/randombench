# RandomBench

RandomBench measures how well LLMs generate random integers from 0 to 99 and scores their answers for biases, such as favoring primes or round numbers. It's built with [`kaggle_benchmarks`](https://github.com/Kaggle/kaggle-benchmarks) and runs on Kaggle Benchmarks.

[PLAN.md](PLAN.md) holds the design, the decisions and the build steps. Read it before starting work, and record decisions there.

## How we work

- **One step at a time.** Each PLAN.md step is one small commit. This is a learning project, so explain what each step shows, and raise open decisions instead of settling them quietly.
- **Ask before any LLM call.** Local task runs, probe scripts and `kaggle b t run` all spend Model Proxy quota. Say which models and how many calls, then wait for a yes. Read-only commands such as `kaggle b t models` are fine.

## Writing tasks

The `kaggle-benchmarks` skill covers the general rules. On top of those:

- Keep the slug in `@kbench.task(name=...)` stable, so a push makes a new version rather than a new task.
- A push uploads only the task file, so everything the task needs on Kaggle (prompts, parsing) must be in it.
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

# Push, then run on Kaggle. Ask first: the run calls models.
kaggle b t push <slug> -f <task>.py --wait
kaggle b t run <slug> -m <model> --wait

# Check a run and download its results into results/ (gitignored)
kaggle b t status <slug>
kaggle b t download <slug> -o results

# Make the task public. This can't be undone from the CLI, so ask first.
kaggle b t publish <slug>
```

- `PYTHONUTF8=1` is needed on Windows. Without it, the console UI's emoji crash the default cp1252 encoding.
- Local runs reach only the models in `LLMS_AVAILABLE` in `.env`.
- The API key in `.env` is short-lived; refresh it with `kaggle b auth -y`. Never commit `.env`.

## Skills

`.claude/skills/` holds two vendored skills, `kaggle-benchmarks` and `kaggle-cli`. They're unmodified upstream copies pinned in [SOURCES.md](.claude/skills/SOURCES.md), so don't edit them. Where `kaggle-benchmarks` points at `src/kaggle_benchmarks/`, read the installed package instead: `python -c "import kaggle_benchmarks; print(kaggle_benchmarks.__file__)"`.
