# randombench

An LLM benchmark built with the [`kaggle_benchmarks`](https://github.com/Kaggle/kaggle-benchmarks) library and published to Kaggle Benchmarks with the `kaggle` CLI.

## Skills

Two Claude Code skills are vendored in `.claude/skills/` and load automatically:

- `kaggle-benchmarks`: writing tasks (decorators, schemas, assertions, judges, `.evaluate()`).
- `kaggle-cli`: `kaggle benchmarks` push / run / status / download / publish, plus auth.

They are unmodified upstream copies pinned to specific commits; see [.claude/skills/SOURCES.md](.claude/skills/SOURCES.md). Don't edit them. Project-specific rules go in this file.

When the `kaggle-benchmarks` skill says to check `src/kaggle_benchmarks/`, use the installed package instead. Find it with `python -c "import kaggle_benchmarks; print(kaggle_benchmarks.__file__)"`.

## Task files

- Percent-format `.py` scripts: `# %%` cells, `.run()` / `.evaluate()` at top level, never inside `if __name__ == "__main__":`.
- The slug from `@kbench.task(name=...)` is what every `kaggle b t` command takes. Keep it stable so a push creates a new version of the task, not a new task.
- One leaderboard task per file; helper sub-tasks use `store_task=False`.
- Keep IPython magics commented (`# !pip install ...`) so files run locally.

## Iteration loop

1. One-time setup (Python 3.11+): `python -m venv .venv && source .venv/Scripts/activate && pip install -r requirements.txt`, then `kaggle auth login`, then `kaggle b init -y`. Init writes `.env`, plus `example_task.py` and `kaggle_benchmarks_reference.md`, which are gitignored scaffolding.
2. Local run (Git Bash): `set -a && source .env && set +a && python <task>.py`. Check the assertion output and the `.run.json`.
3. Push a new version: `kaggle b t push <slug> -f <task>.py --wait`
4. Run on Kaggle: `kaggle b t run <slug> -m <model> [-m <model> ...] --wait` (`kaggle b t models` lists slugs).
5. Inspect: `kaggle b t status <slug>`, `kaggle b t logs <slug>`, `kaggle b t download <slug> -o results`.
6. Release: `kaggle b t publish <slug>`. This makes the task public and can't be undone from the CLI, so confirm with the user first.

Local runs only reach the models in `LLMS_AVAILABLE`. Validate on the target models server-side before drawing conclusions.

## Secrets and outputs

- `.env` holds `MODEL_PROXY_API_KEY`. Never commit it. The key is short-lived; refresh it with `kaggle b auth -y`.
- Local `*.run.json` / `*.task.json` files and `results/` are gitignored. Commit specific results deliberately when the README cites them.
