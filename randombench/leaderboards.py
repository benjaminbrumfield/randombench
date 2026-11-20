"""Check that both Kaggle Benchmarks show the scores data/ gives.

The update-leaderboards skill runs this last, after it pushes the scoring
tasks, runs each new version on every model, and the user moves both
Benchmarks to the new versions on the website. A Benchmark pins each task row
to one version, so until then it shows the old version's results (PLAN.md
4.12).

Each leaderboard row names a model by its -m slug, which is also its data/
folder's name. For each task, it gives the version the row is pinned to and
the result at full precision, with its interval. This scores data/ as
`python -m randombench.score` does, and compares every value and interval.
Kaggle's results have matched the local ones exactly since 4.6, so a
difference means a Benchmark shows a version built from other draws.

A model Kaggle no longer lists can't be run, so the new versions have no
result for it, and it drops off a Benchmark that moves to them. The report
lists such models, but they're not a problem.

It reads Kaggle's model list, as `python -m randombench.models` does, and
both Benchmarks. It's read-only and prompts no model.

Usage: python -m randombench.leaderboards [--data DIR]
"""

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

from randombench.build import SCORING_TASKS
from randombench.extract import load_draws
from randombench.models import _rows, fetch_listed, stored_models
from randombench.score import score_draws
from randombench.sync import ROOT

OWNER = "benjaminbrumfield"
# Each Benchmark's slug and its scoring tasks. The headline is alone on
# randombench-overall, so that Benchmark ranks the models by it (D15).
BENCHMARKS = {
    "randombench-overall": tuple(t for t in SCORING_TASKS if t.score == "headline"),
    "randombench": tuple(t for t in SCORING_TASKS if t.score != "headline"),
}
# Kaggle's results have matched to every digit (4.6, 5.2). This only allows
# for a last-digit difference, should the server's numpy change.
TOLERANCE = 1e-12

# A task's result: (value, ±), or None where the task returned False.
Result = tuple[float, float] | None


def read_rows(rows: list[dict]) -> dict[str, dict[str, tuple[int, Result]]]:
    """Map each row's -m slug to {task slug: (version, result)}, from the API's
    rows as to_dict() writes them. A task with no result is left out."""
    table = {}
    for row in rows:
        cells = table.setdefault(row["modelVersionSlug"], {})
        for cell in row.get("taskResults", []):
            # "(Overall)" names no task.
            if "benchmarkTaskSlug" not in cell or "result" not in cell:
                continue
            numeric = cell["result"].get("numericResult")
            # A task that returned False has no numeric result. Protobuf's
            # JSON leaves out a field at its default, so a score of exactly 0
            # has no value (4.6), and an interval of 0 has none either.
            result = None if numeric is None else (numeric.get("value", 0.0), numeric.get("confidenceInterval", 0.0))
            cells[cell["benchmarkTaskSlug"].rsplit("/", 1)[-1]] = (cell.get("taskVersion"), result)
    return table


def local_results(data_dir: Path) -> dict[str, dict[str, Result]]:
    """Map each data/ folder's -m slug to {task slug: result}, the result its
    scoring task returns. A model with no replies is left out: its every
    call failed, and a scoring task has nothing to score."""
    scored = {m.model: m for m in score_draws(load_draws(data_dir))}
    results = {}
    for slug, llm_name in stored_models(data_dir).items():
        # A scoring task finds a model's replies by the llm.name its run gets,
        # the one the model's newest stored run holds.
        m = scored.get(llm_name)
        if m is None or m.draws == m.failed:
            continue
        results[slug] = {
            t.slug: None if m.scores[t.score] is None else (m.scores[t.score], m.intervals[t.score])
            for t in SCORING_TASKS
        }
    return results


def same(theirs: Result, ours: Result) -> bool:
    if theirs is None or ours is None:
        return theirs is ours
    return all(math.isclose(a, b, rel_tol=0, abs_tol=TOLERANCE) for a, b in zip(theirs, ours))


@dataclass
class Check:
    """How the Benchmarks compare with data/."""

    # Each Benchmark's slug and how many models it shows.
    shown: dict[str, int] = field(default_factory=dict)
    # Each task's slug and the versions its rows are pinned to.
    versions: dict[str, set[int]] = field(default_factory=dict)
    matched: int = 0
    # (Benchmark, model, task, Kaggle's result, data/'s result)
    differ: list[tuple[str, str, str, Result, Result]] = field(default_factory=list)
    # (Benchmark, model, task): stored replies, and Kaggle lists the model,
    # but the Benchmark has no result. The task is "" when the model has no
    # row at all.
    missing: list[tuple[str, str, str]] = field(default_factory=list)
    # (Benchmark, model): on the Benchmark, with no stored replies.
    extra: list[tuple[str, str]] = field(default_factory=list)
    # Models with stored replies that Kaggle no longer lists.
    unlisted: list[str] = field(default_factory=list)

    @property
    def problems(self) -> int:
        return len(self.differ) + len(self.missing) + len(self.extra)


def check(boards: dict[str, list[dict]], local: dict[str, dict[str, Result]], listed: set[str]) -> Check:
    """Compare each Benchmark's rows, as the API returns them, with data/'s
    results. listed holds the -m slugs Kaggle lists."""
    result = Check(unlisted=sorted(set(local) - listed))
    for board, tasks in BENCHMARKS.items():
        table = read_rows(boards[board])
        result.shown[board] = len(table)
        result.extra += [(board, slug) for slug in sorted(set(table) - set(local))]
        for slug in sorted(set(local) & listed):
            if slug not in table:
                result.missing.append((board, slug, ""))
                continue
            for task in tasks:
                if task.slug not in table[slug]:
                    result.missing.append((board, slug, task.slug))
                    continue
                version, theirs = table[slug][task.slug]
                result.versions.setdefault(task.slug, set()).add(version)
                if same(theirs, local[slug][task.slug]):
                    result.matched += 1
                else:
                    result.differ.append((board, slug, task.slug, theirs, local[slug][task.slug]))
    return result


def _result(r: Result) -> str:
    return "FAIL" if r is None else f"{r[0]:.6f} ±{r[1]:.6f}"


def format_check(c: Check) -> str:
    def versions(tasks) -> str:
        found = sorted({v for t in tasks for v in c.versions.get(t.slug, ())})
        return ", ".join(f"v{v}" for v in found) or "no version"

    lines = [f"{board}: {c.shown[board]} models, {len(tasks)} task{'s' * (len(tasks) > 1)} at {versions(tasks)}"
             for board, tasks in BENCHMARKS.items()]
    lines += [f"{c.matched} results match data/, value and interval.", ""]
    lines += [f"Differ ({len(c.differ)}): the Benchmark's result isn't the one data/ gives."]
    lines += _rows([(b, m, t, f"Kaggle {_result(k)}, data/ {_result(d)}") for b, m, t, k, d in c.differ])
    lines += ["", f"Missing ({len(c.missing)}): stored replies, and Kaggle lists the model,"
                  " but the Benchmark has no result for it."]
    lines += _rows([(b, m, t or "no row: add the model on the website") for b, m, t in c.missing])
    lines += ["", f"Extra ({len(c.extra)}): on the Benchmark, with no stored replies."]
    lines += _rows([(b, m) for b, m in c.extra])
    lines += ["", f"Not on Kaggle ({len(c.unlisted)}): stored replies, but Kaggle no longer lists"
                  " the model, so no run can score it. Not a problem."]
    lines += [f"  {m}" for m in c.unlisted] or ["  none"]
    lines += ["", "Up to date." if not c.problems else f"Not up to date: {c.problems} problems."]
    return "\n".join(lines)


def fetch_boards() -> dict[str, list[dict]]:
    """Read each Benchmark's rows, as `kaggle b leaderboard` does."""
    # Imported here so the offline tests don't need the CLI's sign-in.
    from kaggle import api

    return {board: [row.to_dict() for row in api.benchmark_leaderboard_view(f"{OWNER}/{board}").rows]
            for board in BENCHMARKS}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Check that both Kaggle Benchmarks show the scores data/ gives. Read-only: it prompts no model."
    )
    parser.add_argument("--data", type=Path, default=ROOT / "data", help="where the stored draws are")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    listed = {m.slug for m in fetch_listed()}
    print(format_check(check(fetch_boards(), local_results(args.data), listed)))


if __name__ == "__main__":
    main()
