"""Compare the models Kaggle lists with the ones RandomBench covers.

`kaggle b t models` lists every model a task can run on, but it prints only
each model's slug and display name. The API it calls also gives each model's
Model Proxy name, which is the `llm.name` a run sees and the string MODELS in
tasks/collect.py must hold (PLAN.md 1.3, D12), and whether Model Proxy allows
the model at all. This module asks the API directly and compares its reply
with data/ and MODELS. It's read-only and prompts no model, so it spends no
quota.

A model's data/ folder is named by its -m slug (extract.py), and its records
hold its `llm.name`. So the comparison matches Kaggle's list to data/ by slug,
and to MODELS by `llm.name`.

It can't tell two things:
- Whether Model Proxy serves a model. The API allows Grok 4.5, yet all of
  its calls got a 404 (D24). Only a pilot shows it.
- Which model is the server's default, the one every push's creation run
  uses (D23). `isDefault` marks each model's default version, and every
  model has one.

Usage: python -m randombench.models [--data DIR]
"""

import argparse
import ast
import json
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from randombench.sync import ROOT

COLLECT = ROOT / "tasks" / "collect.py"


@dataclass(frozen=True)
class Listed:
    """One model that Kaggle lists."""

    slug: str  # the -m slug, such as gemma-4-26b-a4b-it, and its data/ folder's name
    name: str  # the display name, such as Gemma 4 26B A4B
    llm_name: str  # the Model Proxy name, such as google/gemma-4-26b-a4b: llm.name in a run
    proxy: bool  # whether Model Proxy allows it
    released: date | None  # Kaggle gives no date for some models


def listed_from_api(model) -> Listed:
    """Read the fields that matter from one of the API's BenchmarkModel objects."""
    version = model.version
    return Listed(
        slug=version.slug,
        name=model.display_name,
        llm_name=version.model_proxy_slug or "",
        proxy=bool(version.allow_model_proxy),
        released=version.release_date.date() if version.release_date else None,
    )


def fetch_listed() -> list[Listed]:
    """Ask Kaggle for every benchmark model, as `kaggle b t models` does."""
    # Imported here so the offline tests don't need the CLI's sign-in.
    # Importing kaggle signs in with the CLI's stored credentials.
    from kaggle import api

    # The CLI's own fetch, which pages through the list and retries. It's
    # private, but requirements.txt pins kaggle.
    with api.build_kaggle_client() as kaggle:
        return [listed_from_api(m) for m in api._fetch_all_benchmark_models(kaggle)]


def stored_models(data_dir: Path) -> dict[str, str]:
    """Map each data/ folder's name, a model's -m slug, to the llm.name its
    newest run holds, the name a run on that slug gets now. Kaggle's run ids
    count up, so the newest run has the highest."""
    stored = {}
    for folder in sorted(p for p in Path(data_dir).iterdir() if p.is_dir()):
        newest = max(folder.glob("*.jsonl"), key=lambda f: int(f.stem), default=None)
        if newest:
            with open(newest, encoding="utf-8") as f:
                stored[folder.name] = json.loads(f.readline())["model"]
    return stored


def allowed_models(task: Path = COLLECT) -> tuple[str, ...]:
    """Read MODELS from the collection task without running it.

    Running it would import kaggle_benchmarks, which loads .env.
    """
    tree = ast.parse(Path(task).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "MODELS" for t in node.targets):
            return tuple(ast.literal_eval(node.value))
    raise ValueError(f"{task} doesn't assign MODELS")


@dataclass
class Report:
    """How Kaggle's list compares with data/ and MODELS. Each listed model is
    in exactly one of new, renamed, left_out, and covered, and in not_allowed
    too if Model Proxy doesn't allow it."""

    listed: list[Listed]
    # Listed, with no folder in data/.
    new: list[Listed] = field(default_factory=list)
    # Listed, with a folder in data/ whose newest run holds another llm.name, the second item.
    renamed: list[tuple[Listed, str]] = field(default_factory=list)
    # Listed, with stored draws, but its llm.name isn't in MODELS.
    left_out: list[Listed] = field(default_factory=list)
    # Listed, with stored draws, and its llm.name is in MODELS.
    covered: list[Listed] = field(default_factory=list)
    # Listed, but Model Proxy doesn't allow it.
    not_allowed: list[Listed] = field(default_factory=list)
    # Not listed: a data/ folder (slug, llm.name), or a MODELS entry with no
    # folder ("", llm.name).
    gone: list[tuple[str, str]] = field(default_factory=list)
    allowed: tuple[str, ...] = ()


def compare(listed: list[Listed], stored: dict[str, str], allowed: tuple[str, ...]) -> Report:
    report = Report(listed=listed, allowed=allowed)
    for m in listed:
        if not m.proxy:
            report.not_allowed.append(m)
        if m.slug not in stored:
            report.new.append(m)
        # Kaggle gives no llm.name for a model Model Proxy doesn't allow, and
        # that's no rename.
        elif m.llm_name and stored[m.slug] != m.llm_name:
            report.renamed.append((m, stored[m.slug]))
        elif stored[m.slug] in allowed:
            report.covered.append(m)
        else:
            report.left_out.append(m)
    slugs = {m.slug for m in listed}
    report.gone = [(slug, name) for slug, name in stored.items() if slug not in slugs]
    names = {m.llm_name for m in listed} | set(stored.values())
    report.gone += [("", name) for name in allowed if name not in names]
    return report


def _rows(rows: list[tuple[str, ...]]) -> list[str]:
    """Indent the rows and pad every column but the last to its widest cell."""
    if not rows:
        return ["  none"]
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]) - 1)]
    return ["  " + "  ".join(c.ljust(w) for c, w in zip(r, widths)) + "  " + r[-1] for r in rows]


def format_report(report: Report) -> str:
    def released(m: Listed) -> str:
        return f"released {m.released}" if m.released else "no release date"

    def kept_in(slug: str, name: str) -> str:
        return ", ".join((["stored draws"] if slug else []) + (["in MODELS"] if name in report.allowed else []))

    lines = [
        f"Kaggle lists {len(report.listed)} models. RandomBench covers {len(report.covered)}:"
        " their draws are stored, and MODELS lists them.",
        "",
        f"New ({len(report.new)}): listed, with no draws in data/. Each would need a pilot"
        " before its full run (D23).",
    ]
    lines += _rows([(m.slug, m.name, m.llm_name or "(no Model Proxy name)",
                     released(m) + (", already in MODELS" if m.llm_name in report.allowed else ""))
                    for m in report.new])
    lines += ["", f"Gone ({len(report.gone)}): in data/ or MODELS, but Kaggle no longer lists it."]
    lines += _rows([(slug or "(no data/ folder)", name, kept_in(slug, name)) for slug, name in report.gone])
    lines += ["", f"Left out ({len(report.left_out)}): listed, with stored draws, but not in MODELS,"
                  " so a run on it makes no draws. D23 leaves out the server's default model, and"
                  " D24 leaves out Grok 4.5."]
    lines += _rows([(m.slug, m.name, m.llm_name or "(no Model Proxy name)") for m in report.left_out])
    lines += ["", f"Renamed ({len(report.renamed)}): its newest run in data/ holds another llm.name. A run"
                  " makes draws only if MODELS holds Kaggle's new name (D12)."]
    lines += _rows([(m.slug, f"stored as {old}", f"now {m.llm_name}") for m, old in report.renamed])
    lines += ["", f"Not allowed ({len(report.not_allowed)}): listed, but Model Proxy doesn't"
                  " allow it, so a run can't call it."]
    lines += _rows([(m.slug, m.name) for m in report.not_allowed])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare the models Kaggle lists with the ones RandomBench covers."
        " Read-only: it prompts no model."
    )
    parser.add_argument("--data", type=Path, default=ROOT / "data", help="where the stored draws are")
    args = parser.parse_args(argv)
    # A display name may hold a character outside Windows' default encoding for a pipe.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(format_report(compare(fetch_listed(), stored_models(args.data), allowed_models())))


if __name__ == "__main__":
    main()
