"""Describe one batch's answers, for the report after each collect-draws batch.

It reads one run's stored draws, data/<model>/<run_id>.jsonl, and prints
Markdown: a summary line, then every value the model gave, grouped by how
often it came up, then the answers by decade, by last digit, and by prompt,
then the batch's RandomBench Score and most common answer beside the
model's over all its stored runs. Failed calls have no reply and are left
out, as in scoring (3.4). It reads data/ only, so it's offline and prompts
no model.

Usage: python -m randombench.batch data/<model>/<run_id>.jsonl
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from randombench.checks import counts_of, last_digit_counts, score_answers, score_intervals
from randombench.extract import load_draws
from randombench.page import DISPLAY_NAMES
from randombench.parsing import parse_reply


def load_run(file: Path) -> list[dict]:
    with open(file, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def answers_of(draws: list[dict]) -> list[int | None]:
    """Each reply's answer, None where the parser rejects it. Failed calls have
    no reply, so they're left out."""
    return [parse_reply(d["reply"]) for d in draws if d["reply"] is not None]


def by_count(counts) -> list[tuple[int, list[int]]]:
    """The values that came up, grouped by how often each did, most often first."""
    groups = defaultdict(list)
    for value, count in enumerate(counts):
        if count:
            groups[int(count)].append(value)
    return sorted(groups.items(), reverse=True)


def most_common(counts) -> str:
    """The most common value, or every one tied for it, out of the valid answers."""
    top, total = int(max(counts)), int(sum(counts))
    if top < 2:
        return "none repeats" if total else "-"
    values = ", ".join(str(v) for v, c in enumerate(counts) if c == top)
    each = " each" if "," in values else ""
    return f"{values}: {top} of {total}{each}"


def _n(count: int, noun: str, plural: str | None = None) -> str:
    """'1 failed call', '2 failed calls'."""
    return f"{count} {noun if count == 1 else plural or noun + 's'}"


def _score(answers: list[int | None]) -> str:
    if not answers:
        return "-"
    score, half = score_answers(answers)["headline"], score_intervals(answers)["headline"]
    return "-" if score is None else f"{score:.3f} ± {half:.3f}"


def _row(cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _table(headings: list[str], rows: list[list], align: list[str]) -> list[str]:
    return [_row(headings), _row(align), *(_row(r) for r in rows)]


def rundown(run: list[dict], model_draws: list[dict]) -> str:
    """The batch's report. run is one run's draws, and model_draws every stored
    draw of the same model, the run's own included."""
    first = run[0]
    name = DISPLAY_NAMES.get(first["model"], (first["model"],))[0]
    answers = answers_of(run)
    valid = [a for a in answers if a is not None]
    summary = (
        f"**{name}, run {first['run_id']} (v{first['task_version']}):** {_n(len(run), 'draw')},"
        f" {_n(len(run) - len(answers), 'failed call')},"
        f" {_n(len(answers) - len(valid), 'invalid reply', 'invalid replies')},"
        f" {_n(len(valid), 'valid answer')}."
    )
    if not valid:
        return summary + "\n\nNo valid answers to describe."
    counts = counts_of(valid)
    summary += (
        f" They take {_n(sum(1 for c in counts if c), 'distinct value')}, and their mean is"
        f" {sum(valid) / len(valid):.1f}, against 49.5 for a uniform pick."
    )
    tenth = f"{len(valid) / 10:.1f}"

    lines = [summary, "", "**How often each value came up**", ""]
    lines += _table(["Times", "Values"], [[n, ", ".join(map(str, vs))] for n, vs in by_count(counts)],
                    ["---:", "---"])

    lines += ["", f"**By decade** (a uniform pick: {tenth} each)", ""]
    decades = counts.reshape(10, 10).sum(axis=1)
    lines += _table([f"{d * 10}–{d * 10 + 9}" for d in range(10)], [list(decades)], ["---:"] * 10)

    lines += ["", f"**By last digit** (a uniform pick: {tenth} each)", ""]
    lines += _table([str(d) for d in range(10)], [list(last_digit_counts(counts))], ["---:"] * 10)

    lines += ["", "**By prompt**", ""]
    by_prompt = defaultdict(list)
    for d in run:
        if d["reply"] is not None:
            by_prompt[(d["prompt_id"] or "?").rsplit("-", 1)[0]].append(parse_reply(d["reply"]))
    rows = []
    for prompt, prompt_answers in sorted(by_prompt.items()):
        prompt_valid = [a for a in prompt_answers if a is not None]
        prompt_counts = counts_of(prompt_valid)
        rows.append([prompt, len(prompt_valid), len(prompt_answers) - len(prompt_valid),
                     sum(1 for c in prompt_counts if c), most_common(prompt_counts)])
    lines += _table(["Prompt", "Valid", "Invalid", "Distinct", "Most common"], rows,
                    ["---", "---:", "---:", "---:", "---"])

    rows = [["This batch", len(valid), _score(answers), most_common(counts)]]
    runs = {d["run_id"] for d in model_draws}
    if runs - {first["run_id"]}:
        lines += ["", "**Against all its runs**", ""]
        all_answers = answers_of(model_draws)
        all_valid = [a for a in all_answers if a is not None]
        rows.append([f"All {len(runs)} runs", len(all_valid), _score(all_answers),
                     most_common(counts_of(all_valid))])
    else:
        lines += ["", "**Its first run**", ""]
    lines += _table(["", "Valid", "RandomBench Score", "Most common"], rows, ["---", "---:", "---:", "---"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Describe one stored run's answers. Offline: it prompts no model."
    )
    parser.add_argument("run_file", type=Path, help="data/<model>/<run_id>.jsonl, as extract prints it")
    args = parser.parse_args(argv)
    # The table cells hold characters outside Windows' default encoding for a pipe.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    run = load_run(args.run_file)
    if not run:
        print(f"{args.run_file}: no draws")
        return
    # data/<model>/<run_id>.jsonl, so the data folder is two levels up.
    model_draws = [d for d in load_draws(args.run_file.parent.parent) if d["model"] == run[0]["model"]]
    print(rundown(run, model_draws))


if __name__ == "__main__":
    main()
