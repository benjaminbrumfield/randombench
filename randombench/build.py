"""Write the ten scoring task files from one template (PLAN.md 4.5b, D20),
and the leaderboard page, docs/index.html (5.3, D26).

Each score gets a task of its own (D2): the headline and the nine
sub-scores. A push uploads one file, so each task needs a file, and the ten
differ only in their score, title, and description. For each one, this
writes tasks/scores/<slug>.py from tasks/templates/score.py. It fills in the
template's placeholders, __SCORE__, __TITLE__, __DESCRIPTION__, and
__STORED__, which embeds every model's stored replies from data/ (D13). Then
it copies in parsing.py and checks.py, as `python -m randombench.sync` does
(D6).

Each file holds its title as a string literal. The CLI finds a task's title
by reading the file's syntax, and a title that's looked up or computed isn't
found.

The page shows every model's scores from data/, as randombench/page.py
explains.

A test fails when a file is out of date, so build again after changing a
template, a shared module, or data/.

Usage: python -m randombench.build [--data data]
"""

import argparse
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from randombench.extract import load_draws
from randombench.page import PAGE, newest, render_page
from randombench.score import score_draws
from randombench.sync import ROOT, synced

TEMPLATE = ROOT / "tasks" / "templates" / "score.py"
SCORES_DIR = ROOT / "tasks" / "scores"
DATA = ROOT / "data"

PLACEHOLDERS = {"__SCORE__", "__TITLE__", "__DESCRIPTION__", "__STORED__"}
_PLACEHOLDER = re.compile(r"__[A-Z]+__")


class BuildError(Exception):
    """The template or a task's text breaks the rules above."""


@dataclass(frozen=True)
class ScoringTask:
    score: str  # its key in score_answers()
    title: str  # its slug is made from this at the first push, and is then permanent (D10)
    description: str  # the subtitle under the title on the task page (1.3)

    @property
    def slug(self) -> str:
        # What the CLI's slugify makes of these titles. A test checks it.
        return self.title.lower().replace(" ", "-")

    @property
    def file(self) -> Path:
        return SCORES_DIR / f"{self.slug}.py"


# In score_answers' order, with the titles under "Names" in PLAN.md.
SCORING_TASKS = (
    ScoringTask(
        "headline",
        "RandomBench Score",
        "RandomBench's headline: the share of the model's replies that are valid, times the"
        " uniformity of its random integers from 0 to 99. 1 for a perfect random generator.",
    ),
    ScoringTask(
        "invalid_answers",
        "RandomBench Valid Answers",
        "The share of the model's replies that give a single integer from 0 to 99, as asked."
        " 1 means every reply was valid.",
    ),
    ScoringTask(
        "uniformity",
        "RandomBench Uniformity",
        "The effective number of values among the model's random integers from 0 to 99, out of"
        " 100: a fair pick among that many values repeats itself as often. 1 for a perfect random generator.",
    ),
    ScoringTask(
        "primes",
        "RandomBench Primes",
        "1 minus the gap between the share of the model's random integers from 0 to 99 that are"
        " prime and the 25% a perfect random generator gives.",
    ),
    ScoringTask(
        "multiples_of_10",
        "RandomBench Multiples of 10",
        "1 minus the gap between the share of the model's random integers from 0 to 99 that are"
        " multiples of 10 and the 10% a perfect random generator gives.",
    ),
    ScoringTask(
        "multiples_of_5",
        "RandomBench Multiples of 5",
        "1 minus the gap between the share of the model's random integers from 0 to 99 that are"
        " multiples of 5 and the 20% a perfect random generator gives.",
    ),
    ScoringTask(
        "favorite",
        "RandomBench Favorite Number",
        "1 minus how far the share of the model's favorite number among its random integers from"
        " 0 to 99 exceeds 1%, scaled so that always giving one number scores 0.",
    ),
    ScoringTask(
        "edges",
        "RandomBench Edges",
        "1 minus the gap between the share of the model's random integers from 0 to 99 that are"
        " in 0 to 9 or 90 to 99 and the 20% a perfect random generator gives.",
    ),
    ScoringTask(
        "repeated_digits",
        "RandomBench Repeated Digits",
        "1 minus the gap between the share of the model's random integers from 0 to 99 that"
        " repeat a digit (11, 22, and so on up to 99) and the 9% a perfect random generator gives.",
    ),
    ScoringTask(
        "last_digit",
        "RandomBench Last Digit",
        "The effective number of last digits among the model's random integers from 0 to 99,"
        " out of 10. 1 for a perfect random generator.",
    ),
)


def stored_replies(draws: list[dict]) -> dict[str, dict]:
    """Each model's stored replies, as the task files embed them: its run ids,
    its failed calls, and a tally of its raw replies, most common first."""
    by_model = defaultdict(list)
    for draw in draws:
        by_model[draw["model"]].append(draw)
    stored = {}
    for model in sorted(by_model):
        model_draws = by_model[model]
        tally = Counter(d["reply"] for d in model_draws if d["reply"] is not None)
        stored[model] = {
            # Run ids are numbers, so shorter ones come first.
            "runs": sorted({d["run_id"] for d in model_draws}, key=lambda r: (len(r), r)),
            "failed": sum(d["reply"] is None for d in model_draws),
            "replies": dict(sorted(tally.items(), key=lambda item: (-item[1], item[0]))),
        }
    return stored


def format_stored(stored: dict[str, dict]) -> str:
    """The stored replies as a Python literal, one reply per line.

    ascii() writes each string so that it reads back exactly, and in plain
    ASCII, whatever encoding the CLI reads the file with.
    """
    lines = ["{"]
    for model, s in stored.items():
        lines += [
            f"    {ascii(model)}: {{",
            f"        'runs': [{', '.join(ascii(r) for r in s['runs'])}],",
            f"        'failed': {s['failed']},",
            "        'replies': {",
            *(f"            {ascii(reply)}: {count}," for reply, count in s["replies"].items()),
            "        },",
            "    },",
        ]
    lines.append("}")
    return "\n".join(lines)


def render(task: ScoringTask, stored: dict[str, dict]) -> str:
    """Return a scoring task file's text."""
    text = TEMPLATE.read_text(encoding="utf-8")
    found = set(_PLACEHOLDER.findall(text))
    if found != PLACEHOLDERS:
        raise BuildError(f"the template's placeholders are {sorted(found)}, not {sorted(PLACEHOLDERS)}")
    for value in [task.score, task.title, task.description]:
        # Each goes inside a double-quoted string literal.
        if any(c in value for c in '"\\\n'):
            raise BuildError(f"{value!r} can't hold a double quote, a backslash, or a line break")
    text = (
        text.replace("__SCORE__", task.score)
        .replace("__TITLE__", task.title)
        .replace("__DESCRIPTION__", task.description)
        # Last, so that a reply can't be mistaken for a placeholder.
        .replace("__STORED__", format_stored(stored))
    )
    return synced(text)


def _write(file: Path, text: str) -> tuple[Path, bool]:
    """Write the file if its text differs. Return it, and whether it changed."""
    changed = not file.is_file() or file.read_text(encoding="utf-8") != text
    if changed:
        file.parent.mkdir(exist_ok=True)
        file.write_text(text, encoding="utf-8", newline="\n")
    return file, changed


def build(data_dir: Path = DATA) -> list[tuple[Path, bool]]:
    """Write every scoring task file and the page. Return each file, and whether it changed."""
    draws = load_draws(data_dir)
    stored = stored_replies(draws)
    results = [_write(task.file, render(task, stored)) for task in SCORING_TASKS]
    results.append(_write(PAGE, render_page(score_draws(draws), newest(draws))))
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write the scoring task files and the page from data/.")
    parser.add_argument("--data", type=Path, default=DATA, help="where the records are")
    args = parser.parse_args(argv)
    for file, changed in build(args.data):
        print(f"{file.relative_to(ROOT)}: {'updated' if changed else 'up to date'}")


if __name__ == "__main__":
    main()
