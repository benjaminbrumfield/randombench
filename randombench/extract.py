"""Turn downloaded run files into per-draw records under data/ (PLAN.md 2.5, D5).

`kaggle b t download <task> -o results` writes each run to
results/<task>/<version>/<model>/<run_id>/. The run id and the task version
exist only in that path: inside the files, the version is always 1 and the
run is always "Run #1". So data/ takes only runs downloaded from Kaggle, and
local runs stay in results/.

A run folder holds the parent run file, which embeds every draw that
finished, or, when the run died before the end, only the per-draw files
(1.2). The parent is read when there is one. Otherwise each per-draw file is
read, and the empty ones a crash leaves behind are skipped.

Each run becomes data/<model>/<run_id>.jsonl, one draw per line, in draw
order. Extracting a run again rewrites the same file, so no run is ever
counted twice. Replies are stored raw and never parsed here: scores are
always recomputed from the raw text.

A run shaped unlike the ones seen so far, such as a draw with two requests
or a prompt text that doesn't match its prompt id, raises ExtractError
instead of being guessed at.

Usage: python -m randombench.extract results/<task> [--data data]
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from randombench.prompts import PROMPTS

PROMPTS_BY_ID = {p.id: p for p in PROMPTS}

COMPLETED = "BENCHMARK_TASK_RUN_STATE_COMPLETED"
ERRORED = "BENCHMARK_TASK_RUN_STATE_ERRORED"
USER = "CONTENT_ROLE_USER"
ASSISTANT = "CONTENT_ROLE_ASSISTANT"

# A per-draw file is named after its row in the task's evaluation data, and
# collect.py numbers the rows by draw index: draw-run_param_id_37_<model>.run.json.
_PER_DRAW = re.compile(r"-run_param_id_(\d+)_")

# What the draw sub-task returns: its index and prompt id, plus either the
# reply or the error it caught (D11).
_RESULT_KEYS = ({"draw_index", "prompt_id", "reply"}, {"draw_index", "prompt_id", "error"})


class ExtractError(Exception):
    """A run isn't shaped the way the extractor expects."""


@dataclass(frozen=True)
class RunPath:
    """What a run's download path says: <task>/<version>/<model>/<run_id>."""

    task: str
    version: int | None  # None where the CLI writes "unset"
    model: str  # the -m slug, such as gpt-5.4-nano-2026-03-17
    run_id: str


def parse_run_path(run_dir: Path) -> RunPath:
    parts = Path(run_dir).parts
    if len(parts) < 4:
        raise ExtractError(f"{run_dir}: not a <task>/<version>/<model>/<run_id> folder")
    task, version, model, run_id = parts[-4:]
    if not run_id.isdigit():
        raise ExtractError(f"{run_dir}: {run_id!r} isn't a Kaggle run id")
    if version != "unset" and not version.isdigit():
        raise ExtractError(f"{run_dir}: {version!r} isn't a task version")
    return RunPath(task, None if version == "unset" else int(version), model, run_id)


def find_run_dirs(task_dir: Path) -> list[Path]:
    """Return every run folder in one task's download: <version>/<model>/<run_id>."""
    return sorted(p for p in Path(task_dir).glob("*/*/*") if p.is_dir())


def read_run(run_dir: Path) -> list[dict]:
    """Return one record for each draw in a run folder, in draw order."""
    run_dir = Path(run_dir)
    path = parse_run_path(run_dir)
    files = sorted(run_dir.glob("*.run.json"))
    parents = [f for f in files if not _PER_DRAW.search(f.name)]
    if len(parents) > 1:
        raise ExtractError(f"{run_dir}: {len(parents)} parent run files")

    if parents:
        subruns = [(None, s) for s in _load(parents[0]).get("subruns", [])]
    else:
        subruns = [
            (int(_PER_DRAW.search(f.name).group(1)), _load(f))
            for f in files
            if f.stat().st_size > 0  # a crash leaves the files it was writing empty
        ]

    try:
        records = [_draw_record(sub, path, file_index) for file_index, sub in subruns]
    except ExtractError as e:
        raise ExtractError(f"{run_dir}: {e}") from None

    indices = [r["draw_index"] for r in records if r["draw_index"] is not None]
    if len(indices) != len(set(indices)):
        raise ExtractError(f"{run_dir}: a draw index appears twice")
    models = {r["model"] for r in records}
    if len(models) > 1:
        raise ExtractError(f"{run_dir}: draws from several models: {sorted(models)}")
    # A draw whose index was lost (an uncaught error, in a parent file) goes last.
    records.sort(key=lambda r: (r["draw_index"] is None, r["draw_index"] or 0))
    return records


def _load(file: Path) -> dict:
    try:
        return json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ExtractError(f"{file}: not valid JSON ({e})") from None


def _draw_record(sub: dict, path: RunPath, file_index: int | None) -> dict:
    state = sub.get("state")
    prompt_text, reply_text, request = _read_conversation(sub)

    if state == COMPLETED:
        results = sub.get("results", [])
        if len(results) != 1 or "dictResult" not in results[0]:
            raise ExtractError(f"draw {file_index}: expected one dict result")
        result = results[0]["dictResult"]
        if set(result) not in _RESULT_KEYS:
            raise ExtractError(f"draw {file_index}: unexpected result keys {sorted(result)}")
        draw_index = _as_int(result["draw_index"])  # a dict result stores it as 37.0
        if file_index is not None and draw_index != file_index:
            raise ExtractError(f"draw {draw_index} is in the file for draw {file_index}")
        prompt_id = result["prompt_id"]
        if prompt_id not in PROMPTS_BY_ID:
            raise ExtractError(f"draw {draw_index}: unknown prompt id {prompt_id!r}")
        reply = result.get("reply")
        error = result.get("error")
        if reply is not None and reply != reply_text:
            raise ExtractError(f"draw {draw_index}: the result's reply isn't the conversation's")
    elif state == ERRORED:
        # An error the sub-task didn't catch. There's no result, so the index
        # comes from the file name, and the prompt id from the text sent.
        draw_index = file_index
        prompt_id = next((p.id for p in PROMPTS if p.text == prompt_text), None)
        reply = None
        error = sub.get("errorMessage", "")
    else:
        raise ExtractError(f"draw {file_index}: unexpected state {state!r}")

    expected_text = PROMPTS_BY_ID[prompt_id].text if prompt_id else None
    if prompt_text is not None and prompt_text != expected_text:
        raise ExtractError(f"draw {draw_index}: the text sent isn't the text of prompt {prompt_id}")

    metrics = request.get("metrics", {}) if request else {}
    return {
        "model": sub["modelVersion"]["slug"],
        "run_id": path.run_id,
        "task_version": path.version,
        "draw_index": draw_index,
        "prompt_id": prompt_id,
        "prompt_author": PROMPTS_BY_ID[prompt_id].author if prompt_id else None,
        "started_at": sub.get("startTime"),
        "reply": reply,
        "reasoning": request.get("reasoningTraces") if request else None,
        "input_tokens": _as_int(metrics.get("inputTokens")),
        "output_tokens": _as_int(metrics.get("outputTokens")),
        "input_cost_nanodollars": _as_int(metrics.get("inputTokensCostNanodollars")),
        "output_cost_nanodollars": _as_int(metrics.get("outputTokensCostNanodollars")),
        "latency_ms": _as_int(metrics.get("totalBackendLatencyMs")),
        "error": error,
    }


def _read_conversation(sub: dict) -> tuple[str | None, str | None, dict | None]:
    """Return a draw's prompt text, reply text and request.

    A draw is one conversation with at most one request, whose contents are
    the prompt and then the reply. A draw that failed may have no request,
    or a request that holds only the prompt.
    """
    conversations = sub.get("conversations", [])
    if len(conversations) > 1:
        raise ExtractError(f"{len(conversations)} conversations in one draw")
    requests = conversations[0].get("requests", []) if conversations else []
    if len(requests) > 1:
        raise ExtractError(f"{len(requests)} requests in one draw")
    if not requests:
        return None, None, None
    request = requests[0]

    contents = request.get("contents", [])
    if not 1 <= len(contents) <= 2:
        raise ExtractError(f"a draw's request has {len(contents)} contents")
    texts = []
    for content, role in zip(contents, [USER, ASSISTANT]):
        parts = content.get("parts", [])
        if content.get("role") != role or len(parts) != 1 or set(parts[0]) != {"text"}:
            raise ExtractError(f"unexpected content in a draw: {json.dumps(content)[:200]}")
        texts.append(parts[0]["text"])
    prompt_text, reply_text = (texts + [None])[:2]
    return prompt_text, reply_text, request


def _as_int(value) -> int | None:
    """Return a stored count as an int. Protobuf writes 64-bit integers as
    strings ("5000"), and a dict result stores numbers as floats (37.0)."""
    if value is None:
        return None
    if isinstance(value, float) and not value.is_integer():
        raise ExtractError(f"{value!r} isn't a whole number")
    return int(value)


def run_file(data_dir: Path, path: RunPath) -> Path:
    return Path(data_dir) / path.model / f"{path.run_id}.jsonl"


def extract_run(run_dir: Path, data_dir: Path) -> tuple[Path | None, list[dict]]:
    """Write one run's records to data/<model>/<run_id>.jsonl, replacing any
    earlier copy. Return the file and the records. A run with no draws
    writes nothing and returns None for the file."""
    records = read_run(run_dir)
    if not records:
        return None, records
    file = run_file(data_dir, parse_run_path(run_dir))
    file.parent.mkdir(parents=True, exist_ok=True)
    with open(file, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    return file, records


def load_draws(data_dir: Path) -> list[dict]:
    """Return every stored draw from every run, in run and draw order."""
    records = []
    for file in sorted(Path(data_dir).glob("*/*.jsonl")):
        with open(file, encoding="utf-8") as f:
            records.extend(json.loads(line) for line in f)
    keys = [(r["run_id"], r["draw_index"]) for r in records if r["draw_index"] is not None]
    if len(keys) != len(set(keys)):
        raise ExtractError(f"{data_dir}: a run's draw is stored twice")
    return records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract downloaded runs into data/.")
    parser.add_argument("task_dir", type=Path, help="one task's download: results/<task>")
    parser.add_argument("--data", type=Path, default=Path("data"), help="where records go")
    args = parser.parse_args(argv)
    for run_dir in find_run_dirs(args.task_dir):
        path = parse_run_path(run_dir)
        file, records = extract_run(run_dir, args.data)
        if file is None:
            print(f"{path.model} {path.run_id}: no draws, nothing written")
            continue
        errors = sum(r["error"] is not None for r in records)
        print(f"{path.model} {path.run_id} (v{path.version}): {len(records)} draws, {errors} errors -> {file}")


if __name__ == "__main__":
    main()
