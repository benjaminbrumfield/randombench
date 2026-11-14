import json

import pytest

from randombench.extract import (
    ExtractError,
    extract_run,
    find_run_dirs,
    load_draws,
    parse_run_path,
    read_run,
)
from randombench.prompts import PROMPTS

PLAIN, UNIFORM = PROMPTS[0], PROMPTS[1]
NANO = "openai/gpt-5.4-nano-2026-03-17"
COMPLETED = "BENCHMARK_TASK_RUN_STATE_COMPLETED"
ERRORED = "BENCHMARK_TASK_RUN_STATE_ERRORED"


# Run files shaped like the real ones from 2.4 (results/2.4-local).


def sub_run(draw_index, prompt=PLAIN, reply="47", error=None):
    """A completed draw, as the draw sub-task records it. With error= set, the
    draw caught that error and has no reply (D11)."""
    contents = [{"parts": [{"text": prompt.text}], "role": "CONTENT_ROLE_USER", "senderName": "User"}]
    result = {"draw_index": float(draw_index), "prompt_id": prompt.id}
    metrics = {}
    if error is None:
        contents.append({"parts": [{"text": reply}], "role": "CONTENT_ROLE_ASSISTANT", "senderName": NANO})
        result["reply"] = reply
        metrics = {
            "inputTokens": 25,
            "outputTokens": 4,
            "inputTokensCostNanodollars": "5000",
            "outputTokensCostNanodollars": "5000",
            "totalBackendLatencyMs": "475",
        }
    else:
        result["error"] = error
    return {
        "taskVersion": {"versionNumber": 1, "name": "draw"},
        "modelVersion": {"slug": NANO},
        "state": COMPLETED,
        "startTime": f"2026-11-14T22:25:10.{draw_index:06d}Z",
        "conversations": [{"id": "draw-1", "requests": [{"contents": contents, "metrics": metrics}]}],
        "results": [{"type": "AGGREGATED", "dictResult": result}],
        "pyRunId": "draw-Run #1",
    }


def errored_sub_run(prompt=PLAIN):
    """A draw that raised outside the sub-task's own try: no result."""
    run = sub_run(0, prompt=prompt, error="unused")
    run["state"] = ERRORED
    run["errorMessage"] = "Traceback ...\nRuntimeError: boom"
    del run["results"]
    return run


def run_dir(tmp_path, version="2", model="gpt-5.4-nano-2026-03-17", run_id="6807323"):
    d = tmp_path / "results" / "randombench-collect" / version / model / run_id
    d.mkdir(parents=True)
    return d


def write_parent(d, sub_runs):
    parent = {"taskVersion": {"versionNumber": 1}, "modelVersion": {"slug": NANO}, "subruns": sub_runs}
    (d / f"randombench-collect-run_id_Run_1_{NANO.replace('/', '_')}.run.json").write_text(json.dumps(parent))


def write_per_draw(d, index, sub_run):
    text = "" if sub_run is None else json.dumps(sub_run)
    (d / f"draw-run_param_id_{index}_{NANO.replace('/', '_')}.run.json").write_text(text)


# Reading a run


def test_reads_a_parent_file_in_draw_order(tmp_path):
    d = run_dir(tmp_path)
    # Sub-runs are stored in the order they finished, not in draw order (1.1).
    write_parent(d, [sub_run(2, UNIFORM, "42"), sub_run(0), sub_run(1)])
    records = read_run(d)
    assert [r["draw_index"] for r in records] == [0, 1, 2]
    assert records[2] == {
        "model": NANO,
        "run_id": "6807323",
        "task_version": 2,
        "draw_index": 2,
        "prompt_id": UNIFORM.id,
        "prompt_author": UNIFORM.author,
        "started_at": "2026-11-14T22:25:10.000002Z",
        "reply": "42",
        "reasoning": None,
        "input_tokens": 25,
        "output_tokens": 4,
        "input_cost_nanodollars": 5000,
        "output_cost_nanodollars": 5000,
        "latency_ms": 475,
        "error": None,
    }


def test_reads_per_draw_files_when_there_is_no_parent(tmp_path):
    # A run that died keeps its finished draws only in per-draw files, and
    # the files it was writing are empty (1.2).
    d = run_dir(tmp_path)
    write_per_draw(d, 0, sub_run(0))
    write_per_draw(d, 1, None)
    write_per_draw(d, 2, sub_run(2))
    assert [r["draw_index"] for r in read_run(d)] == [0, 2]


def test_the_parent_is_the_only_source_when_present(tmp_path):
    d = run_dir(tmp_path)
    write_parent(d, [sub_run(0), sub_run(1)])
    write_per_draw(d, 0, sub_run(0))
    write_per_draw(d, 7, sub_run(7))
    assert [r["draw_index"] for r in read_run(d)] == [0, 1]


def test_a_caught_error_is_kept(tmp_path):
    d = run_dir(tmp_path)
    write_parent(d, [sub_run(0, error="APITimeoutError: Request timed out.")])
    (record,) = read_run(d)
    assert record["error"] == "APITimeoutError: Request timed out."
    assert record["reply"] is None
    assert record["prompt_id"] == PLAIN.id
    assert record["output_tokens"] is None


def test_an_uncaught_error_gets_its_index_from_the_file_name(tmp_path):
    d = run_dir(tmp_path)
    write_per_draw(d, 5, errored_sub_run(prompt=UNIFORM))
    (record,) = read_run(d)
    assert record["draw_index"] == 5
    assert record["prompt_id"] == UNIFORM.id  # found from the text sent
    assert record["error"].endswith("RuntimeError: boom")


def test_an_uncaught_error_in_a_parent_goes_last_without_an_index(tmp_path):
    d = run_dir(tmp_path)
    write_parent(d, [errored_sub_run(), sub_run(1), sub_run(0)])
    records = read_run(d)
    assert [r["draw_index"] for r in records] == [0, 1, None]


def test_reasoning_traces_are_kept(tmp_path):
    d = run_dir(tmp_path)
    run = sub_run(0)
    run["conversations"][0]["requests"][0]["reasoningTraces"] = "Let me think."
    write_parent(d, [run])
    assert read_run(d)[0]["reasoning"] == "Let me think."


def test_an_unset_version_is_none(tmp_path):
    d = run_dir(tmp_path, version="unset")
    write_parent(d, [sub_run(0)])
    assert read_run(d)[0]["task_version"] is None


# Rejected shapes


def two_requests(run):
    requests = run["conversations"][0]["requests"]
    requests.append(requests[0])


def wrong_prompt_text(run):
    run["conversations"][0]["requests"][0]["contents"][0]["parts"][0]["text"] = UNIFORM.text


def unknown_prompt_id(run):
    run["results"][0]["dictResult"]["prompt_id"] = "plain-00000000"


def reply_differs(run):
    run["results"][0]["dictResult"]["reply"] = "48"


def extra_result_key(run):
    run["results"][0]["dictResult"]["value"] = 47


def three_contents(run):
    contents = run["conversations"][0]["requests"][0]["contents"]
    contents.append(contents[1])


def two_parts(run):
    parts = run["conversations"][0]["requests"][0]["contents"][1]["parts"]
    parts.append(parts[0])


def unknown_state(run):
    run["state"] = "BENCHMARK_TASK_RUN_STATE_RUNNING"


@pytest.mark.parametrize(
    "change",
    [two_requests, wrong_prompt_text, unknown_prompt_id, reply_differs,
     extra_result_key, three_contents, two_parts, unknown_state],
)
def test_rejects_an_unexpected_shape(tmp_path, change):
    d = run_dir(tmp_path)
    run = sub_run(0)
    change(run)
    write_parent(d, [run])
    with pytest.raises(ExtractError):
        read_run(d)


def test_rejects_an_unknown_prompt_text_in_an_uncaught_error(tmp_path):
    d = run_dir(tmp_path)
    run = errored_sub_run()
    wrong = run["conversations"][0]["requests"][0]["contents"][0]["parts"][0]
    wrong["text"] = "Pick a number."
    write_per_draw(d, 0, run)
    with pytest.raises(ExtractError):
        read_run(d)


def test_rejects_a_draw_in_another_draws_file(tmp_path):
    d = run_dir(tmp_path)
    write_per_draw(d, 3, sub_run(4))
    with pytest.raises(ExtractError):
        read_run(d)


def test_rejects_a_repeated_draw_index(tmp_path):
    d = run_dir(tmp_path)
    write_parent(d, [sub_run(0), sub_run(0)])
    with pytest.raises(ExtractError):
        read_run(d)


def test_rejects_a_truncated_file(tmp_path):
    d = run_dir(tmp_path)
    (d / "draw-run_param_id_0_x.run.json").write_text('{"state": "BENCH')
    with pytest.raises(ExtractError):
        read_run(d)


# Paths


def test_parses_the_download_path(tmp_path):
    path = parse_run_path(run_dir(tmp_path))
    assert (path.task, path.version, path.model, path.run_id) == (
        "randombench-collect", 2, "gpt-5.4-nano-2026-03-17", "6807323")


@pytest.mark.parametrize("local", ["results/2.4-local", "a/b/c/Run_1", "x/v2/model/123"])
def test_rejects_paths_that_arent_kaggle_downloads(local):
    with pytest.raises(ExtractError):
        parse_run_path(local)


def test_finds_every_run_in_a_task_download(tmp_path):
    a = run_dir(tmp_path, version="1", run_id="100")
    b = run_dir(tmp_path, version="2", model="gemini-3.5-flash-lite", run_id="200")
    assert find_run_dirs(tmp_path / "results" / "randombench-collect") == [a, b]


# Writing and loading


def test_extracting_a_run_twice_stores_it_once(tmp_path):
    d = run_dir(tmp_path)
    write_parent(d, [sub_run(1, reply="Sure! **47** 🎲"), sub_run(0)])
    data = tmp_path / "data"
    file, records = extract_run(d, data)
    assert file == data / "gpt-5.4-nano-2026-03-17" / "6807323.jsonl"
    first = file.read_bytes()
    extract_run(d, data)
    assert file.read_bytes() == first
    assert load_draws(data) == records
    assert "🎲" in file.read_text(encoding="utf-8")  # stored as written, not escaped


def test_a_run_with_no_draws_writes_nothing(tmp_path):
    d = run_dir(tmp_path)
    write_per_draw(d, 0, None)
    data = tmp_path / "data"
    assert extract_run(d, data) == (None, [])
    assert not data.exists()


def test_load_rejects_a_draw_stored_twice(tmp_path):
    d = run_dir(tmp_path)
    write_parent(d, [sub_run(0)])
    data = tmp_path / "data"
    file, _ = extract_run(d, data)
    copy = data / "renamed-model" / file.name
    copy.parent.mkdir()
    copy.write_bytes(file.read_bytes())
    with pytest.raises(ExtractError):
        load_draws(data)
