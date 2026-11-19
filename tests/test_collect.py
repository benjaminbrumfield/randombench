"""The collection task, run offline with stand-in models (PLAN.md 4.1).

No model is reached: a stand-in answers every prompt. The tests run the
task file's own code, minus its last two cells, which run the task on
kbench.llm and hold the %choose magic.
"""

import os
import runpy
import shutil
import threading
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import kaggle_benchmarks as kbench
from kaggle_benchmarks.actors.llms import LLMResponse
from kaggle_benchmarks.kaggle.model_proxy import ModelProxy

from randombench.extract import read_run

TASK = Path(__file__).resolve().parent.parent / "tasks" / "collect.py"


class StandIn(kbench.LLMChat):
    """A model that answers offline and remembers every prompt it gets."""

    def __init__(self, name: str, reply: str = "42"):
        super().__init__(name=name)
        self.reply = reply
        self.prompts = []

    def invoke(self, messages, **kwargs):
        self.prompts.append(messages)
        return LLMResponse(content=self.reply)


@pytest.fixture
def task(tmp_path, monkeypatch):
    """The task file's namespace, without the cells that run it. Run files land in tmp_path."""
    cells = TASK.read_text(encoding="utf-8").split("\n# %%\n")
    body = [c for c in cells if "collect.run(" not in c and "# %choose" not in c]
    assert len(body) == len(cells) - 2, "expected the last two cells to run the task and %choose it"
    code = tmp_path / "collect_cells.py"
    code.write_text("\n# %%\n".join(body), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return runpy.run_path(str(code))


def test_the_allowlist_holds_38_distinct_models(task):
    assert len(task["MODELS"]) == len(set(task["MODELS"])) == 38


def test_the_servers_default_model_is_never_prompted(task):
    # Gemini 3.8 Flash's draws are done, so a push's creation run prompts
    # nothing (D23).
    default = StandIn("google/gemini-3.8-flash")
    run = task["collect"].run(default)
    assert run.result == 0
    assert default.prompts == []


def test_the_local_default_model_isnt_listed(task):
    # A plain local run uses .env's LLM_DEFAULT, which kaggle_benchmarks loads
    # on import. It must stay off the list, or the run spends quota (D12, D23).
    default = os.environ.get("LLM_DEFAULT")
    if not default:
        pytest.skip("no LLM_DEFAULT, so there's no .env to check")
    assert default not in task["MODELS"]


def test_a_listed_model_gets_every_draw(task):
    nano = StandIn("openai/gpt-5.4-nano-2026-03-17")
    run = task["collect"].run(nano)
    assert task["MIN_DRAWS"] <= len(nano.prompts) <= task["MAX_DRAWS"]
    assert run.result == len(nano.prompts)  # draws with a reply (D14)


def test_each_run_picks_its_own_number_of_draws(task, monkeypatch):
    # random.randint picks it, so a run's size can be set here without
    # changing the task.
    picked = []
    monkeypatch.setattr(task["random"], "randint", lambda a, b: picked.append((a, b)) or 53)
    nano = StandIn("openai/gpt-5.4-nano-2026-03-17")
    assert task["collect"].run(nano).result == 53
    assert picked == [(100, 300)]


def test_each_draw_gets_a_prompt_at_random(task):
    ids = {p.id for p in task["PROMPTS"]}
    assert len(task["random_schedule"](53)) == 53
    # The chance that 600 draws miss one of the six prompts is about 2 in 10^47.
    per_prompt = Counter(task["random_schedule"](600))
    assert set(per_prompt) == ids
    # Not a balanced shuffle (D9): exactly 100 each has a chance of about 2.5 in 10 million.
    assert len(set(per_prompt.values())) > 1


def test_the_run_files_extract(task, tmp_path):
    # Lay the local run files out as `kaggle b t download` would, and read
    # them with the extractor: the task's output and extract.py must agree.
    task["collect"].run(StandIn("openai/gpt-5.4-nano-2026-03-17", reply="**37**"))
    run_dir = tmp_path / "results" / "randombench-draws" / "unset" / "gpt-5.4-nano-2026-03-17" / "1"
    run_dir.mkdir(parents=True)
    for f in tmp_path.glob("*.run.json"):
        shutil.move(str(f), run_dir)

    records = read_run(run_dir)
    assert task["MIN_DRAWS"] <= len(records) <= task["MAX_DRAWS"]
    assert [r["draw_index"] for r in records] == list(range(len(records)))
    assert {r["model"] for r in records} == {"openai/gpt-5.4-nano-2026-03-17"}
    assert {r["reply"] for r in records} == {"**37**"}
    # Every draw's prompt is one of the six.
    assert {r["prompt_id"] for r in records} <= {p.id for p in task["PROMPTS"]}


# What the pilot reads in the run's log (4.1)


def test_every_run_prints_the_servers_facts(task, capsys):
    # Even a run that prompts nothing.
    task["collect"].run(StandIn("google/gemini-3.8-flash"))
    out = capsys.readouterr().out
    assert "kaggle_benchmarks 0.6.2" in out
    assert "KAGGLE_AGENT_LLM_REASONING_EFFORT=" in out
    assert "no draws" in out


def test_a_stand_in_isnt_counted(task, capsys):
    task["collect"].run(StandIn("openai/gpt-5.4-nano-2026-03-17"))
    assert "HTTP requests: not counted" in capsys.readouterr().out


def fake_model(client):
    """Something shaped like a ModelProxy: llm.client._client is its httpx client."""
    return SimpleNamespace(client=SimpleNamespace(_client=client))


def test_the_request_counter_sees_every_request_from_every_thread(task):
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200)))
    with task["RequestCounter"](fake_model(client)) as requests:
        assert requests.counting
        threads = [threading.Thread(target=lambda: [client.get("https://proxy.test/") for _ in range(5)])
                   for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert requests.count == 20
    # The hook comes off afterwards, so a later run doesn't count twice.
    client.get("https://proxy.test/")
    assert requests.count == 20
    assert client.event_hooks["request"] == []


def test_the_request_counter_finds_a_model_proxys_hooks(task):
    # A real ModelProxy, as kbench.llm is on Kaggle. Building one sends nothing.
    llm = ModelProxy(model="openai/gpt-5.4-nano-2026-03-17", api_key="unused", base_url="http://127.0.0.1:9", api="openai")
    assert task["RequestCounter"](llm).counting
