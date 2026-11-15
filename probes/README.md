# Probes

Small throwaway tasks that tested how Kaggle Benchmarks behaves before the real tasks relied on it. They're kept as the record of what [PLAN.md](../PLAN.md) found, not maintained, and their Kaggle tasks have been deleted. Each file's header says what it tested.

| Probe | PLAN.md step | What it tested | Model Proxy calls |
|---|---|---|---|
| [fake_draws.py](fake_draws.py) | 1.1, 1.1b, 1.2 | How a 600-row sub-task evaluation is recorded, failed rows included | none |
| [fake_draws_crash.py](fake_draws_crash.py) | 1.2 | What a run keeps when its process dies partway | none |
| [fake_primes.py](fake_primes.py) | 1.3 | A scoring task that reads embedded draws and returns a score with an interval | none |
| [one_draw.py](one_draw.py) | 2.3 | Whether a null seed is accepted, and where the reply, reasoning, and token usage land | 2 |
| [temperature.py](temperature.py) | 2.4b | Whether Gemini 3.1 Flash-Lite's 42s change with temperature 1.0 sent explicitly | 12 |
| [fake_sd.py](fake_sd.py) | 4.13 | Whether a result's standard deviation makes Kaggle show its interval | none |
| [fake_ci.py](fake_ci.py) | 4.13 | A task whose value is another task's interval | none |
| [fake_extra.py](fake_extra.py) | 4.13 | Whether more tasks than models changes a Benchmark's layout | none |

`one_draw.py` and `temperature.py` prompt real models, so running them spends quota. Both import `randombench`, so they run only locally, with the repo root on `PYTHONPATH`.
