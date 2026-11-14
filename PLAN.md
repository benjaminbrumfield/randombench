# RandomBench plan

This is how RandomBench gets built, one small step at a time. Each step is one commit and comes with something to learn. Steps marked **quota** make LLM calls through Kaggle's Model Proxy. They wait for an explicit go-ahead, and each one states its call count up front.

## Goal

Measure how well LLMs generate random integers in [0, 99]. Each model answers hundreds of independent single-turn prompts, and its answers are checked for known biases: favoring primes, round numbers, and so on. Each check yields a sub-score in [0, 1], and a headline score combines them.

## Decided

- **Range:** integers 0–99.
- **Names:** the Kaggle Benchmark is called RandomBench. A Benchmark is the page that groups tasks and shows the leaderboard. The task gets a different name (D10). Benchmark and task slugs are both scoped to their owner (`/benchmarks/{owner}/{slug}` and `/benchmarks/tasks/{owner}/{slug}`), so nobody else can take ours and there's nothing to reserve early. Benchmarks are created on the Kaggle website; the CLI handles only tasks.
- **Calls:** single-turn, in a fresh conversation each time. No tools, and the model never sees its earlier answers.
- **Replies:** plain text for every model, read by one parser of our own (2.2). No structured output (`schema=`). kaggle-benchmarks 0.6.2 sends a schema natively to most models, but gemma, qwen, deepseek and meta models get it only as text in a system message (`kaggle_benchmarks/kaggle/model_proxy.py`, lines 73–79). The models would then answer under different conditions, and their invalid-answer rates couldn't be compared. The prompt states the range, and an answer outside it counts as invalid.
- **Prompts:** six, in [randombench/prompts.py](randombench/prompts.py), each with a name, an author and its text. D9 decides which prompt each draw gets.
  - **Ids:** a prompt's id is its name plus the first 8 hex digits of its text's SHA-256, such as `plain-47858739`. Editing the text gives a new id by itself, and a test pins every id, so no edit goes unnoticed, not even to whitespace. Once a prompt has been used in a run, it's never edited or deleted, and a change means adding a new prompt. A way to retire a prompt can wait until one first needs replacing.
  - **Authors:** Claude drafted all six (`claude-opus-5-5`). A prompt the user rewords lists `benjaminbrumfield` instead.
  - **Style:** every prompt asks for a random integer, gives the range as 0 to 99 with both ends included, and asks for the number only. A model that wasn't asked to be random can't be faulted for having a favorite. No prompt gives an example number or names a bias, since either could steer the answer. Beyond that, the prompts differ in wording, tone, length and framing, so no score depends on a single phrasing. A test checks the rules code can check.
- **Volume:** 600 draws per model per run. That gives a 95% margin of error of at most ±4 percentage points on any share a check measures, and 6 expected answers per value, above the usual minimum of 5 for a chi-squared test. Halving the margin would take 4× the draws. 600 also splits evenly across six prompts (D9). If the pilot shows spare quota, 1,200 is the next step up: ±2.8 points for twice the calls. Failed calls are recorded, not dropped, and a run that stops early keeps what it collected.
- **Raw data:** every raw response is stored. Parsed values and scores are always recomputed from the raw text, so a new bias check or a new parsing rule never requires calling the models again. [randombench/extract.py](randombench/extract.py) stores each Kaggle run as one JSON Lines file, `data/<model>/<run_id>.jsonl` (D5).
- **Parsing:** [randombench/parsing.py](randombench/parsing.py) reads a reply leniently on formatting and strictly on content (D4). Markdown, LaTeX, punctuation and words are ignored. The numbers written in digits must all be the same integer, and it must be in 0–99. So `Sure! **47**.` is 47, and `47 or 52`, `3.5`, `47.0`, `100` and `forty-seven` are invalid. A reply that restates the range, as in `from 0 to 99: 47`, holds three integers and is invalid too, because the prompt asked for the number only. If a provider puts its reasoning into the reply text, every number in the reasoning would count. 2.3 shows where reasoning ends up.
- **Invalid answers:** the rate of invalid answers is its own sub-score.
- **Shared code:** a push uploads only the task file, so each task file holds a copy of the `randombench` modules it uses, refreshed by `python -m randombench.sync` (D6).
- **Leaderboard:** every sub-score and the headline appear on Kaggle, and no model is ever called twice for the same draw (D2, settled in 1.2). A notebook puts only one task on the leaderboard, and a dict result shows a blank Result. So the collection task makes each model's draws, and each score gets a scoring task of its own. A scoring task never prompts a model: run with `-m X`, it recomputes its score from X's stored draws and returns it. 1.2 showed that a task can be run against a model without prompting it. The RandomBench Benchmark groups all the tasks.
- **Temperature:** the provider's default. This was not our choice: kaggle-benchmarks 0.6.2 never sends temperature through Model Proxy (`kaggle_benchmarks/kaggle/model_proxy.py`, line 101). With temperature 1.0 sent explicitly, Gemini 3.1 Flash-Lite answered exactly as it does with none sent: 42 every time (2.4b, D16).
- **Seed:** `seed=None`, so providers sample freshly, as in normal API use. The library's default seed is 0, and it is sent to every non-Google model. A backend that honors seeds could then return the same number every time, which would look like a bias. GPT-5.4 nano accepted a null seed (2.3). The library never sends a seed to a `google/` model, so the two Gemini models and Gemma sample freshly regardless. Claude Haiku 4.5 and gpt-oss-120b get checked in the pilot (4.1). If a backend rejects a null seed, the fallback is a distinct seed on every call.
- **Models:** six, with their exact ids. Kaggle slugs come from `kaggle b t models`, and local ids from `LLMS_AVAILABLE` in `.env` (both checked 2026-11-14). Local development uses the first two. gpt-oss-120b is also available locally.

| Model | Kaggle slug (`-m`) | Local id |
|---|---|---|
| GPT-5.4 nano | `gpt-5.4-nano-2026-03-17` | `openai/gpt-5.4-nano-2026-03-17` |
| Gemini 3.1 Flash-Lite (preview) | `gemini-3.1-flash-lite-preview` | `google/gemini-3.1-flash-lite-preview` |
| Gemini 3.5 Flash-Lite | `gemini-3.5-flash-lite` | — |
| Claude Haiku 4.5 | `claude-haiku-4-5-20251001` | — |
| Gemma 4 26B A4B | `gemma-4-26b-a4b-it` | — |
| gpt-oss-120b | `gpt-oss-120b` | `openai/gpt-oss-120b` |

## Bias checks (planned, not implemented)

A perfect generator gives each value probability 1/100, so the expected rates are:

| Check | What it measures | Expected from a perfect generator |
|---|---|---|
| Invalid answers | Share of replies that aren't a single integer in [0, 99] | 0% |
| Uniformity | How far the whole distribution is from uniform | Calibrated (see below) |
| Primes | Share of answers that are prime (2, 3, 5, …, 97: 25 values) | 25% |
| Round numbers | Share of multiples of 10, and separately of 5 | 10% (20%) |
| Favorite number | Share of answers taken by the single most common value | Calibrated |
| Edges | Share of answers in 0–9 or 90–99 | 20% |
| Repeated digits | Share of 11, 22, …, 99 | 9% |
| Last digit | How far the last-digit frequencies are from 10% each (7 is the usual suspect) | 10% each |

"Calibrated" means even a true random generator won't hit the ideal value exactly in 600 draws. We'll measure what Python's generator scores at the same sample size and compare models against that.

Some checks overlap. For example, 7, 17, 37, 47, 67 and 97 are all primes ending in 7. This matters when combining sub-scores into the headline.

## Steps

### Phase 0: groundwork

- [x] **0.1** Commit the environment fixes: the `hishel` pin (`91ef7a4`) and `PYTHONUTF8=1` in `CLAUDE.md` (`38cfc5f`).
- [x] **0.2** Commit this plan (`7c8488d`).
- [x] **0.3** Set the working method: quiz blanks for offline code (in `CLAUDE.md`), pytest in `requirements-dev.txt`, exact model ids, and plain-text replies for every model. Quiz blanks were dropped after 2.4b, and Claude now writes all the code. No committed code ever held a blank.

### Phase 1: learn the platform without spending quota

- [x] **1.1 Fake-draw task, run locally.** [probes/fake_draws.py](probes/fake_draws.py) is a throwaway task whose "draws" come from Python's `random`, not from an LLM:
  - A sub-task runs 600 times through `.evaluate(on_failure="continue")`, with failures injected on purpose at 6 rows.
  - A parent task returns a dict of made-up sub-scores.

  *Learn:* what a `.run.json` contains, where sub-task results end up, and how failures are recorded. It makes no LLM calls.

  *Found:*
  - The parent's `.run.json` embeds all 600 sub-runs, each with its state, result, conversation and timing. Each sub-run is also written to its own file as it finishes, named by its row (`param_id_37`). The run took 5 seconds and left 602 files (3.4 MB) in the working directory.
  - A sub-run record doesn't hold its parameters. With `n_jobs=8`, the `subruns` array is in completion order (305 of 600 out of place). Inside the parent file, only a draw's own result can tie it to its row and prompt. This raises D11.
  - A failed sub-run has state `ERRORED` and the full traceback in `errorMessage`. Its result is replaced by `booleanResult: false`.
  - Numbers in a dict result become floats (`594` is stored as `594.0`), because the dict is stored as a protobuf `Struct`. Extraction must convert them back.
  - Every sub-run repeats the sub-task's source code in `taskVersion`, which made up 42% of the parent file here. This matters for D5.
  - From the library source, not this run: `.evaluate()` inside a parent task forces `max_attempts=1`, so the library won't retry failed draws in our setup. This matters for D3.
  - The parent records the model as `google/gemini-3-flash-preview` (`LLM_DEFAULT`), even though it never called it. Fixed in 1.1b.
- [x] **1.1b Label the probe's model honestly.** The library labels a run with the first `LLMChat` passed to the task (`evaluated_subject` in `kaggle_benchmarks/runs.py`), whether or not that model was prompted. The label goes into `modelVersion.slug` and into every file name. The probe now passes a stand-in, `PythonRandom`, named `python-random`, whose `invoke` raises, so a stray prompt fails loudly instead of reaching a model.

  *Found:*
  - All 601 records (the parent and 600 sub-runs) and all their file names now say `python-random`.
  - The probe runs without `.env`: with no Model Proxy credentials loaded, it can't spend quota.
  - In the real task the label will be right, because the model passed in is the model prompted.
- [x] **1.2 Fake-draw task, on Kaggle.** Push and run the same task, plus a variant that crashes halfway through. `kaggle b t run` needs a model, but the task never calls it. Because this is a server run, it waits for a go-ahead anyway. The server doesn't support `kaggle b t delete` yet, so the probe stays on the account as a private task.

  *Learn:*
  - which model the leaderboard shows for a run labelled `python-random`: the stand-in, the `-m` model, or neither because the run is rejected. If it's rejected, run again with `kbench.llm` so D2 can still be settled.
  - how a dict result appears on the leaderboard (settles D2)
  - what `kaggle b t download` returns, and whether it includes the 600 per-draw files and the ATIF companion files (`.atif.json`, `.result.json`) that are switched off locally
  - whether a run that crashes partway keeps anything (informs D3)

  *Found:*
  - A push runs the whole task once to create it, against a model the server picks: `gemini-3.8-flash`, which isn't in our table. The task can be run with `-m` only if this creation run succeeds, and it's listed with the task's other runs. This raises D12.
  - The run labelled `python-random` completed, and then the server failed the task with "Not found". It accepts only models it serves. On Kaggle the probe now passes `kbench.llm`, the `-m` model, which it never prompts. It detects Kaggle the way the library does (`KAGGLE_KERNEL_RUN_TYPE=batch`) and keeps `python-random` for local runs. Version 2 then completed, and so did its `-m gpt-5.4-nano-2026-03-17` run. All 601 conversations in each run are empty, so no model was prompted.
  - Without `%choose`, `kaggle b t download` returns, for each run, the parent `.run.json` (0.6 MB), the 600 per-draw files (0.5 MB in all) and the `.task.json`. There are no ATIF files on the server either. `-s` adds `__notebook__.ipynb`, 9.9 MB, nearly all of it the console UI's output.
  - Without `%choose`, the task page showed `fake-draw`: one draw's run, with the sub-task's code, no description and a blank Result. At the end of the notebook, `%choose <task>` deletes every `.run.json` and `.task.json` except the task's own and its latest run (`kaggle_benchmarks/ui/ipython_magics.py`). The `.py` file can carry it as the comment `# %choose randombench-fake-draws`, because jupytext turns it into the magic when the CLI converts the file. Other comments therefore must not start with `# %`. Version 3 adds it, and its download holds only the parent `.run.json` and the `.task.json` (0.6 MB). The real task needs it too. A run that dies never reaches `%choose`, so it keeps its per-draw files.
  - With `%choose`, the task page shows the parent's title, description and code. Its Result cell for the dict result is blank. A task can also return a bool, a number, or a `(value, confidence interval)` pair (`_prepare_results_data` in `kaggle_benchmarks/kaggle/serialization.py`). We haven't seen how those display.
  - The crash variant killed the kernel at draw 300 during creation (`DeadKernelError`). The task version is `Errored`, so it can't be run. Its download still holds 300 per-draw files: 294 completed draws, the 3 injected failures, and 3 empty files from draws that were mid-write. The other draws in flight left nothing, and there's no parent file, so the run has no result. The same crash run locally leaves 295 per-draw files and no parent. This informs D3: a run that dies keeps its finished draws, but only in the per-draw files.
- [x] **1.3 Scoring-task probe, on Kaggle.** A throwaway scoring task holds fake draws for two model labels. It picks the draws for the model it's run with and returns a `(value, CI)` pair. Push it, run it with `-m gpt-5.4-nano-2026-03-17`, and add it with `randombench-fake-draws` to a private probe Benchmark, which you create on the Kaggle website. It never prompts a model, but its server runs wait for a go-ahead.

  *Learn:*
  - how a `(value, CI)` result displays on the leaderboard
  - whether a Benchmark shows each task as a column
  - how a task finds its model's draws, through `llm.name` or something else
  - what a creation run with no stored draws for its model should return (D12)
  - whether draws embedded in the task file are workable (D13)

  *Found:*
  - [probes/fake_primes.py](probes/fake_primes.py) embeds 600 made-up draws for each of two models, as plain lists of ints: 7.7 KB in all. Annotated `-> tuple[float, float]`, it returns the share of primes and the 95% half-width. Run locally, its run file stores them as `numericResult` with a `value` and a `confidenceInterval`.
  - A task finds its model's draws through `llm.name`. On Kaggle that's the full id with its provider prefix, `google/gemini-3.8-flash` in the creation run. It's the same string as the run's label and as the local ids in our model table. Key stored draws by it.
  - Raising in the creation run fails the push. Version 1 raised `LookupError` for `gemini-3.8-flash`, which has no stored draws. The server marked the version `Errored`, so it can't be run with `-m`. A scoring task's creation run must return a result even without draws for its model (D12).
  - Returning `False` instead works. Version 2's creation run completed with `booleanResult: false`, and its `-m gpt-5.4-nano-2026-03-17` run completed with `numericResult` 0.4 ± 0.0392 from nano's embedded draws. Both conversations hold no requests, so no model was prompted. The library only warns that `False` doesn't match `-> tuple[float, float]`, but its display formats such a result as a pair and crashes on `False` after the run file is written. The probe therefore assigns the run (`run = fake_primes.run(...)`), so the notebook doesn't display it.
  - Nothing in the environment marks a creation run. Both runs have the same variable names and the same `LLMS_AVAILABLE`. `LLM_DEFAULT` and `KAGGLE_AGENT_LLM` hold the model being run, and `LLM_DEFAULT_EVAL` is `google/gemini-3.8-flash` in both runs. So the only signal is the one D12 already leans on: the model is the server's default. A `-m gemini-3.8-flash` run would look the same.
  - `LLMS_AVAILABLE` lists 37 models. Ours appear as `openai/gpt-5.4-nano-2026-03-17`, `google/gemini-3.1-flash-lite-preview`, `google/gemini-3.5-flash-lite`, `anthropic/claude-haiku-4-5@20251001`, `google/gemma-4-26b-a4b` and `openai/gpt-oss-120b`. Two differ from their `-m` slugs. Only nano's is confirmed as a run's `llm.name`. Keying stored draws by the label each collection run records (`modelVersion.slug`) avoids a mapping table: that label is built from the same attribute the scoring task looks up.
  - A `(value, CI)` result shows only its value. The task page shows nano's result as `0.40`, and the Benchmark shows it as `40.0%`. Neither table shows the interval, though the run file stores it.
  - `False` shows as a red `✕ FAIL` in both views.
  - Our private Benchmark's page puts tasks in rows and models in columns. Each cell holds one task's result for one model, and a dict result's cell is blank. Each row is pinned to a task version (`v3`, `v2`) and shows only that version's runs. Nano's `randombench-fake-draws` run was on v2, so its v3 cell offers a ▷ run button instead. A new version of a scoring task therefore needs a fresh run on every model, as 4.4 counts.
  - Each model column has a number in its header: `0.00` for Gemini 3.8 Flash (one FAIL and one blank) and `-` for nano (one result missing). The "Top Models" sidebar seems to rank by it. It looks like the Benchmark's own aggregate across tasks, but how it's computed is unknown (D15).
  - Two public Benchmarks, APEX-Agents 1.1 and ArXivMath August 2026 (seen 2026-11-14), turn this around. Models are in rows, ranked by an average column labelled "Avg N tasks", and tasks are further columns. APEX-Agents averages 3 tasks but shows more task columns (Pass^4, Pass@4, Mean Score – Corporate Law, …). So a Benchmark seems able to average a chosen subset of its tasks (D15).
  - ArXivMath shows a "95% CI (±%)" column beside its average. Every row matches 1.96 × √(p(1 − p)/171): an interval for a share of 171 pass/fail trials, which is 57 tasks × 3. So a leaderboard can show an interval. This one looks computed from many pass/fail results, not read from tasks' `(value, CI)` results. Whether our intervals can appear there is unknown (D15).
  - A Benchmark's Settings choose its "Overall score":
    - average of task scores
    - weighted average of task scores
    - weighted average of task scores, with confidence interval (likely what ArXivMath uses)
    - percentage of tasks passed, the default, which gave Gemini 3.8 Flash `0.00`
    - percentage of tasks at full score
    - no overall score

    Settings also say "All leaderboard tasks must be public to change visibility" (D15).
  - The overall score was then switched to "weighted average of task scores, with confidence interval", with only `randombench-fake-primes` left on the Benchmark. Nano's header shows `0.40` with no interval, and the model columns are ordered by it. The private page shows no interval, so whether our `(value, CI)` reaches the overall score stays open until RandomBench is public (5.2). Gemini 3.8 Flash's FAIL gives `-`, not 0. An average of scores leaves out a boolean result, so a creation run's FAIL doesn't count against its model (D12, D15).
  - A task row's `⋮` menu on the Benchmark page runs that task for all its models, or only for those missing results (4.4).
  - The task page's code panel shows the task function, starting at `@kbench.task`, not the module-level `STORED_DRAWS`. Embedded draws stay out of view there; View Notebook shows the whole file (D13).
  - The decorator's `description` is the subtitle under the task's title. The page's Description section is a separate field, edited on the website, and it's empty.
  - The private Benchmark says to "add a model and public tasks to share it with others". A public RandomBench therefore needs every one of its tasks published (5.2).

### Phase 2: from one call to a dataset

- [x] **2.1 Prompts.** Add the six or more prompts, each with a stable id and an author. Rule: editing a prompt's text gives it a new id, so every stored draw points at the exact text it answered.
- [x] **2.2 Parser and tests.** Write a function that turns a raw reply into an integer or "invalid", with unit tests on sample replies: `47`, `**47**`, `Sure! 47`, `47 or 52`, `100`, `3.5` and a refusal. No quota. Settles D4.
- [x] **2.3 One real draw.** **Quota: 2 calls**, one per local model, with `seed=None`. [probes/one_draw.py](probes/one_draw.py) sends the `plain` prompt once to each local model, with reasoning left at the provider default (D1).

  *Learn:* whether a null seed is accepted, and where the raw reply, the reasoning trace and the token usage appear in the run file.

  *Found:*
  - Both calls completed, 2 in all. Nano answered 37 and Gemini 3.1 Flash-Lite answered 42, two of the best-known "random" numbers, though one draw each shows nothing yet.
  - Nano accepted `seed=None`, which the library passes on to the OpenAI client as `seed`. Gemini got no seed, because the library drops it for every `google/` model (`_should_remove_seed` in `kaggle_benchmarks/actors/proxy_openai.py`). The run file doesn't record request parameters such as seed, temperature or reasoning. Only the code shows what was sent.
  - The raw reply is in `conversations[0].requests[0].contents[1].parts[0].text`. `contents[0]` holds the prompt text exactly as sent, so each draw's record carries the text it answered.
  - With the default reasoning, no trace came back. `last_reasoning_traces()` returned `None`, and the contents hold only the prompt and the reply. The library asks for traces only when `reasoning=` is passed: it sets `include_thoughts` for Google models and parses `<think>` tags only then (D1).
  - Token usage is in each request's `metrics`: `inputTokens`, `outputTokens`, the input and output costs in nanodollars, and `totalBackendLatencyMs`. The costs and latency are JSON strings (`"5000"`), because protobuf writes 64-bit integers as strings, so extraction must convert them. There's no separate count of reasoning tokens. Nano used 4 output tokens for `37` and Gemini 2 for `42`.
  - A draw cost about $0.00001 on these two models, so 600 draws cost under a cent per model. Reasoning models may cost much more (4.1).
  - From the library source: the HTTP client can cache responses, keyed by the request body (`build_httpx_client` in `kaggle_benchmarks/utils.py`). It's off unless `ENABLE_LOCAL_CACHING` is set, and our `.env` doesn't set it. If it were on, every draw with the same prompt would get the first draw's answer back. The collection task must keep it off, and a Kaggle run should confirm that the server doesn't turn it on. It's separate from `kbench.client.enable_cache()`, which only skips a run whose run file already completed.
- [x] **2.4 Small collection run.** **Quota: 24 calls**, 12 on each local model. This is the real collection task at a small size. Settles D9.

  *Learn:* how parallel calls and retries behave, and whether each draw's record is complete.

  *Found:*
  - [tasks/collect.py](tasks/collect.py) is the collection task, here at 12 draws. A shuffled balanced schedule gives each prompt 2 draws per model (D9). The draw sub-task returns its draw index and prompt id together with the reply, or with the error it caught (D11). The parent returns the number of draws with a reply (D14). It parses nothing, so the pushed file will need the prompts but not the parser (D6). It imports `randombench`, so for now it runs only locally, and its name is provisional (D10).
  - Both runs completed with 12 replies each, 24 calls in all. Each took about 2 seconds with 4 workers. No draw failed, so the error path hasn't been tried.
  - Each per-draw file stands on its own. It holds the model (`modelVersion.slug`), the prompt text and the reply (the conversation), the token metrics, and the draw index, prompt id and reply (the result). The parent file embeds the same 12 records. The sub-runs again finished out of order (`[0, 2, 3, 1, …]`), and the draw index in each result is what ties a draw to its row. As in 1.1, `draw_index` is stored as a float (`0.0`). The parent's `-> int` result is stored as `numericResult` 12.0.
  - Retries: nothing failed, so nothing was retried. Each conversation holds exactly one request. The OpenAI client can retry a failed request up to twice on its own (its default `max_retries`), and a retry wouldn't show in the run file. `OPENAI_LOG=info` printed nothing here, so the actual number of HTTP requests is unverified.
  - Gemini 3.1 Flash-Lite answered 42 on all 12 draws, under all six prompts. Nano gave 42 five times, 37, 47 and 73 twice each, and 57 once. All 24 replies were bare numbers, so all of them parse.
  - The twelve 42s aren't from a cache. The HTTP cache is off, and the six prompts make different request bodies. Either they reflect the model's own distribution, or Model Proxy applies a low temperature when none is sent. This run can't tell those apart (D16).
- [x] **2.4b Temperature probe.** **Quota: 12 calls**, all on Gemini 3.1 Flash-Lite. [probes/temperature.py](probes/temperature.py) repeats 2.4's 12 draws with `temperature=1.0` sent explicitly. It builds the model with `ModelProxy(..., support_temperature=True)` and records the body of every HTTP request it sends, but none of the headers, which hold the API key.

  *Learn:* whether the 42s come from the model or from Model Proxy's default temperature (D16), and how many HTTP requests a draw really takes.

  *Found:*
  - All 12 draws answered 42 again, under all six prompts. Sending temperature 1.0 changed nothing. So the 42s come from the model, unless Model Proxy ignores the parameter, which this probe can't rule out (D16).
  - Exactly 12 HTTP requests went out for 12 draws, so nothing was retried. Each body held only `model`, `messages` and `temperature: 1.0`. There was no `seed`, as 2.3 predicted for `google/` models. The library's unexplained temperature bug (b/439876083) didn't show up for this model.
  - An httpx request hook (`llm.client._client.event_hooks`) counts requests reliably where `OPENAI_LOG` printed nothing (2.4). The pilot can use it to check for hidden retries.
- [x] **2.5 Extract and store.** [randombench/extract.py](randombench/extract.py) turns a task's download into per-draw records. `python -m randombench.extract results/<task>` writes each run to `data/<model>/<run_id>.jsonl`, one draw per line, in draw order. Each record holds:
  - model, run id, task version and start time
  - draw index, prompt id and author
  - raw reply, reasoning trace, token counts, costs and latency
  - the error, for failed calls

  No quota. Settles D5 and D13.

  *Found:*
  - The run id and task version exist only in the download path, `<task>/<version>/<model>/<run_id>/`. Inside the files, `taskVersion.versionNumber` is 1 and `pyRunId` is `Run #1` in every run, even in 1.2's versions 2 and 3. Local runs have no unique id at all. So `data/` takes only runs downloaded from Kaggle, and the extractor refuses any other path (D5).
  - A run is read from its parent file when it has one, and otherwise from its per-draw files, skipping the empty ones a crash leaves. The two never need merging: with `%choose`, a finished run keeps only its parent, and a run that dies keeps only per-draw files (1.2). A per-draw file's name gives its draw index (`param_id_37`), because `evaluate()` takes it from the evaluation data's index (`kaggle_benchmarks/tasks.py`, line 305), and `collect.py` numbers rows by draw index. So a draw that raised outside the sub-task's own `try` keeps its index there. Inside a parent file it has none, and it's stored without one.
  - Runs merge without duplicates because each run has one file, named by its run id, and extracting it again rewrites that file. `load_draws` reads every file and still checks that no draw appears twice.
  - The extractor refuses a run shaped unlike the ones seen so far: a draw with more than one request or more than two messages, a prompt text that isn't its prompt id's text in `prompts.py`, a result's reply that differs from the conversation's, unexpected result keys, or an unknown state. A new shape, such as a reasoning model's output in the pilot, then stops extraction instead of being misread. A reasoning trace, if one comes back, is read from the request's `reasoningTraces` (`kaggle_benchmarks/kaggle/serialization.py`, line 374). Counts and costs are converted to integers.
  - Tried on 2.4's real files, copied into the download layout in a scratch folder with made-up run ids. Each run gave byte-identical records whether read from its parent file or from its per-draw files alone. All 24 draws passed every check, and the replies match 2.4's findings. These trial records stay out of `data/`.
  - A record is about 385 bytes, so a 600-draw run is about 230 KB, and six models about 1.4 MB. That's fine to commit (D5). The raw replies alone are about 3 KB per model and 17 KB for all six, small enough to embed in a scoring task (D13).
  - A task's download includes its creation run on the server's default model (1.2), so extracting it would also store `gemini-3.8-flash` draws. Until D12 settles what a creation run does, download with `-m` for each of our models, or leave that run's file out of `data/`.

### Phase 3: scoring, entirely offline

These steps use only stored draws and simulated random numbers. No quota.

- [x] **3.1 Share code between the task and the analysis.** [randombench/sync.py](randombench/sync.py) copies modules into task files. A task file marks where a module goes with `# --- begin randombench/prompts.py ---` and `# --- end randombench/prompts.py ---`, and `python -m randombench.sync` replaces what's between the markers with the module's current text. [tasks/collect.py](tasks/collect.py) now holds a copy of `prompts.py` instead of importing it. Settles D6.

  *Found:*
  - The CLI converts a task file with `jupytext.reads(source, fmt="py:percent")` (`_convert_py_to_notebook` in `kaggle/api/kaggle_api_extended.py`). Besides turning `# %…` comments into magics (1.2), jupytext turns `# !cmd` into a shell command and `#%%` into a cell break. Inside a string, such lines are left alone. A test runs the conversion on every task file and checks that it changes no line except a `# %choose` comment. That covers the copies and the task's own code alike.
  - Run in isolated mode from outside the repo, where `randombench` can't be imported, `collect.py`'s cells up to the last one define both tasks and give the six pinned prompt ids. Before a push, the last cell must run `kbench.llm` instead of two named local models, `%choose` must be added, and D10 and D12 must be settled.
  - The sync refuses a copied module that imports `randombench`, since that import would fail on Kaggle. The scoring tasks will need the parser and the scoring code together, so this comes back once scoring code uses the parser (3.4). Copied modules also share one namespace with the task's own code, so two of them mustn't define the same top-level name.
- [ ] **3.2 Calibration harness.** Score Python's random generator many times at N = 600, and at smaller N for runs that stop early, to see what a perfect generator scores.
- [ ] **3.3 Score mapping.** Decide together how a measured bias becomes a sub-score in [0, 1]. Base it on how big the bias is (effect size), not on p-values. A p-value gets more extreme as N grows, so more draws would make every model look worse. Settles D7.
- [ ] **3.4 Bias checks,** one commit each, in the order of the table above.
- [ ] **3.5 Headline score.** Decide together. Harmonic mean is the starting candidate. Watch for two problems: it drops to 0 if any single sub-score is 0, and overlapping checks penalize the same bias twice. Settles D8.
- [ ] **3.6 Scoring tasks.** One task per sub-score and one for the headline, built from the shared scoring code (D6, D13). Each recomputes its score from the stored draws of the model it's run with, and never prompts that model.

### Phase 4: real runs on Kaggle

- [ ] **4.1 Pilot.** **Quota: 120 calls**, 20 on each of the six models.

  *Learn:* whether all six accept a null seed, how many tokens a draw costs (reasoning models may think at length), and the invalid-answer rates. This is the go/no-go for full runs.
- [ ] **4.2 Full runs.** **Quota: 3,600 calls**, 600 on each of the six models. Download, extract and commit the data.
- [ ] **4.3 Top-ups** for any model short of 600, if D3 chooses to accumulate draws across runs.
- [ ] **4.4 Scores on Kaggle.** Push the scoring tasks with the stored draws, and run each one against the six models. They never prompt a model, so they spend no quota. Still, it's about 9 tasks × 6 models = 54 runs every time the data changes. Add every task to the RandomBench Benchmark on the website. Each task row's menu there can run that task for all the Benchmark's models at once (1.3).

### Phase 5: release

- [ ] **5.1 README.** Covers the method, the results, and the limitations, such as no temperature control and reliance on provider defaults.
- [ ] **5.2 Publish.** A public Benchmark needs public tasks (1.3), so the collection task and every scoring task get published. `kaggle b t publish` can't be undone from the CLI, so each one needs an explicit go-ahead.

## Open decisions

| ID | Question | Leaning | Settled in |
|---|---|---|---|
| D1 | Reasoning: use each model's default, or force it on or off? | **Settled:** the provider default. Not every model can turn reasoning off: gpt-oss-120b offers only low, medium and high. Forcing a setting would treat models unevenly either way. The default also matches how people use each model, and how we already handle temperature. The default returns no reasoning traces (2.3), so only token counts show how much a model reasoned. Getting traces without changing the reasoning level would take provider-specific request options, and OpenAI's chat API never returns them. Watch the cost in the pilot. | 2.3 |
| D2 | Leaderboard: one task that returns a dict of sub-scores plus the headline, or one task per sub-score? | **Settled:** one scoring task per score, all reading the same stored draws. See "Leaderboard" under Decided. | 1.2 |
| D3 | Runs that stop early: score each run on what it collected and merge all draws offline? Or have later runs top up to 600 from stored draws, through an attached Kaggle dataset? | Start with self-contained runs, and revisit after seeing real failures. A run with no parent file is extracted from its per-draw files, skipping the empty ones, so a run that dies keeps the draws it finished (1.2, 2.5). | 4.1 |
| D4 | How strict is parsing? | **Settled:** lenient on formatting, strict on content. Only digits count, and they must be one integer in 0–99. A restated range, number words and `47.0` make a reply invalid. See "Parsing" under Decided. | 2.2 |
| D5 | Data layout: one committed file per run under `data/`? How large do reasoning traces make it? | **Settled:** one JSON Lines file per run, `data/<model>/<run_id>.jsonl`, one draw per line, committed. Only runs downloaded from Kaggle, because only their download path holds a run id and a task version (2.5). A 600-draw run is about 230 KB. The default reasoning returns no traces (2.3), so traces add nothing unless D1 changes. | 2.5 |
| D6 | A push uploads a single `.py` file, but the offline analysis needs the same parsing and scoring code. Options: a source module plus a script that copies it into the pushed task file; the task file as the only copy; or code in an attached Kaggle dataset. | **Settled:** a copy in the task file. Each task file holds the `randombench` modules it uses between `# --- begin <path> ---` and `# --- end <path> ---` markers, and `python -m randombench.sync` refreshes the copies, so the file run locally is exactly the file pushed. Tests fail when a copy is out of date, when a task imports `randombench`, or when converting a task to a notebook would change a line (3.1). A copied module can't import `randombench` itself. | 3.1 |
| D7 | How does a measured bias become a sub-score in [0, 1]? | Based on effect size | 3.3 |
| D8 | How is the headline score computed? | Harmonic mean, still to be decided | 3.5 |
| D9 | Prompt assignment: pick a prompt independently at random for each draw, or shuffle a balanced schedule with each prompt used exactly 100 times? | **Settled:** a balanced shuffle. The order is still random, and equal counts make prompts easier to compare. `balanced_schedule` in [tasks/collect.py](tasks/collect.py) does it (2.4). | 2.4 |
| D10 | What is the collection task's name? Its slug must stay stable once pushed, because a new slug makes a new task instead of a new version. | — | Before the real task's first push |
| D11 | Should the draw sub-task catch its own errors and return them in its result, rather than letting the library mark the draw `ERRORED`? An errored sub-run keeps no result, so its prompt id would be lost, and the library won't retry inside a parent task anyway (1.1). | **Settled:** catch and return. Each draw's result always holds its draw index and prompt id, plus either the reply or the error (2.4). | 2.4 |
| D12 | Every push runs the whole task once against the server's default model, `gemini-3.8-flash` (1.2). For the real task, that's 600 calls per push on a model outside our six, and they count against quota. Should the creation run make only a few draws? If so, how does the task recognize it? A scoring task's creation run has a related problem: there are no stored draws for that model, but the run must still complete, or the task can't be run. | Make it cheap. The model is the only signal: the creation run gets the server's default, and the environment marks nothing else (1.3). The task could make a few draws for any model outside our six. A scoring task returns `False` for a model with no stored draws. That lets its creation run complete, and the resulting FAIL doesn't count in an average of task scores (1.3). Until this is settled, count 600 creation calls in the quota of every push. | Before the real task's first push |
| D13 | How does a scoring task get the stored draws? A push uploads one file, so the options are an attached Kaggle dataset (`push -d`), or draws embedded in the task file by the script that builds it (D6). | **Settled:** embedded. A build script copies the draws from `data/` into each scoring task file (D6). It needs no new infrastructure, and each task version holds exactly the draws it scored. 1.3 showed it works, and the task page shows only the task function, not the draws. The raw replies for six models come to about 17 KB (2.5). Which fields to embed is decided in 3.6. | 2.5 |
| D14 | What does the collection task itself return? Its Result is a leaderboard cell too. If it returned the headline from its own run's draws, that could disagree with the scoring tasks once they score draws merged from several runs (D3). | The number of draws it collected, so every score comes from a scoring task | 3.6 |
| D15 | The Benchmark shows its own average for each model and ranks by it (1.3). Public Benchmarks label it "Avg N tasks". One averages only some of its tasks, and another shows a 95% interval beside it. How is it computed, and how does it fit our headline (D8)? A plain mean over all our tasks would mix the headline with the sub-scores it's built from. It would also count the creation runs' FAIL against `gemini-3.8-flash`, unless that model can be left out. Can our tasks' intervals appear? | Settings offer six kinds of overall score (1.3). Use "weighted average of task scores, with confidence interval", weighting only the headline task. That needs two things: a weight can be 0, and the interval comes from the task's own `(value, CI)`. The ranking is then our headline with its interval, and the sub-scores show as columns beside it. Otherwise use "no overall score", with the headline as one column among the others. The private page didn't show the interval (1.3), so check it once RandomBench is public. The setting is a dropdown that can be changed at any time. Weighting the sub-score tasks instead would hand D8 to Kaggle, and its interval would treat sub-scores computed from the same 600 draws as independent. | Before 4.4 |
| D16 | What temperature do the models actually sample at? The library sends none, and it's unknown whether Model Proxy then uses the provider's default or a lower one of its own. Gemini 3.1 Flash-Lite answered 42 on all 12 draws (2.4). If the effective temperature were 0, RandomBench would measure each model's single likeliest answer, not its sampled distribution. The library can send a temperature: building the model with `ModelProxy(..., support_temperature=True)` overrides its `setdefault` (`model_proxy.py`, line 102). It's switched off because of an unresolved bug (b/439876083), whose nature isn't stated. | **Settled:** keep the provider's default. With `temperature=1.0` sent explicitly, all 12 Gemini draws were 42 again (2.4b), so the 42s come from the model, not from a low default. One caveat: if Model Proxy silently ignored the parameter, the probe would look exactly the same. Sending 2.0 would test that, but it isn't needed to keep the default. | 2.4b |
