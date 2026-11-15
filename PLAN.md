# RandomBench plan

This is how RandomBench gets built, one small step at a time. Each step is one commit and comes with something to learn. Steps marked **quota** make LLM calls through Kaggle's Model Proxy. They wait for an explicit go-ahead, and each one states its call count up front.

## Goal

Measure how well LLMs generate random integers in [0, 99]. Each model answers hundreds of independent single-turn prompts, and its answers are checked for known biases: favoring primes, round numbers, and so on. Each check yields a sub-score in [0, 1]. The headline is the share of valid replies times the uniformity score (D8), and the other sub-scores show which biases lie behind it.

## Decided

- **Range:** integers 0–99.
- **Names:** the Kaggle Benchmark is called RandomBench. A Benchmark is the page that groups tasks and shows the leaderboard. The collection task is RandomBench Draws, slug `randombench-draws`, and each scoring task follows that pattern (D10). The ten scoring tasks are RandomBench Score for the headline, then RandomBench Valid Answers, Uniformity, Primes, Multiples of 10, Multiples of 5, Favorite Number, Edges, Repeated Digits and Last Digit, with slugs from `randombench-score` to `randombench-last-digit`. Valid Answers is the invalid-answer sub-score, named for the direction of its score: 1.00 means no reply was invalid. Benchmark and task slugs are both scoped to their owner (`/benchmarks/{owner}/{slug}` and `/benchmarks/tasks/{owner}/{slug}`), so nobody else can take ours and there's nothing to reserve early. Benchmarks are created on the Kaggle website; the CLI handles only tasks.
- **Calls:** single-turn, in a fresh conversation each time. No tools, and the model never sees its earlier answers.
- **Replies:** plain text for every model, read by one parser of our own (2.2). No structured output (`schema=`). kaggle-benchmarks 0.6.2 sends a schema natively to most models, but gemma, qwen, deepseek and meta models get it only as text in a system message (`kaggle_benchmarks/kaggle/model_proxy.py`, lines 73–79). The models would then answer under different conditions, and their invalid-answer rates couldn't be compared. The prompt states the range, and an answer outside it counts as invalid.
- **Prompts:** six, in [randombench/prompts.py](randombench/prompts.py), each with a name, an author and its text. D9 decides which prompt each draw gets. The checks pool a model's answers across all six prompts (D17).
  - **Ids:** a prompt's id is its name plus the first 8 hex digits of its text's SHA-256, such as `plain-47858739`. Editing the text gives a new id by itself, and a test pins every id, so no edit goes unnoticed, not even to whitespace. Once a prompt has been used in a run, it's never edited or deleted, and a change means adding a new prompt. A way to retire a prompt can wait until one first needs replacing.
  - **Authors:** Claude drafted all six (`claude-opus-5-5`). A prompt the user rewords lists `benjaminbrumfield` instead.
  - **Style:** every prompt asks for a random integer, gives the range as 0 to 99 with both ends included, and asks for the number only. A model that wasn't asked to be random can't be faulted for having a favorite. No prompt gives an example number or names a bias, since either could steer the answer. Beyond that, the prompts differ in wording, tone, length and framing, so no score depends on a single phrasing. A test checks the rules code can check.
- **Volume:** 600 draws per model per run. That gives a 95% margin of error of at most ±4 percentage points on any share a check measures, and 6 expected answers per value, above the usual minimum of 5 for a chi-squared test. Halving the margin would take 4× the draws. 600 also splits evenly across six prompts (D9). If the pilot shows spare quota, 1,200 is the next step up: ±2.8 points for twice the calls. The quota turned out to be $10 a day, and 600 draws on all six models cost $0.13 (4.3), so quota doesn't limit this. A second 600-draw run per model, for 1,224 in all, isn't needed (4.5a). Nano and gpt-oss-120b tie at 0.0327 ±0.0034 and 0.0325 ±0.0016, and a second run would narrow those intervals by only about √2. Every other pair of models is already apart. Failed calls are recorded, not dropped, and a run that stops early keeps what it collected.
- **Raw data:** every raw response is stored. Parsed values and scores are always recomputed from the raw text, so a new bias check or a new parsing rule never requires calling the models again. [randombench/extract.py](randombench/extract.py) stores each Kaggle run as one JSON Lines file, `data/<model>/<run_id>.jsonl` (D5).
- **Parsing:** [randombench/parsing.py](randombench/parsing.py) reads a reply leniently on formatting and strictly on content (D4). Markdown, LaTeX, punctuation and words are ignored. The numbers written in digits must all be the same integer, and it must be in 0–99. So `Sure! **47**.` is 47, and `47 or 52`, `3.5`, `47.0`, `100` and `forty-seven` are invalid. A reply that restates the range, as in `from 0 to 99: 47`, holds three integers and is invalid too, because the prompt asked for the number only. If a provider puts its reasoning into the reply text, every number in the reasoning would count. 2.3 shows where reasoning ends up.
- **Invalid answers:** the rate of invalid answers is its own sub-score. A failed call, which returns no reply, doesn't count as invalid. With fewer than 2 valid answers, the other checks give no score (3.4).
- **Shared code:** a push uploads only the task file, so each task file holds a copy of the `randombench` modules it uses, refreshed by `python -m randombench.sync` (D6).
- **Leaderboard:** every sub-score and the headline appear on Kaggle, and no model is ever called twice for the same draw (D2, settled in 1.2). A notebook puts only one task on the leaderboard, and a dict result shows a blank Result. So the collection task makes each model's draws, and each score gets a scoring task of its own. A scoring task never prompts a model: run with `-m X`, it recomputes its score from X's stored draws and returns it. 1.2 showed that a task can be run against a model without prompting it. The RandomBench Benchmark groups all the tasks.
- **Temperature:** the provider's default. This was not our choice: kaggle-benchmarks 0.6.2 never sends temperature through Model Proxy (`kaggle_benchmarks/kaggle/model_proxy.py`, line 101). With temperature 1.0 sent explicitly, Gemini 3.1 Flash-Lite answered exactly as it does with none sent: 42 every time (2.4b, D16).
- **Seed:** `seed=None`, so providers sample freshly, as in normal API use. The library's default seed is 0, and it is sent to every non-Google model. A backend that honors seeds could then return the same number every time, which would look like a bias. GPT-5.4 nano accepted a null seed (2.3). The library never sends a seed to a `google/` model, so the two Gemini models and Gemma sample freshly regardless. Claude Haiku 4.5 and gpt-oss-120b accepted a null seed in the pilot too, and both gave more than one answer (4.2).
- **Models:** six, with their exact ids. Kaggle slugs come from `kaggle b t models`. The `llm.name` of each comes from the server's `LLMS_AVAILABLE` (1.3), and `MODELS` in [tasks/collect.py](tasks/collect.py) lists them (D12). The pilot confirmed all six as runs' `llm.name` (4.2). Local runs reach the models in `LLMS_AVAILABLE` in `.env` (both checked 2026-11-14). Local development uses the first two.

| Model | Kaggle slug (`-m`) | `llm.name` | Local |
|---|---|---|---|
| GPT-5.4 nano | `gpt-5.4-nano-2026-03-17` | `openai/gpt-5.4-nano-2026-03-17` | yes |
| Gemini 3.1 Flash-Lite (preview) | `gemini-3.1-flash-lite-preview` | `google/gemini-3.1-flash-lite-preview` | yes |
| Gemini 3.5 Flash-Lite | `gemini-3.5-flash-lite` | `google/gemini-3.5-flash-lite` | — |
| Claude Haiku 4.5 | `claude-haiku-4-5-20251001` | `anthropic/claude-haiku-4-5@20251001` | — |
| Gemma 4 26B A4B | `gemma-4-26b-a4b-it` | `google/gemma-4-26b-a4b` | — |
| gpt-oss-120b | `gpt-oss-120b` | `openai/gpt-oss-120b` | yes |

## Bias checks

A perfect generator gives each value probability 1/100. Every check but the first scores only the valid answers, in [randombench/checks.py](randombench/checks.py). Each sub-score is in [0, 1]: about 1 for a perfect generator, and lower the more biased the model (D7). A gap is the share of answers that would have to change.

| Check | Sub-score | Ideal | Lowest | Perfect generator at N = 600 (mean, 95% range) |
|---|---|---|---|---|
| Invalid answers | 1 − the share of replies that aren't a single integer in [0, 99] (3.4) | 0% | 0 | 1 |
| Uniformity | The effective number of values K, as K/100. A fair pick among K values repeats itself as often as the model does | K = 100 | 0.01, one value | 0.991 (0.952–1) |
| Primes | 1 − the gap between the share of primes (2, 3, 5, …, 97: 25 values) and 25% | 25% | 0.25, all prime | 0.994 (0.965–1) |
| Round numbers | 1 − the gap from 10% multiples of 10, and separately from 20% multiples of 5 | 10% (20%) | 0.10 (0.20) | 0.996 (0.976–1), 0.994 (0.967–1) |
| Favorite number | 1 − the favorite's share above 1%, scaled so that 100% scores 0 | 1% | 0, one value | 0.996 (0.983–1) |
| Edges | 1 − the gap from 20% of answers in 0–9 or 90–99 | 20% | 0.20 | 0.994 (0.968–1) |
| Repeated digits | 1 − the gap from 9% of answers in 11, 22, …, 99 | 9% | 0.09 | 0.996 (0.977–1) |
| Last digit | The effective number of last digits K, as K/10 (7 is the usual suspect) | K = 10 | 0.10, one digit | 0.997 (0.983–1) |

Each sub-score estimates the model's true score with sampling noise removed on average, so a model's expected score doesn't depend on its number of valid answers (3.3). [randombench/calibration.py](randombench/calibration.py) simulates any distribution at any N, and the tests use it to check that. Each score also has a 95% interval, from a bootstrap of the model's replies (D19).

The headline is the invalid-answer score times the uniformity score (D8). It's `None` whenever uniformity is, below 2 valid answers.

Some checks overlap. For example, 7, 17, 37, 47, 67 and 97 are all primes ending in 7. The headline uses only uniformity and validity, so an overlapping bias isn't counted twice. Uniformity already reflects every bias in the distribution, and the other checks show which ones lie behind it.

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
  - Our private Benchmark's page puts tasks in rows and models in columns. Each cell holds one task's result for one model, and a dict result's cell is blank. Each row is pinned to a task version (`v3`, `v2`) and shows only that version's runs. Nano's `randombench-fake-draws` run was on v2, so its v3 cell offers a ▷ run button instead. A new version of a scoring task therefore needs a fresh run on every model, as 4.6 counts.
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
  - A task row's `⋮` menu on the Benchmark page runs that task for all its models, or only for those missing results (4.6).
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
  - A draw cost about $0.00001 on these two models, so 600 draws cost under a cent per model. Reasoning models may cost much more (4.2).
  - From the library source: the HTTP client can cache responses, keyed by the request body (`build_httpx_client` in `kaggle_benchmarks/utils.py`). It's off unless `ENABLE_LOCAL_CACHING` is set, and our `.env` doesn't set it. If it were on, every draw with the same prompt would get the first draw's answer back. The collection task must keep it off, and a Kaggle run should confirm that the server doesn't turn it on. It's separate from `kbench.client.enable_cache()`, which only skips a run whose run file already completed. The server doesn't set it: 1.3's `-m gpt-5.4-nano-2026-03-17` run printed all 50 of its environment variables, and `ENABLE_LOCAL_CACHING` isn't one of them (checked 2026-11-14).
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

These steps use only stored draws and simulated random numbers. No quota. The scoring tasks moved to 4.5, after the full runs, because they embed real draws (D13) and the pilot could still change the parser or the model list.

- [x] **3.1 Share code between the task and the analysis.** [randombench/sync.py](randombench/sync.py) copies modules into task files. A task file marks where a module goes with `# --- begin randombench/prompts.py ---` and `# --- end randombench/prompts.py ---`, and `python -m randombench.sync` replaces what's between the markers with the module's current text. [tasks/collect.py](tasks/collect.py) now holds a copy of `prompts.py` instead of importing it. Settles D6.

  *Found:*
  - The CLI converts a task file with `jupytext.reads(source, fmt="py:percent")` (`_convert_py_to_notebook` in `kaggle/api/kaggle_api_extended.py`). Besides turning `# %…` comments into magics (1.2), jupytext turns `# !cmd` into a shell command and `#%%` into a cell break. Inside a string, such lines are left alone. A test runs the conversion on every task file and checks that it changes no line except a `# %choose` comment. That covers the copies and the task's own code alike.
  - Run in isolated mode from outside the repo, where `randombench` can't be imported, `collect.py`'s cells up to the last one define both tasks and give the six pinned prompt ids. Before a push, the last cell must run `kbench.llm` instead of two named local models, `%choose` must be added, and D10 and D12 must be settled.
  - The sync refuses a copied module that imports `randombench`, since that import would fail on Kaggle. The scoring tasks will need the parser and the scoring code together, so this comes back once scoring code uses the parser (3.4). Copied modules also share one namespace with the task's own code, so two of them mustn't define the same top-level name.
- [x] **3.2 Calibration harness.** Measure a perfect generator many times at N = 600, and at smaller N for runs that stop early, to see what it scores. No check existed yet to measure with, so this step also wrote each check's raw measurement, in [randombench/checks.py](randombench/checks.py). Each takes the 100 counts of a model's valid answers, or a batch of them. The distance checks use the total variation distance. [randombench/calibration.py](randombench/calibration.py) simulates 10,000 runs at any N (seeded, so it's reproducible), and `python -m randombench.calibration` prints the table below. A run's counts are drawn from a multinomial distribution, which is exactly how the counts of N independent uniform draws are distributed.

  *Found:*
  - Mean and 95% range over 10,000 runs of a perfect generator:

    | Measurement | N = 100 | N = 300 | N = 500 | N = 600 | N = 1200 |
    |---|---|---|---|---|---|
    | Uniformity | 0.366 (0.300–0.430) | 0.223 (0.190–0.257) | 0.175 (0.148–0.202) | 0.160 (0.137–0.185) | 0.114 (0.098–0.132) |
    | Primes | 0.250 (0.170–0.340) | 0.250 (0.200–0.300) | 0.250 (0.212–0.288) | 0.250 (0.215–0.285) | 0.250 (0.225–0.274) |
    | Multiples of 10 | 0.100 (0.050–0.160) | 0.100 (0.067–0.137) | 0.100 (0.074–0.128) | 0.100 (0.077–0.123) | 0.100 (0.083–0.117) |
    | Multiples of 5 | 0.199 (0.120–0.280) | 0.200 (0.157–0.247) | 0.200 (0.166–0.236) | 0.200 (0.168–0.232) | 0.200 (0.177–0.223) |
    | Favorite | 0.042 (0.030–0.060) | 0.027 (0.023–0.037) | 0.023 (0.018–0.028) | 0.022 (0.018–0.027) | 0.018 (0.016–0.022) |
    | Edges | 0.200 (0.130–0.280) | 0.200 (0.157–0.247) | 0.200 (0.166–0.236) | 0.200 (0.168–0.232) | 0.200 (0.178–0.223) |
    | Repeated digits | 0.090 (0.040–0.150) | 0.090 (0.060–0.123) | 0.090 (0.066–0.116) | 0.090 (0.068–0.113) | 0.090 (0.074–0.107) |
    | Last digit | 0.119 (0.060–0.180) | 0.069 (0.037–0.107) | 0.053 (0.030–0.082) | 0.049 (0.027–0.075) | 0.035 (0.019–0.053) |

  - The measurements come in two kinds. A share is unbiased: its mean is the ideal at every N, and only its spread shrinks. At N = 600 the prime share's range is ±3.5 points, in line with the ±4-point margin under "Volume". The two distances and the favorite's share have a floor instead: their means sit above the ideal and fall roughly as 1/√N, from 0.366 at N = 100 to 0.114 at N = 1200 for uniformity. So a raw distance can't be compared across models with different numbers of answers (D7).
  - N is the number of valid answers, not the number of draws. A model with 100 invalid replies is measured at N = 500, where a perfect generator's uniformity is already 0.175 rather than 0.160. Calibration therefore has to happen at each model's own N (D7).
  - Turning a share into a gap from its ideal brings back a floor too, since |share − ideal| is never negative. Every sub-score therefore needs calibration, not only the three marked "Calibrated" (D7).
  - Calibrating at five sizes took 0.8 s, so a scoring task could calibrate at its model's N on Kaggle instead of embedding a table (D13). `calibration.py` imports `checks.py`, though, so the sync would refuse to copy it, the same limit 3.1 found for scoring code that uses the parser.
  - The tests check the harness against exact values that need no simulation. A share's expected value is its set's size over 100. A distance's expected value is the sum of each value's expected gap, by linearity, because one value's count is Binomial(N, 1/100) even though the counts depend on each other. The simulated means match both to within 0.001.
- [x] **3.3 Score mapping.** Decide together how a measured bias becomes a sub-score in [0, 1]. Base it on how big the bias is (effect size), not on p-values. A p-value gets more extreme as N grows, so more draws would make every model look worse. Settles D7.

  [randombench/checks.py](randombench/checks.py) now gives each check's sub-score, as the bias table describes. [randombench/calibration.py](randombench/calibration.py) simulates any distribution, not only a perfect one. One decision fixed all eight distribution checks, so they were built together here, not one at a time in 3.4.

  *Found:*
  - Subtracting a perfect generator's floor, the obvious use of 3.2's calibration, fails. A value the model never picks adds exactly 0.01 to the TVD, with no noise, so a biased model's TVD has almost no floor to subtract. Raw TVD fails the other way, and it has no unbiased estimate, so no correction fixes it. Average uniformity score over 4,000 simulated runs of made-up models, at N = 100 / 600 / 1200:

    | Model | 1 − TVD (true) | 1 − TVD | 1 − (TVD − perfect floor) | Effective values / 100 (true) | Effective values / 100 |
    |---|---|---|---|---|---|
    | Perfect | 1.000 | 0.63 / 0.84 / 0.89 | 0.99 / 1.00 / 1.00 | 1.000 | 0.95 / 0.99 / 1.00 |
    | Prime-leaning | 0.775 | 0.59 / 0.74 / 0.76 | 0.96 / 0.90 / 0.88 | 0.787 | 0.80 / 0.79 / 0.79 |
    | LLM-like | 0.330 | 0.30 / 0.33 / 0.33 | 0.67 / 0.49 / 0.44 | 0.107 | 0.11 / 0.11 / 0.11 |
    | One value | 0.010 | 0.01 / 0.01 / 0.01 | 0.38 / 0.17 / 0.12 | 0.010 | 0.01 / 0.01 / 0.01 |

  - The fix is to score quantities whose noise cancels on average, so that a model's expected score is its true score at any N, and nothing needs subtracting. The chance that two answers are equal has an exactly unbiased estimate, the share of pairs of answers that are equal. Its inverse is the effective number of values. A share's squared gap has one too: subtract the share's estimated sampling variance, s(1 − s)/(N − 1). On a linear scale the shares' raw floors were small (a perfect generator's raw primes score was 0.986 at N = 600), but they're corrected all the same.
  - The plain top share is biased upward, because whichever value leads by chance becomes the favorite. Leaving one answer out fixes that: count how often an answer equals the favorite of all the other answers, with ties counted fractionally. This estimates, without bias, the chance that a new answer equals the current favorite. For a perfect generator it averaged 1.00% at N = 20, 100 and 600, where the plain top share is 2.2% at N = 600. A test checks it against a brute-force version.
  - A test checks the central claim: for seven made-up models, from a perfect generator to one value, at N = 300, 600 and 1200, every check's average score is within 0.02 of the model's true score. A perfect generator averages 0.991–0.997 at N = 600 and 0.952–0.994 at N = 100, lowest on uniformity, whose estimate is the curved 1/rate and capped at 1.
  - Scoring no longer needs calibration, so a scoring task needs only `checks.py` and the parser, and neither imports `randombench`. That settles 3.2's worry about copying `calibration.py`. Code that parses replies and counts them would import both, though (3.4).
  - The sub-scores bottom out at different levels: 0.01 for uniformity, 0.25 for primes, 0 for the favorite (D8).
  - On 2.4's 12 draws, nano behaves like a fair pick among 5.1 values (uniformity 0.051), and Gemini 3.1 Flash-Lite like a pick among one (0.01).
- [x] **3.4 Invalid answers and scoring stored draws.** Score the invalid-answer rate, and turn a model's stored replies into all its sub-scores: parse each reply (2.2), count the valid answers, and run every check (3.3). Decide whether a failed call counts as an invalid answer, and what a model with fewer than 2 valid answers scores, since the checks need at least 2.

  `score_answers` in [randombench/checks.py](randombench/checks.py) takes a model's parsed replies, an int for each valid one and `None` for each invalid one, and returns all nine sub-scores. The caller parses, so `checks.py` still needs no parser and can be copied into a task file on its own (D6). A test checks that `prompts.py`, `parsing.py` and `checks.py` can each be copied. `python -m randombench.score` ([randombench/score.py](randombench/score.py)) loads `data/`, merges each model's runs (D3), and prints each model's counts and sub-scores. It runs offline only.

  *Found:*
  - A failed call doesn't count as an invalid answer. It's the provider's failure, and no answer came back to judge. The invalid-answer score is invalid replies out of replies, and the report shows failed calls in a column of their own.
  - Below 2 valid answers, the eight distribution checks give `None`, shown as `—`. A scoring task can return `False` for it, which shows as FAIL and is left out of the leaderboard's average (1.3). The invalid-answer score still shows what went wrong. Whether a higher minimum is needed waits for real invalid rates in the pilot (4.2).
  - On 2.4's 12 real replies per model, from 2.5's trial records, Gemini 3.1 Flash-Lite scores 0.010 on uniformity, 0.000 on the favorite and 0.100 on the last digit. Nano scores 0.051, 0.589 and 0.314. Both score 0.80 on edges only because neither gave an answer in 0–9 or 90–99 in 12 draws. Twelve draws show the method works, not how biased the models are.
- [x] **3.5 Headline score.** Decide together. Harmonic mean is the starting candidate. Watch for two problems: it drops to 0 if any single sub-score is 0, and overlapping checks penalize the same bias twice. Settles D8 and D17.

  `score_answers` in [randombench/checks.py](randombench/checks.py) now gives the headline first: the invalid-answer score times the uniformity score. `python -m randombench.score` shows it as the first score column.

  *Found:*
  - Any average of the nine sub-scores fails, harmonic included. The share checks are lenient by design (their lowest scores are 0.25 and up), so they swamp the rest. Invalid answers are only one sub-score in nine, so they barely count. Average over 200 simulated runs of 600 draws, 100 per prompt:

    | Model | Valid × uniformity | Mean of 9 | Geometric mean of 9 | Harmonic mean of 9 |
    |---|---|---|---|---|
    | Perfect | 0.992 | 0.996 | 0.996 | 0.996 |
    | Perfect, 30% invalid | 0.692 | 0.961 | 0.956 | 0.949 |
    | LLM-like (a fair pick among 11) | 0.107 | 0.732 | 0.625 | 0.445 |
    | One value | 0.010 | 0.586 | 0 | 0 |

    The harmonic mean's drop to 0 comes only from the favorite, which scores 0 for one value.
  - Valid share × uniformity reads plainly: "a fair pick among K values, scaled down by the share of replies that are valid". It drops to 0 only when no reply is valid. It's `None` below 2 valid answers, like uniformity, so a model that almost never answers validly would show FAIL on Kaggle, not 0.
  - Pooling answers across prompts flatters a model whose answer depends on the wording. If each prompt gets its own constant answer, the pooled uniformity is 0.061 ("a fair pick among 6"), against 0.010 measured within each prompt. If each prompt has its own 50% favorite, it's 0.206 against 0.039. The checks stay pooled anyway (D17), and the README must say so (5.1).
  - With the headline as its own task, D15's leaning still works: weight only the headline task, and the Benchmark ranks models by it.
  - On 2.4's 12 draws per model, the headline is 0.010 for Gemini 3.1 Flash-Lite and 0.051 for nano, the same as uniformity, because every reply was valid.

### Phase 4: real runs on Kaggle

- [x] **4.1 Ready the collection task for Kaggle.** Make [tasks/collect.py](tasks/collect.py) pushable, and give it what the pilot needs. Writing it spends no quota. A local test run, if any, waits for a go-ahead. Settles D10, D12 and D14.
  - Name the task (D10), and stop its creation run from prompting a model it shouldn't (D12).
  - Replace the last cell's two named local models with `collect.run(kbench.llm)`, and add `# %choose <slug>` (1.2).
  - Record what the pilot has to learn: the server's `kaggle_benchmarks` version, the value of `KAGGLE_AGENT_LLM_REASONING_EFFORT`, and the number of HTTP requests each draw makes (2.4b's request hook).
  - Set the pilot's size. The balanced schedule needs a multiple of 6 draws per model and raises an error otherwise, so 4.2's original 20 can't work.

  *Found:*
  - [tasks/collect.py](tasks/collect.py) is named RandomBench Draws (D10). It makes draws only for the six models in `MODELS` (D12) and returns how many got a reply (D14). Its last cells run `collect.run(kbench.llm)` and `# %choose collect`.
  - [tests/test_collect.py](tests/test_collect.py) runs the task file's own code with stand-in models that answer offline. A stand-in named `google/gemini-3.8-flash`, the creation run's model, is never prompted and returns 0. One named like nano gets every draw, and `extract.py` reads the run files it leaves, checking every prompt text against `prompts.py`. So the task's output and the extractor are known to agree before any server run.
  - Every run prints the server's `kaggle_benchmarks` version and the value of `KAGGLE_AGENT_LLM_REASONING_EFFORT`, the creation run included. So the pilot's push answers both before any model is prompted, in `kaggle b t log randombench-draws`.
  - A listed model's run counts its HTTP requests with an httpx hook on the model's client and prints the count against its draws, so retries show. The hook reaches a private attribute (`llm.client._client`). A test builds a real `ModelProxy`, which sends nothing, and finds the hook list there. If the server's library differs, the run prints that it couldn't count, and still runs. A test also sends 20 requests from 4 threads through a mock transport and counts all 20.
  - The pilot draws 24 per model, 4 per prompt: 144 calls in all. The full runs need `N_DRAWS = 600`, so 4.3 pushes a new version.
- [x] **4.2 Pilot.** **Quota: 144 calls**, 24 on each of the six models, plus any retries. First push `tasks/collect.py` as `randombench-draws`. Its creation run prompts no model (D12), and its log shows the server's library version and reasoning-effort variable. If those look wrong, stop before any `-m` run. Then run it with `-m` for each of the six models.

  *Learn:* whether all six accept a null seed, how many tokens a draw costs (reasoning models may think at length), and the invalid-answer rates. This is the go/no-go for full runs. Also check what every run prints in its log (4.1):
  - **The server's `kaggle_benchmarks` version.** Local runs use 0.6.2, and the run-file shapes that `extract.py` expects come from it.
  - **The value of `KAGGLE_AGENT_LLM_REASONING_EFFORT`.** The server sets it, but 0.6.2 never reads it: `kbench.llm` is built from `LLM_DEFAULT` with no reasoning setting (`kaggle_benchmarks/kaggle/models.py`). If the server's version does read it, server runs wouldn't use the provider's default reasoning (D1).
  - **Retries.** Each run prints its HTTP requests against its draws.
  - **Each model's `llm.name`.** A run whose model isn't in `MODELS` prints that it made no draws, so an id that's wrong for one of the five unconfirmed models shows at no cost.

  The kaggle-cli skill also contradicts itself on local reach. It says the Model Proxy key "works for all models, not just those listed" in `LLMS_AVAILABLE`, and elsewhere that local runs are limited to that list. One local call to Claude Haiku 4.5 would tell. If the key works, the pilot could run locally, with no creation run (D12).

  *Found:*
  - The push's creation run on `gemini-3.8-flash` prompted nothing and returned 0 (D12). Then all six `-m` runs completed: 144 calls, 24 per model. Every draw got a reply, and all 144 replies are bare numbers that parse, so no call failed and no answer was invalid. Each run counted exactly 24 HTTP requests for its 24 draws, so nothing was retried.
  - The server runs `kaggle_benchmarks` 0.6.3, not 0.6.2. The diff between them, read before any model was prompted, touches nothing RandomBench relies on: one more OpenAI model that never gets a seed (not one of ours), two fixes for cached run files (our run cache stays off), and a Docker change. `extract.py` read all six run files unchanged. Local runs stay on 0.6.2, the version the vendored skills describe.
  - `KAGGLE_AGENT_LLM_REASONING_EFFORT` is set but empty (`''`) in every run, the creation run included. So no library version has a reasoning level to apply, and the models reason at their provider's default (D1).
  - Every `llm.name` in `MODELS` is right. No run printed that it made no draws, and each record's model matches the model table.
  - Claude Haiku 4.5 and gpt-oss-120b accepted `seed=None`, as nano did in 2.3, and both gave more than one answer.
  - Gemma 4 and gpt-oss-120b reason by default. For a two-digit reply they used 80–273 and 37–185 output tokens, against 2–5 for the other four models. No trace came back, and no reasoning reached the reply text (D1). gpt-oss-120b also counts 61 more input tokens than nano on every prompt, so its provider adds text of its own before ours, likely its chat template's system message.
  - The 144 draws cost $0.0053 in all. At that rate, 600 draws cost $0.006–$0.05 per model and about $0.13 for all six. Gemma is the slowest, with a median of 3.1 seconds a draw, so 600 draws take it about 8 minutes at 4 workers.
  - The answers, out of 24: both Gemini models answered 42 every time. Haiku gave 42 nineteen times and 47 five times. Nano gave 42 fourteen times, then 37, 57, 47, 73 and 63. Gemma split between 73 (12) and 42 (11), plus one 67. gpt-oss-120b spread the most: 42 and 57 eight times each, 73 six times, and 71 and 37 once each. The headlines run from 0.010 for both Gemini models to 0.039 for gpt-oss-120b, against 0.99 for a perfect generator. Twenty-four draws rank the models only roughly.
  - gpt-oss-120b scores 1.000 on the favorite check, though a third of its answers are 42. Its 42s and 57s tie at 8, so leaving out any one of them makes the other value the favorite, and no answer equals the favorite of the rest. The leave-one-out estimate is unbiased on average (3.3), but one run with two near-tied favorites can swing it a long way. Only counts within one of each other do this. With two equal favorites at 600 draws, that happens about 6% of the time. The headline doesn't use this check (D8), so it stays as it is for now.
  - The local Haiku call was dropped. `data/` takes only runs downloaded from Kaggle (D5), and the server facts need a server run, so a local pilot couldn't have replaced this one. Which half of the skill is right doesn't matter to RandomBench.
  - With no invalid answers in 144, the minimum of 2 valid answers (3.4) needs no change for now.
  - The six runs are in `data/`, about 9 KB each. They were made under the same conditions the full runs will have, so they count. After 4.3, each model has 624 draws, 104 per prompt (D3).
- [x] **4.3 Full runs.** **Quota: 3,600 calls**, 600 on each of the six models. Set `N_DRAWS = 600` and push version 2 (4.1). Download, extract and commit the data. Merged with the pilot's draws, each model then has 624.

  *Found:*
  - Version 2's creation run prompted nothing, like version 1's. All six runs started together and completed: 3,600 calls, 600 per model, taking from 1 minute (Haiku) to 8 minutes (Gemma). Each run sent exactly 600 HTTP requests for its 600 draws, so nothing was retried. Every draw got a reply, and every reply was a bare number in 0–99, so no call failed and no answer was invalid. The server facts matched the pilot's: `kaggle_benchmarks` 0.6.3 and an empty reasoning-effort variable.
  - The runs cost $0.13 in all, from $0.006 for Gemini 3.1 Flash-Lite to $0.05 for Gemma. Each run's file in `data/` is 228–237 KB, as 2.5 predicted, and `data/` holds 1.5 MB with the pilot's files. Extracting both versions again left the pilot's files byte for byte the same.
  - Merged with the pilot, each model has 624 draws:

    | Model | Distinct values | Top answers | Effective values | Headline |
    |---|---|---|---|---|
    | GPT-5.4 nano | 10 | 42 (51%), then 37, 57, 47, 73 | 3.3 | 0.033 |
    | gpt-oss-120b | 14 | 57 (38%), 42 (35%), 73 (21%) | 3.2 | 0.032 |
    | Gemma 4 26B A4B | 11 | 42 (59%), 73 (35%) | 2.1 | 0.021 |
    | Claude Haiku 4.5 | 2 | 42 (89%), 47 | 1.3 | 0.013 |
    | Gemini 3.5 Flash-Lite | 2 | 42 (97%), 47 | 1.1 | 0.011 |
    | Gemini 3.1 Flash-Lite | 1 | 42 (100%) | 1.0 | 0.010 |

    Every model is far below a perfect generator's 0.99. Nano and gpt-oss-120b differ by 0.001, too little to rank them without the interval 4.5 adds.
  - No model ever answered a multiple of 5, a number in 0–9 or 90–99, or a repeated digit such as 33. All 3,744 answers lie in 17–87, and none ends in 0, 5, 6 or 9. So the edges, multiples-of-10, multiples-of-5 and repeated-digits checks give all six models the same scores: 0.800, 0.900, 0.800 and 0.910. That's what avoiding a set entirely scores, since a gap is the share of answers that would have to change, and that's only the set's ideal share (D7). On a leaderboard, 0.90 reads as mild for a model that never picks a round number (D18).
  - gpt-oss-120b scores 1.000 on primes, with 25.3% of its answers prime. Its three favorites, 57, 42 and 73, of which only 73 is prime, happen to land near the ideal share. A share check measures one share, not the distribution, which is why the headline leaves it out (D8).
  - Pooling the prompts barely flatters these six models (D17). For each model, its mean uniformity within each prompt is within 0.002 of its pooled uniformity, because its favorite is the same under every prompt. The runner-up does depend on the wording: Haiku gave 47 to 29 of the 104 casual prompts and to 2 of the 104 plain ones, and nano's second choice is 37, 47, 57 or 73 depending on the prompt.
  - Gemma and gpt-oss-120b reasoned at a median of 120 and 81 output tokens a draw, and at most 536 and 219. The other four used 2–5.
  - No model is short of 600, so 4.4's top-ups aren't needed (D3).
  - The task page shows only version 2's runs: 7 results, the six models at `600.00` and the creation run. The pilot's runs stay on version 1. Below the results, the page shows the account's AI quota: $10 a day and $100 a month, with $0.13 of the day's used after the pilot and the full runs (seen 2026-11-14). So quota allows far more draws than 600 per model (see "Volume").
- [x] **4.4 Top-ups** for any model short of 600, if D3 chooses to accumulate draws across runs. Not needed: no draw failed, and every model has 624 (4.3).
- [x] **4.5a Intervals.** Give every score a 95% interval, offline, so that the scoring tasks can return `(value, CI)` (1.3) and the tie between nano and gpt-oss-120b can be judged. Settles D19 and the "Volume" question of a second run.

  `score_intervals` in [randombench/checks.py](randombench/checks.py) gives each score's ± half-width. That's 1.96 times the score's standard deviation over 10,000 bootstrap resamples of the model's replies, invalid ones included. It's seeded, so a rerun gives the same intervals. One call takes 0.09 s at 624 replies. `python -m randombench.score` prints the intervals under the scores. [randombench/calibration.py](randombench/calibration.py) now holds the made-up models and their true scores, and `python -m randombench.calibration --coverage` checks how often the intervals hold the true score.

  *Found:*
  - The headlines with their intervals:

    | Model | Headline | 95% interval |
    |---|---|---|
    | GPT-5.4 nano | 0.0327 | ±0.0034 |
    | gpt-oss-120b | 0.0325 | ±0.0016 |
    | Gemma 4 26B A4B | 0.0214 | ±0.0012 |
    | Claude Haiku 4.5 | 0.0125 | ±0.0006 |
    | Gemini 3.5 Flash-Lite | 0.0106 | ±0.0003 |
    | Gemini 3.1 Flash-Lite | 0.0100 | ±0 |

    Nano and gpt-oss-120b tie: the gap between them, 0.0002, is an eighth of the narrower interval. Every other pair of adjacent models is apart, Gemini 3.5 Flash-Lite from 3.1 included. A second run would narrow the intervals by about √2, to roughly ±0.0024 and ±0.0011, so it wouldn't break the tie. Resolving a gap that small would take about 350 times the draws, and the true gap may be zero. So there's no second run ("Volume").
  - Over 500 simulated runs of 624 replies, here is how often each interval held the model's true score:

    | Model | Headline | Valid | Uniformity | Primes | ×10 | ×5 | Favorite | Edges | Repeats | Last digit |
    |---|---|---|---|---|---|---|---|---|---|---|
    | Perfect | 99% | 100% | 99% | 99% | 98% | 99% | 100% | 98% | 98% | 100% |
    | Near-perfect | 94% | 100% | 94% | 99% | 98% | 98% | 99% | 98% | 98% | 100% |
    | Prime-leaning | 92% | 100% | 92% | 94% | 91% | 95% | 100% | 98% | 84% | 94% |
    | LLM-like | 94% | 100% | 94% | 95% | 94% | 92% | 98% | 94% | 92% | 94% |
    | Two tied | 97% | 100% | 97% | 95% | 94% | 95% | 90% | 97% | 93% | 95% |
    | Edge-avoiding | 99% | 100% | 99% | 99% | 99% | 98% | 100% | 100% | 100% | 100% |
    | One value | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
    | LLM-like, 10% invalid | 96% | 94% | 95% | 95% | 93% | 93% | 98% | 96% | 95% | 95% |

    Where a model's biases stand clear of the noise, as in the LLM-like and two-tied models, the intervals hold the true score 90–98% of the time, close to the 95% they claim. Near a perfect score they miss in both directions, because scores are capped at 1 and the squared gap is floored at 0. A perfect generator's intervals hold the true score 98–100% of the time, more often than 95%. The prime-leaning model's repeated-digits score, whose true value is 0.985, is held only 84% of the time, because its gap from the ideal, 1.5 points, is about the size of the share's sampling noise. Among the real models, only gpt-oss-120b's primes score sits at the cap, at 1.000 ±0.022. Its ± is symmetric, the form Kaggle stores, so the interval runs past 1.
  - A resample holds only answers the model gave. Gemini 3.1 Flash-Lite answered 42 all 624 times, so every one of its intervals is ±0. So is every model's interval on the four share checks whose sets no model ever picked (4.3). This understates the uncertainty only a little. With 624 answers that are all 42, the true share of other answers is likely below 0.5% (3/624, the "rule of three"), which would lift uniformity by about 0.0001.
  - gpt-oss-120b's favorite-number score has the widest interval, ±0.108. Its 57s and 42s are nearly tied, so the score swings from resample to resample (4.2).
- [ ] **4.5b Scoring tasks.** Ten task files, one per score, each holding the stored draws and the shared scoring code (D6, D13, D20). Run with `-m X`, a task parses and scores X's stored replies and returns `(value, ±)` (D19). It never prompts X. It returns `False` for a model with no stored draws, such as the creation run's (D12), and for a score of `None` (3.4). The titles are under "Names". Offline tests run each file with stand-in models, as `test_collect.py` does, and check its result against `python -m randombench.score`.
- [ ] **4.6 Scores on Kaggle.** Push the scoring tasks with the stored draws, and run each one against the six models. They never prompt a model, so they spend no quota. Still, it's 10 tasks × 6 models = 60 runs every time the data changes. Add every task to the RandomBench Benchmark on the website. Each task row's menu there can run that task for all the Benchmark's models at once (1.3). Check that each run's interval matches `python -m randombench.score`. The server's numpy may differ from the local 2.5.4 and draw different resamples.

### Phase 5: release

- [ ] **5.1 README.** Covers the method, the results, and the limitations, such as no temperature control, reliance on provider defaults, and scores pooled across prompts, which flatter a model whose answer depends on the wording (D17).
- [ ] **5.2 Publish.** A public Benchmark needs public tasks (1.3), so the collection task and every scoring task get published. `kaggle b t publish` can't be undone from the CLI, so each one needs an explicit go-ahead.

## Open decisions

| ID | Question | Leaning | Settled in |
|---|---|---|---|
| D1 | Reasoning: use each model's default, or force it on or off? | **Settled:** the provider default. Not every model can turn reasoning off: gpt-oss-120b offers only low, medium and high. Forcing a setting would treat models unevenly either way. The default also matches how people use each model, and how we already handle temperature. The default returns no reasoning traces (2.3), so only token counts show how much a model reasoned. Getting traces without changing the reasoning level would take provider-specific request options, and OpenAI's chat API never returns them. In the pilot, Gemma 4 and gpt-oss-120b reasoned by default, at up to 273 output tokens a draw, and 600 draws still cost at most $0.05 per model (4.2). | 2.3 |
| D2 | Leaderboard: one task that returns a dict of sub-scores plus the headline, or one task per sub-score? | **Settled:** one scoring task per score, all reading the same stored draws. See "Leaderboard" under Decided. | 1.2 |
| D3 | Runs that stop early: score each run on what it collected and merge all draws offline? Or have later runs top up to 600 from stored draws, through an attached Kaggle dataset? | **Settled:** self-contained runs, merged offline. Each run makes its own draws, and scoring pools every stored run of a model. The pilot's draws count with the full runs', giving 624 per model (4.2). No draw failed in 3,744 calls, so nothing needed topping up (4.3). If a later run stops early, it keeps the draws it finished: a run with no parent file is extracted from its per-draw files, skipping the empty ones (1.2, 2.5). Another run then adds to them. | 4.3 |
| D4 | How strict is parsing? | **Settled:** lenient on formatting, strict on content. Only digits count, and they must be one integer in 0–99. A restated range, number words and `47.0` make a reply invalid. See "Parsing" under Decided. | 2.2 |
| D5 | Data layout: one committed file per run under `data/`? How large do reasoning traces make it? | **Settled:** one JSON Lines file per run, `data/<model>/<run_id>.jsonl`, one draw per line, committed. Only runs downloaded from Kaggle, because only their download path holds a run id and a task version (2.5). A 600-draw run is about 230 KB. The default reasoning returns no traces (2.3), so traces add nothing unless D1 changes. | 2.5 |
| D6 | A push uploads a single `.py` file, but the offline analysis needs the same parsing and scoring code. Options: a source module plus a script that copies it into the pushed task file; the task file as the only copy; or code in an attached Kaggle dataset. | **Settled:** a copy in the task file. Each task file holds the `randombench` modules it uses between `# --- begin <path> ---` and `# --- end <path> ---` markers, and `python -m randombench.sync` refreshes the copies, so the file run locally is exactly the file pushed. Tests fail when a copy is out of date, when a task imports `randombench`, or when converting a task to a notebook would change a line (3.1). A copied module can't import `randombench` itself. | 3.1 |
| D7 | How does a measured bias become a sub-score in [0, 1]? | **Settled:** each sub-score is an effect size whose sampling noise cancels on average, so its expected value is the model's true score at any N, and no floor is subtracted. Uniformity and the last digit score the effective number of values out of 100 (of last digits, out of 10), from the unbiased repeat rate. The share checks score 1 − the gap from the ideal share, from an unbiased estimate of the squared gap. The favorite scores 1 − its leave-one-out share above 1%, scaled to [0, 1]. This replaced 3.2's TVD for uniformity and the last digit, because TVD has no unbiased estimate (3.3). | 3.3 |
| D8 | How is the headline score computed? | **Settled:** the share of replies that are valid times the uniformity score (the effective number of values out of 100). Uniformity already reflects every bias, so the other seven checks are diagnostics shown beside it and none counts twice. Any average of the nine failed in simulation: the share checks, whose lowest scores are 0.25 and up, lifted a model that always answers 42 to 0.59 with the plain mean, and invalid answers barely counted (3.5). | 3.5 |
| D9 | Prompt assignment: pick a prompt independently at random for each draw, or shuffle a balanced schedule with each prompt used exactly 100 times? | **Settled:** a balanced shuffle. The order is still random, and equal counts make prompts easier to compare. `balanced_schedule` in [tasks/collect.py](tasks/collect.py) does it (2.4). | 2.4 |
| D10 | What is the collection task's name? Its slug must stay stable once pushed, because a new slug makes a new task instead of a new version. | **Settled:** RandomBench Draws, slug `randombench-draws`. It names what the task holds, and its result is how many draws it collected (D14). The slug is permanent, but the title can change to any other that slugifies the same way. The CLI matches a task by slugifying its name, the library names files with spaces as underscores, and `%choose` normalizes the same way. `# %choose collect` names the task's variable, so the magic line holds no space. The scoring tasks follow the pattern: RandomBench Score for the headline, RandomBench Uniformity, and so on (4.1). | 4.1 |
| D11 | Should the draw sub-task catch its own errors and return them in its result, rather than letting the library mark the draw `ERRORED`? An errored sub-run keeps no result, so its prompt id would be lost, and the library won't retry inside a parent task anyway (1.1). | **Settled:** catch and return. Each draw's result always holds its draw index and prompt id, plus either the reply or the error (2.4). | 2.4 |
| D12 | Every push runs the whole task once against the server's default model, `gemini-3.8-flash` (1.2). For the real task, that's 600 calls per push on a model outside our six, and they count against quota. Should the creation run make only a few draws? If so, how does the task recognize it? A scoring task's creation run has a related problem: there are no stored draws for that model, but the run must still complete, or the task can't be run. | **Settled:** an allowlist. The collection task makes draws only when `llm.name` is one of the six in `MODELS`. Any other model, the creation run's included, makes no draws, prompts nothing and returns 0. The model is the only signal (1.3). An allowlist fails safe: a wrong id costs a run with no draws, which the pilot would show, not unwanted calls. So does a local run, because `.env`'s `LLM_DEFAULT` isn't on the list. An offline test runs the task with stand-in models and checks both cases (4.1). A scoring task returns `False` for a model with no stored draws. That lets its creation run complete, and the resulting FAIL doesn't count in an average of task scores (1.3). | 4.1 |
| D13 | How does a scoring task get the stored draws? A push uploads one file, so the options are an attached Kaggle dataset (`push -d`), or draws embedded in the task file by the script that builds it (D6). | **Settled:** embedded. A build script copies the draws from `data/` into each scoring task file (D6). It needs no new infrastructure, and each task version holds exactly the draws it scored. 1.3 showed it works, and the task page shows only the task function, not the draws. The raw replies for six models come to about 17 KB (2.5). Each file embeds a tally of each model's raw replies, each distinct reply text with its count, and the run ids the replies came from. Scoring pools a model's answers (D17), so it needs neither draw order nor prompts, and the raw text lets parsing run on Kaggle (decided in 4.5). | 2.5 |
| D14 | What does the collection task itself return? Its Result is a leaderboard cell too. If it returned the headline from its own run's draws, that could disagree with the scoring tasks once they score draws merged from several runs (D3). | **Settled:** the number of draws that got a reply, as it already did. 600 means all went well, a smaller number shows a run cut short or failed calls, and a creation run shows 0. It parses nothing, and every score comes from a scoring task (4.1). | 4.1 |
| D15 | The Benchmark shows its own average for each model and ranks by it (1.3). Public Benchmarks label it "Avg N tasks". One averages only some of its tasks, and another shows a 95% interval beside it. How is it computed, and how does it fit our headline (D8)? A plain mean over all our tasks would mix the headline with the sub-scores it's built from. It would also count the creation runs' FAIL against `gemini-3.8-flash`, unless that model can be left out. Can our tasks' intervals appear? | Settings offer six kinds of overall score (1.3). Use "weighted average of task scores, with confidence interval", weighting only the headline task. That needs two things: a weight can be 0, and the interval comes from the task's own `(value, CI)`. The ranking is then our headline with its interval, and the sub-scores show as columns beside it. Otherwise use "no overall score", with the headline as one column among the others. The private page didn't show the interval (1.3), so check it once RandomBench is public. The setting is a dropdown that can be changed at any time. Weighting the sub-score tasks instead would hand D8 to Kaggle, and its interval would treat sub-scores computed from the same 600 draws as independent. | Before 4.6 |
| D16 | What temperature do the models actually sample at? The library sends none, and it's unknown whether Model Proxy then uses the provider's default or a lower one of its own. Gemini 3.1 Flash-Lite answered 42 on all 12 draws (2.4). If the effective temperature were 0, RandomBench would measure each model's single likeliest answer, not its sampled distribution. The library can send a temperature: building the model with `ModelProxy(..., support_temperature=True)` overrides its `setdefault` (`model_proxy.py`, line 102). It's switched off because of an unresolved bug (b/439876083), whose nature isn't stated. | **Settled:** keep the provider's default. With `temperature=1.0` sent explicitly, all 12 Gemini draws were 42 again (2.4b), so the 42s come from the model, not from a low default. One caveat: if Model Proxy silently ignored the parameter, the probe would look exactly the same. Sending 2.0 would test that, but it isn't needed to keep the default. | 2.4b |
| D17 | Should the checks score a model's answers pooled across the six prompts, or within each prompt? | **Settled:** pooled. All of a model's valid answers count together. This flatters a model whose answer depends on the wording: one constant answer per prompt pools to uniformity 0.061, a fair pick among 6, against 0.010 within each prompt (3.5). The varied wording then supplies some of the randomness, which a user could do too. The README states this limitation (5.1). On the real draws, pooling barely matters: each model's favorite is the same under every prompt, and its pooled uniformity is within 0.002 of its mean within each prompt (4.3). | 3.5 |
| D18 | Which sub-scores get a scoring task, and how should the share checks read on a leaderboard? All six models avoid multiples of 5, edges and repeated digits entirely, so those four checks give every model the same score, 0.80–0.91 (4.3). That's right by D7's definition, since a gap is the share of answers that would have to change. But 0.90 reads as mild for a model that never picks a round number. | **Settled:** keep D7, and give every check its own task as planned. Identical scores are themselves a finding: all six models avoid these numbers completely. The README explains what the scores mean (5.1). The alternative was to rescale each share check by the largest gap possible on its side of the ideal, so that avoiding the set entirely scores 0, as answering only from it does. That would have changed D7. | 4.3 |
| D19 | How do the scoring tasks report uncertainty? | **Settled:** every task returns `(value, ±)`. The ± is 1.96 times the score's standard deviation over 10,000 bootstrap resamples of the model's replies, seeded, and one function covers all ten scores. A symmetric ± is the form Kaggle stores, though it can run past 0 or 1. The intervals hold the true score about 95% of the time when a bias stands clear of the noise, and miss in both directions next to a perfect score (4.5a). The alternatives were an interval on the headline only, half the width of the bootstrap's 2.5–97.5% range, and no intervals, which would leave D15's "with confidence interval" option nothing to use. | 4.5a |
| D20 | The ten scoring tasks differ only in which score they return, but a push uploads one file, so each needs a file of its own. Generate them, or write each by hand? | **Settled:** generate them. A build script writes the ten files from one template, with the shared modules and the data filled in, and a test fails if any file is stale. One edit to the template then reaches all ten. Handwritten files would repeat the same task body ten times. | 4.5 |
