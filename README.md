# RandomBench

**How well can language models pick numbers at random?**

People are famously bad at it. We avoid round numbers, shy away from the ends of the range and reach for 7 more often than chance would. Are language models any better? RandomBench measures how random their numbers really are.

For a fun look at the human side, watch Veritasium's [Why is this number everywhere?](https://www.youtube.com/watch?v=d6iQrh2TK98)

## Why random numbers

On most benchmarks, answering at random is the baseline to beat. Here it's the target. A uniform random answer ignores everything but the range and needs no knowledge or judgment, and one line of Python gets it right. Even so, there are good reasons to expect it to be hard for language models.

Strictly, a model doesn't have to create randomness. Whenever the temperature is above zero, the sampler that picks each token supplies it. What the model controls is the distribution being sampled from. To score like a true random source, that distribution has to be flat: the same probability for each of the 100 answers, and none for anything else. Nothing in training pushes that way. A model that predicts text well learns how people answer "pick a random number," biases included. Post-training then rewards the best response, which tends to sharpen a model's output toward its favorite answers. In this task there is no best response. Any single answer is as good as any other, and only the pattern across many answers can be wrong.

The model never sees that pattern. Each draw is a fresh conversation, so it can't look back and even out its answers. It has to be uniform in every call, on its own, the way a fair coin is fair without remembering its last toss.

That makes the problem hard for researchers too:

- **Temperature can't fix it.** Raising it flattens the whole distribution, invalid replies included, and the favorite number stays the favorite.
- **Prompting can backfire.** A model told that people overuse 7 may avoid 7 and land on a new bias. Knowing about a bias isn't the same as being free of it.
- **Reasoning doesn't obviously help.** Any procedure a model works through in its reasoning needs a random starting point, and choosing one is the original problem again. In principle, though, combining several biased choices can wash the bias out, and whether models find that trick is worth watching.
- **Training aims at the wrong level.** Most training scores each reply on its own, and no single reply can show a bias that exists only across many.
- **The shortcuts are closed.** RandomBench allows no tools and no memory of earlier answers, so the only way to score well is for the model's own distribution to be flat.

A model that mastered this would have more than a party trick. It would have control over its own output distribution: the ability to give a spread of answers on request, not only its most likely one. That ability could turn out to be foundational.

- **Strategy.** In poker, negotiation or rock-paper-scissors, the best play often means randomizing, and a predictable player can be exploited.
- **Exploration.** An agent that retries a failed task the same way every time won't find the approach that works.
- **Discovery.** New ideas are, almost by definition, not the most typical ones. A model that always reaches for its likeliest answer tends to rediscover what's already known.
- **Diversity.** The pull toward a favorite number is likely the same pull that makes a model tell the same few jokes. Synthetic training data and simulated survey respondents inherit that sameness.

Those are only the uses that are easy to name. Randomness is one of the oldest and most powerful tools in computing and in nature. Randomized algorithms solve some problems faster than any known deterministic method. Monte Carlo methods estimate quantities that resist direct calculation. Evolution runs on random variation filtered by selection. Many of the most promising ways to get more out of language models work the same way: sample many attempts and keep what works. Such methods can only select from what the model proposes, so their reach is bounded by its range. A model that could sample faithfully from any distribution it was asked for, not just the one training left it with, might unlock abilities no one has thought to list.

Out in the world, the right distribution is rarely known, so the gap is hard to measure. Here the right distribution is known exactly: 1 in 100 for every answer. That makes this the cleanest place to measure how close a model can come to a distribution it was asked for, and to watch whether the gap closes.

## How it works

1. **Draw.** Each model is asked for a random integer from 0 to 99, hundreds of times. Every request is a fresh single-turn conversation with no tools, so the model never sees its earlier answers. The requests rotate through several wordings, each used equally often, from the terse `Random integer, 0–99 inclusive. Number only.` to a raffle draw ([randombench/prompts.py](randombench/prompts.py)). Every wording asks for a random integer, says both ends of the range count, and asks for the number only. None gives an example number or names a bias, since either could steer the answer. Temperature and reasoning are left at each provider's default.
2. **Parse.** Each reply is read leniently on formatting and strictly on content ([randombench/parsing.py](randombench/parsing.py)). `Sure! **47**.` counts as 47. `47 or 52`, `47.0`, `100` and `forty-seven` are invalid, and so is a reply that restates the range, since the prompt asked for the number only.
3. **Score.** A model's valid answers, pooled across the wordings, are checked for the biases below ([randombench/checks.py](randombench/checks.py)). Every score is in [0, 1], about 1 for a perfect random source, and comes with a 95% interval from a bootstrap of the model's replies.

The draws are made on [Kaggle Benchmarks](https://www.kaggle.com/benchmarks) by one task, RandomBench Draws ([tasks/collect.py](tasks/collect.py)). Every raw reply is stored in [data/](data/), and each score is a Kaggle task of its own that recomputes the score from those stored replies. So a new check or a new parsing rule never means asking the models again.

## Scores

The headline, **RandomBench Score**, is the share of replies that are valid times the uniformity score.

Uniformity asks how many values the model is really choosing among. A model that repeats itself as often as a fair pick among K values would gets a uniformity of K/100. A perfect generator has K = 100, and a model that always gives the same answer has K = 1. So a uniformity of 0.05 means the model is as repetitive as a fair pick among 5 numbers.

The other checks show which biases lie behind the headline. They don't count toward it. Uniformity already reflects every bias, and adding the checks in would count some biases twice: 7, 17, 37, 47, 67 and 97 are all primes that end in 7.

| Score | What it measures | Ideal | Lowest |
|---|---|---|---|
| RandomBench Score | Valid Answers × Uniformity | 1 | 0 |
| Valid Answers | The share of replies that are a single integer in 0–99 | 100% | 0 |
| Uniformity | The effective number of values, as K/100 | K = 100 | 0.01, one value |
| Primes | 1 − the gap from 25% primes (2, 3, 5, …, 97) | 25% | 0.25, all prime |
| Multiples of 10 | 1 − the gap from 10% multiples of 10 | 10% | 0.10 |
| Multiples of 5 | 1 − the gap from 20% multiples of 5 | 20% | 0.20 |
| Favorite Number | 1 − the favorite's share above 1%, scaled so that 100% scores 0 | 1% | 0, one value |
| Edges | 1 − the gap from 20% in 0–9 or 90–99 | 20% | 0.20 |
| Repeated Digits | 1 − the gap from 9% in 11, 22, …, 99 | 9% | 0.09 |
| Last Digit | The effective number of last digits, as K/10 | K = 10 | 0.10, one digit |

A gap is the share of answers that would have to change to reach the ideal share. That makes the share checks lenient toward avoidance. A model that never picks a multiple of 10 is only 10% away from the ideal, so it scores 0.90. One that picks nothing else scores 0.10.

Even a perfect random source isn't perfectly even over a finite sample, so it scores a little under 1. `python -m randombench.calibration <N>` simulates one and shows what it scores at N answers. Each score is estimated so that its sampling noise cancels on average. A model's expected score is then its true score at any number of answers, so models with different numbers of draws can be compared.

## Results

The models so far:

- GPT-5.4 nano
- Gemini 3.1 Flash-Lite (preview)
- Gemini 3.5 Flash-Lite
- Gemini 3.8 Flash
- Claude Haiku 4.5
- Gemma 4 26B A4B
- gpt-oss-120b

`python -m randombench.score` prints each model's current scores and their intervals from the stored replies. The Kaggle Benchmarks will show them too, once they're public.

What stands out so far:

- **No model comes close.** Even the best behaves like a fair pick among only a few numbers, where a perfect generator scores close to 1.
- **42 dominates.** It's the favorite of almost every model, and Gemini 3.1 Flash-Lite has answered nothing else. Gemini 3.8 Flash is the exception, with 73.
- **Whole regions go unused.** Every answer so far lies well inside the range. No model has answered a multiple of 5, a number in 0–9 or 90–99, or a repeated digit like 33, and nearly every answer ends in 2, 3 or 7. Avoiding a set entirely gives every model the same score on those four checks. Here the identical scores are the finding.
- **A share check can pass by accident.** gpt-oss-120b scores close to perfect on primes. Its share of primes is near the ideal, but it comes from three favorites, 57, 42 and 73, of which only 73 is prime. Gemini 3.8 Flash answers with a prime far more often than any other model.
- **Every reply is valid.** Every call so far has returned a reply, and every reply has been a bare number in 0–99, so every model scores perfectly on Valid Answers.
- **Reasoning doesn't fix it.** Gemini 3.8 Flash, gpt-oss-120b and Gemma 4 reason by default, and spend many tokens on a two-digit answer that the other models give directly. They're still far from uniform.

## Limitations

- **Provider defaults.** Temperature and reasoning are whatever each provider uses by default, because kaggle-benchmarks doesn't send a temperature through Kaggle's Model Proxy. RandomBench measures each model as it's served, not at the temperature that would suit it best. In a small probe, Gemini 3.1 Flash-Lite still answered 42 every time when a temperature was sent explicitly. So its 42s don't come from a low default temperature, unless the proxy ignores the parameter.
- **Pooled wordings.** A model's answers are scored together across the wordings. A model whose favorite changes with the wording therefore looks more random pooled than under any one wording. Most models so far have the same favorite under every wording, and pooling barely changes their scores. Gemini 3.8 Flash and gpt-oss-120b are the exceptions: their favorite changes with the wording, and pooling lifts their scores noticeably.
- **Lenient share checks.** A share check measures one share's distance from its ideal. Avoiding a set entirely costs only that set's ideal share, and a lopsided distribution can still hit the ideal by chance, as gpt-oss-120b's primes do. That's why the headline uses only validity and uniformity.
- **Intervals.** A bootstrap resample holds only answers the model gave, so a model that has only ever given one answer gets an interval of zero width, which understates its uncertainty slightly. Near a perfect score the intervals are approximate, because scores are capped at 1. The Favorite Number score swings widely when two favorites nearly tie, as gpt-oss-120b's 57s and 42s do. Kaggle shows only each score's value, not its interval.
- **One question, one number.** Each call asks for a single number in English, with no tools. Asking for many numbers at once, or letting a model run code, would measure something else.
- **A snapshot of a few models.** They're mostly small, fast models, run through Kaggle's Model Proxy. Providers can update a model behind the same name, and larger models may behave differently.

## Reproduce the scores

The scores are recomputed from the raw replies in [data/](data/), so this needs no API key and makes no model calls:

```bash
python -m venv .venv && source .venv/bin/activate   # Git Bash on Windows: .venv/Scripts/activate
pip install -r requirements-dev.txt
python -m pytest                      # the offline tests
python -m randombench.score           # every model's scores and intervals
python -m randombench.calibration <N> # what a perfect generator scores at N answers
```

On Windows, set `PYTHONUTF8=1` first. Making new draws needs a Kaggle account and spends Model Proxy quota. [CLAUDE.md](CLAUDE.md) lists every command.

## Repository

- [randombench/](randombench/): the prompts, the parser, the checks, and the scripts that extract, score and build.
- [tasks/collect.py](tasks/collect.py): RandomBench Draws, the Kaggle task that makes the draws.
- [tasks/scores/](tasks/scores/): the scoring tasks, one per score. `python -m randombench.build` generates them from [tasks/templates/score.py](tasks/templates/score.py), with the stored replies embedded.
- [data/](data/): every raw reply, one JSON Lines file per Kaggle run.
- [probes/](probes/): small experiments that tested the platform along the way.
- [PLAN.md](PLAN.md): the design, every decision and the reasons for it, and what each step found.

## Status

The first results are in. Publishing the tasks and the Kaggle Benchmarks comes next, and they'll be linked here then. This is going to be iterative and adaptive.

## Built with

[kaggle-benchmarks](https://github.com/Kaggle/kaggle-benchmarks). The benchmark runs on [Kaggle Benchmarks](https://www.kaggle.com/benchmarks).
