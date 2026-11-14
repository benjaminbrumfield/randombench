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

1. Ask a model for a random integer from 0 to 99, about 600 times. Each request is a fresh conversation, so the model never sees its earlier answers.
2. Check the answers for patterns a truly random source wouldn't show:
   - too many primes, or too many round numbers like 10, 20 and 50
   - a lean toward numbers ending in 7
   - one favorite number that keeps coming up
   - avoiding, or crowding, the ends of the range
   - replies that aren't a valid number at all
3. Turn each check into a score from 0 to 1, then combine them into one headline score.

Even real randomness isn't perfectly even over 600 draws, so Python's random number generator is scored the same way. That shows what a perfect score looks like in practice.

## Models

The first runs will cover six models:

- GPT-5.4 nano
- Gemini 3.1 Flash-Lite (preview)
- Gemini 3.5 Flash-Lite
- Claude Haiku 4.5
- Gemma 4 26B A4B
- gpt-oss-120b

## Status

Early days. The design is drafted and the build is under way, but there are no results yet. [PLAN.md](PLAN.md) has the full design, the open questions and the step-by-step plan. This is going to be iterative and adaptive.

## Built with

[kaggle-benchmarks](https://github.com/Kaggle/kaggle-benchmarks). The benchmark runs on [Kaggle Benchmarks](https://www.kaggle.com/benchmarks).
