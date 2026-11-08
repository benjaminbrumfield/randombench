# RandomBench

**How well do language models generate random numbers?**

People are famously bad at it. We avoid round numbers, shy away from the ends of the range and reach for 7 more often than chance would. Language models learned from what people write, so they may have the same habits. RandomBench measures how random their numbers really are.

For a fun look at the human side, watch Veritasium's [Why is this number everywhere?](https://www.youtube.com/watch?v=d6iQrh2TK98)

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
