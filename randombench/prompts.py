"""The prompts that ask a model for a random number (PLAN.md 2.1).

Each draw sends one of these as the only message in a fresh conversation.
A prompt's id is computed from its name and its exact text, so editing the
text gives it a new id, and every stored draw points at the text it answered.

Style guide:
- Ask for a random choice. A model that isn't asked to be random can't be
  faulted for having a favorite.
- Say "integer" or "whole number", so a decimal is clearly a wrong answer.
- State the range as 0 to 99, in digits, with both ends included.
- Ask for the number only, so the parser (2.2) sees plain replies.
- Give no example number and mention no bias (7, primes, round numbers).
  Either could steer the answer.
- Vary everything else: wording, tone, length and framing, so no score
  depends on one phrasing.

prompt_problems() checks the rules that code can check.

Once a prompt has been used in a run, never edit or delete it. To change one,
add a new prompt.
"""

import hashlib
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    name: str
    author: str
    text: str

    @property
    def id(self) -> str:
        return prompt_id(self.name, self.text)


def prompt_id(name: str, text: str) -> str:
    """Return a prompt's id: its name, a hyphen, then the first 8 hex digits
    (lowercase) of the SHA-256 digest of its text encoded as UTF-8.

    The name only makes the id readable. The hash is the part that changes
    when the text does. prompt_id("plain", "Pick a number.") returns
    "plain-cf160a12".
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{name}-{digest[:8]}"


def prompt_problems(prompt: Prompt) -> list[str]:
    """Return one message for each rule below that the prompt breaks. An
    empty list means the prompt passes.

    1. Name: lowercase letters, digits and hyphens, starting with a letter.
       "plain" and "raffle-v2" pass; "Plain", "raffle_v2", "2nd" and "" fail.
    2. Numbers: every run of digits in the text is "0" or "99", and both
       appear. That's how the text states the range without giving an
       example number. "0 to 99" and "0–99, so 0 and 99 count" pass;
       "1 to 100", "0 to 99, like 42", "0 to 99.5" and "a random number" fail.

    A prompt that breaks both rules gets two messages, and one that breaks
    the numbers rule in several places still gets only one.
    """
    problems = []
    if not re.fullmatch(r"[a-z][a-z0-9-]*", prompt.name):
        problems.append(
            f"name {prompt.name!r} must be lowercase letters, digits and hyphens,"
            " starting with a letter"
        )
    # Compared as strings, so "00" or "099" doesn't pass for 0 or 99.
    numbers = set(re.findall(r"\d+", prompt.text))
    if numbers != {"0", "99"}:
        problems.append(
            f"text's numbers are {sorted(numbers)}; they must be exactly 0 and 99"
        )
    return problems


CLAUDE = "claude-opus-5-5"

PROMPTS = (
    Prompt(
        name="plain",
        author=CLAUDE,
        text="Pick a random integer from 0 to 99, inclusive. Reply with only the number.",
    ),
    Prompt(
        name="uniform",
        author=CLAUDE,
        text=(
            "Choose an integer uniformly at random from 0 to 99, inclusive, so that"
            " every value in that range is equally likely. Reply with the integer"
            " and nothing else."
        ),
    ),
    Prompt(
        name="raffle",
        author=CLAUDE,
        text=(
            "I'm drawing the winner of a raffle. The tickets are numbered from 0 to"
            " 99, and every ticket has the same chance. Draw one ticket at random and"
            " reply with only its number."
        ),
    ),
    Prompt(
        name="terse",
        author=CLAUDE,
        text="Random integer, 0–99 inclusive. Number only.",
    ),
    Prompt(
        name="role",
        author=CLAUDE,
        text=(
            "You are a random number generator. Output one integer in the range"
            " [0, 99], and nothing else."
        ),
    ),
    Prompt(
        name="casual",
        author=CLAUDE,
        text=(
            "hey, can you give me a random whole number between 0 and 99? 0 and 99"
            " both count. just the number please"
        ),
    ),
)
