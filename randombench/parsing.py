"""Turn a model's raw reply into its number, or None if the reply is invalid
(PLAN.md 2.2, D4).

Lenient on formatting, strict on content. Markdown, LaTeX, punctuation and
words around the number are ignored. What counts are the numbers written in
digits: they must all be the same integer, and it must be in 0..99. So
"Sure! **47**." is 47, while "47 or 52", "3.5", "100" and "forty-seven" are
invalid. A reply that restates the range ("from 0 to 99: 47") is invalid too,
because it holds three integers. The prompt asked for the number only.

Raw replies are always stored, so these rules can change later without
calling a model again.
"""

import re

LOW, HIGH = 0, 99

# A number written in digits: an optional minus sign, digits, and an optional
# decimal part. A minus sign counts only when no digit comes right before it,
# so "0-99" is 0 and 99, not 0 and -99. U+2212 is the typographic minus.
_NUMBER = re.compile(r"(?<!\d)[-−]?\d+(?:\.\d+)?")


def parse_reply(reply: str) -> int | None:
    """Return the integer in [0, 99] that the reply gives, or None if it's invalid."""
    numbers = _NUMBER.findall(reply)
    if not numbers or any("." in n for n in numbers):
        return None
    values = {int(n.replace("−", "-")) for n in numbers}
    if len(values) != 1:
        return None
    (value,) = values
    return value if LOW <= value <= HIGH else None
