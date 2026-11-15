import pytest

from randombench.parsing import parse_reply


@pytest.mark.parametrize(
    "reply, expected",
    [
        # Formatting only
        ("47", 47),
        ("47.", 47),
        ("  47\n", 47),
        ("**47**", 47),
        ("`47`", 47),
        ("```\n47\n```", 47),
        (r"$\boxed{47}$", 47),
        ("- 47", 47),
        ("...47", 47),
        # Words around exactly one integer
        ("Sure! 47", 47),
        ("My number is 47.", 47),
        ("Here you go: 47 🎲", 47),
        ("47 (forty-seven)", 47),
        # The same integer more than once
        ("47\n\n**47**", 47),
        ("07", 7),
        # The ends of the range
        ("0", 0),
        ("00", 0),
        ("99", 99),
        # Digits from other scripts are still digits: full-width 47
        ("４７", 47),
    ],
)
def test_valid_replies(reply, expected):
    assert parse_reply(reply) == expected


@pytest.mark.parametrize(
    "reply",
    [
        # More than one integer
        "47 or 52",
        "Between 23 and 71, I pick 47.",
        "47/100",
        # Grok 4.6's reply with a confidence score after it, twice (4.10, 4.11)
        "47\n\\confidence{80}",
        # A restated range counts as more integers (D4)
        "Here's a random number from 0 to 99: 47",
        "0-99: 47",
        # Out of range
        "100",
        "-5",
        "−5",
        # Not an integer (D4)
        "3.5",
        "47.0",
        "forty-seven",
        # No number at all
        "",
        "   ",
        "I can't generate a truly random number.",
    ],
)
def test_invalid_replies(reply):
    assert parse_reply(reply) is None
