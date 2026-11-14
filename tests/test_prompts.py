import pytest

from randombench.prompts import PROMPTS, Prompt, prompt_id, prompt_problems

# Pinned, so that any edit to a prompt's name or text fails here. If the edit
# is intended and the prompt has never been used in a run, update its id
# below. If it has been used, restore it and add a new prompt (PLAN.md 2.1).
EXPECTED_IDS = [
    "plain-47858739",
    "uniform-66c21a46",
    "raffle-66f5f2a6",
    "terse-b228d566",
    "role-b701252c",
    "casual-d08bc022",
]


# Ids


def test_id_is_name_and_text_hash():
    assert prompt_id("plain", "Pick a number.") == "plain-cf160a12"


def test_id_hashes_the_utf8_bytes():
    assert prompt_id("dash", "0–99") == "dash-0cd4ecc3"


def test_any_edit_changes_the_id():
    base = prompt_id("plain", "Pick a number.")
    assert prompt_id("plain", "Pick a number. ") != base
    assert prompt_id("plain", "pick a number.") != base
    assert prompt_id("other", "Pick a number.") != base


def test_prompt_ids_are_pinned():
    assert [p.id for p in PROMPTS] == EXPECTED_IDS


# Style rules


def make(name="plain", text="Pick a random integer from 0 to 99."):
    return Prompt(name=name, author="test", text=text)


@pytest.mark.parametrize("name", ["plain", "raffle-v2", "a1"])
def test_good_names_pass(name):
    assert prompt_problems(make(name=name)) == []


@pytest.mark.parametrize("name", ["Plain", "raffle_v2", "2nd", "plain prompt", ""])
def test_bad_names_fail(name):
    assert len(prompt_problems(make(name=name))) == 1


@pytest.mark.parametrize(
    "text",
    [
        "Pick from 0 to 99.",
        "Pick from 0–99, so 0 and 99 count.",
        "Pick an integer in [0, 99].",
    ],
)
def test_texts_stating_the_range_pass(text):
    assert prompt_problems(make(text=text)) == []


@pytest.mark.parametrize(
    "text",
    [
        "Pick from 1 to 100.",
        "Pick from 0 to 99, like 42.",
        "Pick from 0 to 99.5.",
        "Pick a random number.",
        "Pick a number up to 99.",
        "Pick from 00 to 99.",
    ],
)
def test_texts_not_stating_the_range_fail(text):
    assert len(prompt_problems(make(text=text))) == 1


def test_each_broken_rule_gets_one_message():
    assert len(prompt_problems(make(name="Bad", text="Pick from 1 to 100."))) == 2


@pytest.mark.parametrize("prompt", PROMPTS, ids=lambda p: p.name)
def test_every_prompt_passes(prompt):
    assert prompt_problems(prompt) == []


# The prompt set


def test_six_prompts():
    # 600 draws split into 100 per prompt (PLAN.md D9).
    assert len(PROMPTS) == 6


def test_names_are_unique():
    names = [p.name for p in PROMPTS]
    assert len(set(names)) == len(names)


def test_every_prompt_has_an_author():
    assert all(p.author for p in PROMPTS)
