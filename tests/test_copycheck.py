from copycheck import build_index, copied_mask, copy_rate, distinct_n, longest_copied_run
from model import tokenize

SOURCE = "one two three four five six seven eight nine ten"


def test_full_copy_is_100_percent():
    idx = build_index([SOURCE])
    toks = tokenize(SOURCE)
    assert copy_rate(toks, idx) == 100.0
    assert longest_copied_run(toks, idx) == 10
    assert all(copied_mask(toks, idx))


def test_novel_text_is_zero_and_case_insensitive():
    idx = build_index([SOURCE])
    assert copy_rate(tokenize("a b c d e f g h i j"), idx) == 0.0
    assert copy_rate(tokenize("ONE two Three four five six seven eight"), idx) == 100.0


def test_partial_copy_and_punctuation_ignored():
    idx = build_index([SOURCE])
    toks = tokenize("zero one two, three four five six seven eight. x")
    # windows: zero..seven (no), one..eight (yes), two..x (no)
    assert round(copy_rate(toks, idx), 1) == 33.3
    mask = copied_mask(toks, idx)
    assert mask[0] is False and mask[1] is True and mask[-1] is False


def test_short_text_and_distinct():
    idx = build_index([SOURCE])
    assert copy_rate(tokenize("one two"), idx) == 0.0
    assert distinct_n(tokenize("a a a b"), 1) == 0.5
    assert distinct_n(tokenize("a b a b"), 2) == 2 / 3
