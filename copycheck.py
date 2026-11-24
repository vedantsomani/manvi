"""How much of a generated text is copied word-for-word from the training data?

A window of N consecutive tokens (default 8) counts as "copied" if the exact same
N tokens (case-insensitive) appear anywhere in the training text.
"""
from model import PUNCT, tokenize

WINDOW = 8


def _words(tokens):
    """Lowercase words only: punctuation differences should not hide a copy."""
    return [t.lower() for t in tokens if t not in PUNCT]


def build_index(docs, n=WINDOW):
    """Set of every n-word window in the training docs (strings or token lists)."""
    index = set()
    for d in docs:
        w = _words(tokenize(d) if isinstance(d, str) else d)
        for i in range(len(w) - n + 1):
            index.add(tuple(w[i:i + n]))
    return index


def copy_rate(tokens, index, n=WINDOW):
    """Percent of the output's n-word windows that appear verbatim in training data."""
    w = _words(tokens)
    windows = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    if not windows:
        return 0.0
    return 100.0 * sum(win in index for win in windows) / len(windows)


def copied_mask(tokens, index, n=WINDOW):
    """One bool per token: True if the token is inside any copied n-word window.
    Punctuation inherits the flag of the word before it."""
    word_pos = [i for i, t in enumerate(tokens) if t not in PUNCT]
    w = [tokens[i].lower() for i in word_pos]
    flag_word = [False] * len(w)
    for i in range(len(w) - n + 1):
        if tuple(w[i:i + n]) in index:
            for j in range(i, i + n):
                flag_word[j] = True
    mask, last = [False] * len(tokens), False
    wi = 0
    for i, t in enumerate(tokens):
        if t in PUNCT:
            mask[i] = last
        else:
            last = mask[i] = flag_word[wi]
            wi += 1
    return mask


def longest_copied_run(tokens, index, n=WINDOW):
    """Length in words of the longest stretch covered by consecutive copied windows."""
    w = _words(tokens)
    best = run = 0
    for i in range(len(w) - n + 1):
        run = run + 1 if tuple(w[i:i + n]) in index else 0
        best = max(best, run)
    return n + best - 1 if best else 0


def distinct_n(tokens, n):
    """distinct-n: unique n-grams / total n-grams over words (higher = more varied)."""
    w = _words(tokens)
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return len(set(grams)) / len(grams) if grams else 0.0
