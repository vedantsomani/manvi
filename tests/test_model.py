import math

import pytest

from model import MarkovModel, detokenize, tokenize

DOCS = ["The cat sat on the mat. The cat ate the fish. A dog sat on the log."]


def test_tokenize_detokenize_roundtrip():
    toks = tokenize("hello, world. it's fine!")
    assert toks == ["hello", ",", "world", ".", "it's", "fine", "!"]
    assert detokenize(toks) == "Hello, world. It's fine!"


def test_counts_and_candidates():
    m = MarkovModel(2).fit(DOCS)
    order, dist = m.candidates(tokenize("the cat"), order=2)
    assert order == 2
    assert dict(dist) == pytest.approx({"sat": 0.5, "ate": 0.5})
    assert sum(p for _, p in dist) == pytest.approx(1.0)


def test_backoff_to_lower_order_for_unseen_context():
    m = MarkovModel(3).fit(DOCS)
    order, dist = m.candidates(["purple", "elephant", "the"], order=3)
    assert order == 1  # only "the" was seen
    assert dist[0][0] == "cat"
    order, _ = m.candidates(["zzz"], order=3)
    assert order == 0  # unigram fallback, never a dead end


def test_temperature_sharpens_and_flattens():
    m = MarkovModel(1).fit(DOCS)
    _, cold = m.candidates(["the"], 1, temperature=0.2)
    _, hot = m.candidates(["the"], 1, temperature=5.0)
    assert cold[0][1] > hot[0][1]
    _, greedy = m.candidates(["the"], 1, temperature=0)
    assert greedy == [("cat", 1.0)]


def test_generate_is_reproducible_and_traced():
    m = MarkovModel(2).fit(DOCS)
    a, trace = m.generate("the cat", length=12, order=2, seed=3)
    b, _ = m.generate("the cat", length=12, order=2, seed=3)
    assert a == b and a[:2] == ["the", "cat"]
    assert trace[0] is None and trace[2]["chosen"] == a[2]


def test_witten_bell_is_a_distribution():
    m = MarkovModel(2).fit(DOCS)
    vocab = {w.lower() for w in m.counts[0][()]}
    ctx = tokenize("the cat")
    total = sum(m.prob(w, ctx) for w in vocab) + m.prob("<never-seen>", ctx)
    assert total == pytest.approx(1.0, abs=1e-9)
    ppl, n = m.perplexity(["the cat sat"], 2)
    assert n == 3 and math.isfinite(ppl) and ppl > 1


def test_json_roundtrip(tmp_path):
    m = MarkovModel(3).fit(DOCS)
    path = tmp_path / "m.json"
    m.to_json(path, speaker="test")
    m2 = MarkovModel.from_json(path)
    assert m2.meta["speaker"] == "test"
    assert m2.candidates(["the", "cat"], 3) == m.candidates(["the", "cat"], 3)
    assert m2.generate(seed=1, length=10) == m.generate(seed=1, length=10)
