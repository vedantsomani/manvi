"""Word-level n-gram Markov model (orders 1-4) with temperature sampling and backoff.

Generation: look at the last `order` words. If that context never appeared in training,
back off to the last order-1 words, and so on down to plain word frequencies (order 0).
Sampling weights are count ** (1 / temperature).

Scoring (perplexity): interpolated Witten-Bell smoothing, so every token, including
words never seen in training, gets a non-zero probability and perplexity is well defined.
"""
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

# A token is a word (apostrophes/hyphens allowed inside) or one punctuation mark.
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*|[.!?,;:]")
SENTENCE_END = {".", "!", "?"}
PUNCT = SENTENCE_END | {",", ";", ":"}
MAX_ORDER = 4
SEP = "␟"  # joins context words in JSON keys
FORMAT_VERSION = 2


def tokenize(text):
    return TOKEN_RE.findall(text.replace("’", "'"))


def detokenize(tokens):
    """Turn tokens back into readable text (attach punctuation, capitalise sentences)."""
    out, capitalize = [], True
    for tok in tokens:
        if tok in PUNCT:
            if out:
                out[-1] += tok
            else:
                out.append(tok)
            capitalize = tok in SENTENCE_END
            continue
        if capitalize:
            tok = tok[:1].upper() + tok[1:]
            capitalize = False
        out.append(tok)
    return " ".join(out)


def _key(ctx):
    return tuple(t.lower() for t in ctx)


class MarkovModel:
    """counts[k][context of k lowercase words] = Counter({next token (original case): count})."""

    def __init__(self, max_order=MAX_ORDER):
        if not 1 <= max_order <= MAX_ORDER:
            raise ValueError(f"max_order must be 1..{MAX_ORDER}")
        self.max_order = max_order
        self.counts = [defaultdict(Counter) for _ in range(max_order + 1)]  # counts[0][()] = unigrams
        self.starts = []   # sentence-opening token tuples (len <= max_order)
        self.meta = {}
        self._lc = None    # lazily built lowercase tables for scoring

    # ------------------------------------------------------------------ training
    def fit(self, docs):
        """docs: list of token lists (or raw strings, which get tokenized)."""
        for doc in docs:
            toks = tokenize(doc) if isinstance(doc, str) else list(doc)
            for i, tok in enumerate(toks):
                for k in range(0, self.max_order + 1):
                    if i - k < 0:
                        break
                    self.counts[k][_key(toks[i - k:i])][tok] += 1
                if (i == 0 or toks[i - 1] in SENTENCE_END) and tok not in PUNCT:
                    self.starts.append(tuple(toks[i:i + self.max_order]))
        self._lc = None
        return self

    @property
    def vocab_size(self):
        return len(self.counts[0][()])

    @property
    def n_tokens(self):
        return sum(self.counts[0][()].values())

    # ------------------------------------------------------------------ generation
    def candidates(self, context, order=None, temperature=1.0, top=None):
        """Return (order_used, [(token, prob), ...] sorted by prob) for the next token."""
        order = self.max_order if order is None else min(order, self.max_order)
        context = list(context)
        for k in range(min(order, len(context)), -1, -1):
            nxt = self.counts[k].get(_key(context[len(context) - k:]) if k else ())
            if nxt:
                return k, self._distribution(nxt, temperature, top)
        raise ValueError("Model is empty; train it first.")

    @staticmethod
    def _distribution(counter, temperature, top):
        items = counter.most_common()
        if temperature <= 1e-6:  # greedy
            best = items[0][1]
            items = [(w, c) for w, c in items if c == best]
            temperature = 1.0
        logs = [math.log(c) / temperature for _, c in items]
        m = max(logs)
        weights = [math.exp(x - m) for x in logs]
        z = sum(weights)
        dist = [(w, p / z) for (w, _), p in zip(items, weights)]
        return dist[:top] if top else dist

    def start_tokens(self, start_text, rng):
        if start_text and start_text.strip():
            toks = tokenize(start_text)
            if toks:
                return toks
        if not self.starts:
            raise ValueError("Model has no sentence starts; train on more text.")
        return list(rng.choice(self.starts))

    def generate(self, start_text="", length=100, order=None, temperature=1.0, seed=None,
                 extra=40, trace_top=8):
        """Return (tokens, trace). trace[i] describes how generated token i was chosen."""
        rng = random.Random(seed)
        order = self.max_order if order is None else order
        out = self.start_tokens(start_text, rng)
        trace = [None] * len(out)  # prompt tokens have no trace
        while len(out) < length + extra:
            used, dist = self.candidates(out, order, temperature)
            words = [w for w, _ in dist]
            probs = [p for _, p in dist]
            choice = rng.choices(range(len(words)), weights=probs)[0]
            out.append(words[choice])
            trace.append({"context": out[-1 - used:-1] if used else [], "order_used": used,
                          "chosen": words[choice], "p": probs[choice],
                          "n_options": len(words), "top": dist[:trace_top]})
            if len(out) >= length and out[-1] in SENTENCE_END:
                break
        return out, trace

    # ------------------------------------------------------------------ scoring
    def _lowercase_tables(self):
        if self._lc is None:
            lc = []
            for table in self.counts:
                new = {}
                for ctx, ctr in table.items():
                    c = Counter()
                    for w, n in ctr.items():
                        c[w.lower()] += n
                    new[ctx] = (c, sum(c.values()), len(c))
                lc.append(new)
            self._lc = lc
        return self._lc

    def prob(self, word, context, order=None):
        """Interpolated Witten-Bell P(word | context), case-insensitive, never zero."""
        lc = self._lowercase_tables()
        order = self.max_order if order is None else min(order, self.max_order)
        word = word.lower()
        uni, n, v = lc[0][()]
        p = (uni.get(word, 0) + 1) / (n + v + 1)  # add-one with an <unk> slot
        ctx = _key(context)
        for k in range(1, min(order, len(ctx)) + 1):
            entry = lc[k].get(ctx[len(ctx) - k:])
            if not entry:
                break  # longer contexts are unseen too
            c, total, distinct = entry
            p = (c.get(word, 0) + distinct * p) / (total + distinct)
        return p

    def perplexity(self, docs, order=None):
        """Per-token perplexity over docs (strings or token lists). Context restarts per doc."""
        order = self.max_order if order is None else order
        nll, n = 0.0, 0
        for doc in docs:
            toks = tokenize(doc) if isinstance(doc, str) else doc
            for i, tok in enumerate(toks):
                nll -= math.log(self.prob(tok, toks[max(0, i - order):i], order))
                n += 1
        return math.exp(nll / max(n, 1)), n

    # ------------------------------------------------------------------ persistence
    def to_json(self, path, **meta):
        self.meta.update(meta)
        payload = {
            "format_version": FORMAT_VERSION,
            "model_type": "word-level n-gram Markov chain",
            "max_order": self.max_order,
            "meta": self.meta,
            "counts": [{SEP.join(ctx): dict(nxt) for ctx, nxt in table.items()} for table in self.counts],
            "starts": [list(s) for s in self.starts],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    @classmethod
    def from_json(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"{path} is an old model format; retrain with train_markov.py")
        m = cls(data["max_order"])
        for k, table in enumerate(data["counts"]):
            m.counts[k] = defaultdict(Counter, {
                (tuple(key.split(SEP)) if key else ()): Counter(nxt) for key, nxt in table.items()})
        m.starts = [tuple(s) for s in data["starts"]]
        m.meta = data.get("meta", {})
        return m
