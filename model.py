"""Steps 3 and 4: split text into tokens and build the Markov chain model."""
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

# A token is a word (apostrophes/hyphens allowed inside) or one punctuation mark.
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*|[.!?,;:]")
SENTENCE_END = {".", "!", "?"}


def tokenize(text):
    return TOKEN_RE.findall(text)


def load_docs(clean_dir):
    """Return one token list per cleaned file."""
    files = sorted(Path(clean_dir).glob("*.txt"))
    if not files:
        raise SystemExit(f"No cleaned files in {clean_dir}. Run: python clean.py")
    return [tokenize(f.read_text(encoding="utf-8")) for f in files]


def build_model(docs, order=2):
    """
    model[(w1, ..., w_order)] = Counter({next_word: times_seen})
    States are lowercase so "My dear" and "my dear" share counts.
    Next words keep their original case so the output looks natural.
    Also returns `starts`: states that began a sentence (repeats = more likely).
    """
    model = defaultdict(Counter)
    starts = []
    for tokens in docs:
        for i in range(len(tokens) - order):
            state = tuple(t.lower() for t in tokens[i:i + order])
            model[state][tokens[i + order]] += 1
            begins_sentence = i == 0 or tokens[i - 1] in SENTENCE_END
            if begins_sentence and state[0] not in SENTENCE_END | {",", ";", ":"}:
                starts.append(state)
    return model, starts


def detokenize(tokens):
    """Turn tokens back into readable text."""
    out, capitalize = [], True
    for tok in tokens:
        if tok in {".", "!", "?", ",", ";", ":"}:
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


SEP = "␟"  # joins state words in the JSON export


def save_json(model, starts, path, order, **meta):
    payload = {"format_version": 1, "model_type": "word-level Markov chain", "order": order, **meta,
               "transitions": {SEP.join(k): dict(v) for k, v in model.items()},
               "starts": [list(s) for s in starts]}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_json(path):
    """Return (metadata, model, starts) from a JSON export."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    model = {tuple(k.split(SEP)): Counter(v) for k, v in data.pop("transitions").items()}
    starts = [tuple(s) for s in data.pop("starts")]
    return data, model, starts


def pick_start(model, starts, order, start_text, rng):
    """Return the starting tokens."""
    if not start_text:
        return list(rng.choice(starts))
    toks = tokenize(start_text)
    low = [t.lower() for t in toks]
    if not low:
        raise ValueError("--start has no usable words.")
    if len(low) >= order:
        if tuple(low[-order:]) not in model:
            raise ValueError(f"The phrase {' '.join(low[-order:])!r} never appears in the data. Try another start.")
        return toks
    # Start shorter than the model memory: pick a state that begins with it.
    matches = [s for s in model if list(s[:len(low)]) == low]
    if not matches:
        raise ValueError(f"The word(s) {' '.join(low)!r} never appear in the data. Try another start.")
    weights = [sum(model[s].values()) for s in matches]
    state = rng.choices(matches, weights=weights)[0]
    return toks + list(state[len(low):])


def generate(model, starts, order, start_text, length, rng, extra=40):
    """Walk the chain. After `length` tokens, keep going until a sentence ends (max `extra` more)."""
    out = pick_start(model, starts, order, start_text, rng)
    while True:
        nxt = model.get(tuple(t.lower() for t in out[-order:]))
        if not nxt:  # dead end (only happens at the very end of a source text)
            if len(out) >= length:
                break
            if out[-1] not in SENTENCE_END:
                out.append(".")
            out += list(rng.choice(starts))
            continue
        words, counts = zip(*nxt.items())
        out.append(rng.choices(words, weights=counts)[0])
        if len(out) >= length and out[-1] in SENTENCE_END:
            break
        if len(out) >= length + extra:
            break
    return out
