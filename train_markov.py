"""Train one n-gram Markov model per speaker on the 90% training split (Phase 3).

    python train_markov.py            # every speaker in data/speakers/
    python train_markov.py obama      # just one

Writes model/<speaker>_markov.json.
"""
import argparse
from pathlib import Path

from corpus import ROOT, display_name, speakers, train_val_split
from model import MAX_ORDER, MarkovModel


def train(speaker, max_order=MAX_ORDER, model_dir=ROOT / "model"):
    train_docs, val_chunks = train_val_split(speaker)
    m = MarkovModel(max_order).fit(train_docs)
    path = Path(model_dir) / f"{speaker}_markov.json"
    m.to_json(path, speaker=speaker, display_name=display_name(speaker),
              train_words=sum(len(d.split()) for d in train_docs),
              val_words=sum(len(c.split()) for c in val_chunks),
              train_tokens=m.n_tokens, vocab_size=m.vocab_size,
              contexts={str(k): len(m.counts[k]) for k in range(1, max_order + 1)})
    return m, path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("speakers", nargs="*")
    ap.add_argument("--max-order", type=int, default=MAX_ORDER)
    args = ap.parse_args()
    for sp in args.speakers or speakers():
        m, path = train(sp, args.max_order)
        size = path.stat().st_size / 1e6
        print(f"{sp:8s} tokens={m.n_tokens:7,d} vocab={m.vocab_size:6,d} "
              f"order-{m.max_order} contexts={len(m.counts[m.max_order]):7,d} -> {path.name} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
