"""Steps 5 and 6: generate new text and check it is not just copying."""
import argparse
import random
from pathlib import Path

from model import build_model, detokenize, generate, load_docs

ROOT = Path(__file__).resolve().parent
LABEL = "[AI-generated text. Not a real quote from the speaker.]"


def build_index(docs, n=8):
    """Set of fingerprints of every n-token window in the source speeches."""
    index = set()
    for d in docs:
        low = [t.lower() for t in d]
        for i in range(len(low) - n + 1):
            index.add(hash(tuple(low[i:i + n])))
    return index


def copy_stats(out_tokens, index, n=8):
    """Return (percent of n-token windows found verbatim in the source, longest copied run in tokens)."""
    low = [t.lower() for t in out_tokens]
    flags = [hash(tuple(low[i:i + n])) in index for i in range(len(low) - n + 1)]
    if not flags:
        return 0.0, 0
    best = run = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    return 100 * sum(flags) / len(flags), (n + best - 1 if best else 0)


def copy_report(out_tokens, docs, n=8):
    """Step 6: how much of the output is a word-for-word copy of the source?"""
    if len(out_tokens) < n:
        return f"Output shorter than {n} tokens; nothing to check."
    pct, longest = copy_stats(out_tokens, build_index(docs, n), n)
    longest = f"{longest} tokens" if longest else f"under {n} tokens"
    verdict = "OK" if pct < 20 else "HIGH: mostly copying. Need more data or a lower --order."
    return (f"Copy check ({n}-token windows found verbatim in source): {pct:.0f}%\n"
            f"Longest verbatim run: {longest}\nVerdict: {verdict}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="", help='starting word(s), e.g. "my dear"')
    ap.add_argument("--length", type=int, default=100, help="target length in tokens (words + punctuation)")
    ap.add_argument("--order", type=int, default=2, help="how many previous words the model looks at (1-3)")
    ap.add_argument("--seed", type=int, default=None, help="fix the random seed to repeat a result")
    ap.add_argument("--clean-dir", default=ROOT / "data" / "clean")
    ap.add_argument("--no-check", action="store_true", help="skip the copy check")
    args = ap.parse_args()

    docs = load_docs(args.clean_dir)
    total = sum(len(d) for d in docs)
    model, starts = build_model(docs, args.order)
    print(f"Model: {total:,} tokens, {len(model):,} states, order={args.order}")
    if total < 20_000:
        print("WARNING: small dataset. Expect near word-for-word copies.")

    rng = random.Random(args.seed)
    try:
        tokens = generate(model, starts, args.order, args.start, args.length, rng)
    except ValueError as e:
        raise SystemExit(str(e))
    print("\n" + detokenize(tokens) + "\n")
    print(LABEL)
    if not args.no_check:
        print("\n" + copy_report(tokens, docs))


if __name__ == "__main__":
    main()
