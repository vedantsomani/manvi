"""Generate text from a trained speaker model on the command line and check for copying.

    python generate.py --speaker gandhi --order 3 --temperature 0.8 --seed 7 --start "my countrymen"
"""
import argparse

from copycheck import WINDOW, build_index, copy_rate, longest_copied_run
from corpus import ROOT, speakers, train_val_split
from model import MarkovModel, detokenize

LABEL = "[AI-generated text. Not a real quote from the speaker.]"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speaker", default="gandhi", choices=speakers())
    ap.add_argument("--start", default="", help='starting word(s), e.g. "my dear"')
    ap.add_argument("--length", type=int, default=100, help="target length in tokens (words + punctuation)")
    ap.add_argument("--order", type=int, default=2, help="context words the model looks at (1-4)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=None, help="fix the random seed to repeat a result")
    ap.add_argument("--no-check", action="store_true", help="skip the copy check")
    args = ap.parse_args()

    path = ROOT / "model" / f"{args.speaker}_markov.json"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: python train_markov.py")
    m = MarkovModel.from_json(path)
    tokens, trace = m.generate(args.start, args.length, args.order, args.temperature, args.seed)
    print("\n" + detokenize(tokens) + "\n")
    print(LABEL)
    if not args.no_check:
        index = build_index(train_val_split(args.speaker)[0])
        steps = [t for t in trace if t]
        backoffs = sum(t["order_used"] < args.order for t in steps)
        print(f"\nCopy check ({WINDOW}-word windows found verbatim in training text): "
              f"{copy_rate(tokens, index):.0f}%")
        print(f"Longest verbatim run: {longest_copied_run(tokens, index)} words")
        print(f"Backoff used on {backoffs}/{len(steps)} steps")


if __name__ == "__main__":
    main()
