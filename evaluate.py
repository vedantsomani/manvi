"""Phase 5: evaluate every speaker x model on held-out text and generated samples.

Metrics
  perplexity  - on the 10% validation split, per word/punctuation token (lower = better)
  distinct-1/2 - unique unigrams / bigrams over all generated samples (higher = more varied)
  copy rate   - % of 8-word windows in generated text that appear verbatim in training text

Writes reports/results.json, reports/results.md and reports/*.png.
    python evaluate.py              # Markov + GPT-2 (if model/<speaker>_gpt2 exists)
    python evaluate.py --no-gpt2
"""
import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import neural
from copycheck import build_index, copy_rate, distinct_n
from corpus import ROOT, display_name, speakers, train_val_split
from model import MarkovModel, detokenize, tokenize

REPORTS = ROOT / "reports"
SAMPLES, LENGTH = 20, 150
# Reference categorical palette (dataviz skill), first three slots: validated all-pairs.
SPEAKER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def prompts_for(m, speaker):
    """Same two-word sentence openings for every model of a speaker."""
    rng = random.Random(f"prompts-{speaker}")
    return [detokenize(list(rng.choice(m.starts))[:2]) for _ in range(SAMPLES)]


def sample_metrics(samples, index):
    toks_all = [t for s in samples for t in s]
    return {"copy_rate": sum(copy_rate(s, index) for s in samples) / len(samples),
            "distinct_1": distinct_n(toks_all, 1), "distinct_2": distinct_n(toks_all, 2)}


def evaluate_speaker(sp, use_gpt2):
    train_docs, val = train_val_split(sp)
    index = build_index(train_docs)
    path = ROOT / "model" / f"{sp}_markov.json"
    m = MarkovModel.from_json(path)
    prompts = prompts_for(m, sp)
    rows = []

    # Human reference: the held-out speech itself.
    human = [tokenize(c)[:LENGTH] for c in val[:SAMPLES]]  # same size as model samples
    rows.append({"model": "Held-out human text", "perplexity": None, **sample_metrics(human, index)})

    for order in range(1, m.max_order + 1):
        samples = [m.generate(p, LENGTH, order, 1.0, seed=i, extra=0)[0] for i, p in enumerate(prompts)]
        rows.append({"model": f"Markov order {order}", "perplexity": m.perplexity(val, order)[0],
                     **sample_metrics(samples, index)})

    if use_gpt2 and neural.available(sp):
        t0 = time.time()
        model, tok = neural.load(sp)
        ppl = neural.perplexity_per_word_token(model, tok, [detokenize(tokenize(c)) for c in val],
                                               lambda t: len(tokenize(t)))
        samples = [tokenize(neural.generate(model, tok, p, max_new_tokens=220, seed=i))[:LENGTH]
                   for i, p in enumerate(prompts)]
        meta = json.loads((neural.gpt2_path(sp) / "catbot_meta.json").read_text(encoding="utf-8"))
        rows.append({"model": f"GPT-2 fine-tuned ({meta['base_model']})", "perplexity": ppl,
                     **sample_metrics(samples, index)})
        print(f"  gpt2 eval {time.time() - t0:.0f}s")
        del model
    return rows


def cross_perplexity(sps, order=2):
    """Markov model of A scored on held-out text of B: lower on the diagonal = distinct styles."""
    models = {a: MarkovModel.from_json(ROOT / "model" / f"{a}_markov.json") for a in sps}
    vals = {b: train_val_split(b)[1] for b in sps}
    return [[models[a].perplexity(vals[b], order)[0] for b in sps] for a in sps]


# ---------------------------------------------------------------- charts
def _style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def grouped_bars(results, metric, title, ylabel, out, fmt="{:.0f}", skip_none=True):
    sps = list(results)
    models = [r["model"] for r in results[sps[0]] if not (skip_none and r[metric] is None)]
    for sp in sps[1:]:  # GPT-2 may exist for some speakers only
        for r in results[sp]:
            if r["model"] not in models and r[metric] is not None:
                models.append(r["model"])
    fig, ax = plt.subplots(figsize=(8.6, 3.9), dpi=150)
    width = 0.8 / len(sps)
    for j, sp in enumerate(sps):
        vals = {r["model"]: r[metric] for r in results[sp]}
        xs = [i + (j - (len(sps) - 1) / 2) * width for i in range(len(models))]
        ys = [vals.get(mn) or 0 for mn in models]
        bars = ax.bar(xs, ys, width * 0.92, color=SPEAKER_COLORS[j % 3], label=display_name(sp))
        for b, y in zip(bars, ys):
            if y:
                ax.text(b.get_x() + b.get_width() / 2, y, fmt.format(y), ha="center", va="bottom",
                        fontsize=7, color=MUTED)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([mn.replace(" fine-tuned ", "\n") for mn in models], fontsize=8.5)
    _style(ax, title, ylabel)
    ax.legend(frameon=False, fontsize=8.5, ncol=len(sps), loc="upper left", bbox_to_anchor=(0, 1.02))
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def heatmap(sps, matrix, out):
    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=150)
    im = ax.imshow(matrix, cmap="Blues_r")
    names = [display_name(s) for s in sps]
    ax.set_xticks(range(len(sps)), names, fontsize=8.5)
    ax.set_yticks(range(len(sps)), names, fontsize=8.5)
    ax.set_xlabel("held-out text of ...", color=MUTED, fontsize=9)
    ax.set_ylabel("Markov model (order 2) of ...", color=MUTED, fontsize=9)
    lo, hi = min(map(min, matrix)), max(map(max, matrix))
    for i in range(len(sps)):
        for j in range(len(sps)):
            v = matrix[i][j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=10,
                    color="white" if v < (lo + hi) / 2 else INK)
    ax.set_title("Cross-speaker perplexity (lower = more similar)", loc="left", fontsize=11, color=INK)
    ax.tick_params(length=0, colors=MUTED)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def write_markdown(results, cross, sps, out):
    lines = ["# CATbot evaluation results", "",
             f"Validation = 10% held-out text per speaker. {SAMPLES} samples x {LENGTH} tokens per model, "
             "temperature 1.0 (GPT-2: top-p 0.95), same two-word prompts for every model.", "",
             "| Speaker | Model | Perplexity ↓ | Distinct-1 ↑ | Distinct-2 ↑ | Copy rate (8-word) ↓ |",
             "|---|---|---:|---:|---:|---:|"]
    for sp in sps:
        for r in results[sp]:
            ppl = "–" if r["perplexity"] is None else f"{r['perplexity']:.1f}"
            lines.append(f"| {display_name(sp)} | {r['model']} | {ppl} | {r['distinct_1']:.3f} | "
                         f"{r['distinct_2']:.3f} | {r['copy_rate']:.1f}% |")
    lines += ["", "Perplexity is per word/punctuation token for every model (GPT-2's total BPE "
              "log-loss is divided by the same token count), so the numbers are comparable. "
              "The *held-out human text* row is the speaker's real unseen speech. Its copy rate "
              "shows how often a person naturally repeats their own 8-word phrases.", "",
              "## Cross-speaker perplexity (Markov order 2)", "",
              "| model \\ text | " + " | ".join(display_name(s) for s in sps) + " |",
              "|---|" + "---:|" * len(sps)]
    for a, row in zip(sps, cross):
        lines.append(f"| {display_name(a)} | " + " | ".join(f"{v:.0f}" for v in row) + " |")
    lines += ["", "![perplexity](perplexity.png)", "", "![copy rate](copy_rate.png)", "",
              "![distinct-2](distinct2.png)", "", "![cross](cross_perplexity.png)", ""]
    Path(out).write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-gpt2", action="store_true")
    args = ap.parse_args()
    REPORTS.mkdir(exist_ok=True)
    sps = speakers()
    results = {}
    for sp in sps:
        print(f"evaluating {sp} ...")
        results[sp] = evaluate_speaker(sp, not args.no_gpt2)
    cross = cross_perplexity(sps)
    (REPORTS / "results.json").write_text(json.dumps({"results": results, "cross_perplexity": cross,
                                                      "speakers": sps}, indent=2), encoding="utf-8")
    grouped_bars(results, "perplexity", "Validation perplexity (lower is better)", "perplexity per token",
                 REPORTS / "perplexity.png")
    grouped_bars(results, "copy_rate", "Copied 8-word windows (lower = more original)", "% of windows",
                 REPORTS / "copy_rate.png", skip_none=False)
    grouped_bars(results, "distinct_2", "Distinct-2 diversity (higher = more varied)", "unique bigrams / all",
                 REPORTS / "distinct2.png", fmt="{:.2f}", skip_none=False)
    heatmap(sps, cross, REPORTS / "cross_perplexity.png")
    write_markdown(results, cross, sps, REPORTS / "results.md")
    print(f"Wrote {REPORTS / 'results.md'} and charts.")


if __name__ == "__main__":
    main()
