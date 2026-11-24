"""Fine-tune GPT-2 on each speaker (Phase 4).

    python train_gpt2.py                         # all speakers, gpt2-medium
    python train_gpt2.py obama --base distilgpt2  # one speaker, smaller model

Uses the same 90/10 split as the Markov models (corpus.train_val_split).
Writes model/<speaker>_gpt2/ (bf16 safetensors) and reports/<speaker>_gpt2_loss.{png,json}.
"""
import argparse
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
                          default_data_collator, set_seed)

from corpus import ROOT, display_name, speakers, train_val_split


def blocks(texts, tok, size):
    """Concatenate texts (EOS between them) and cut into equal blocks for causal LM training."""
    ids = []
    for t in texts:
        ids += tok(t)["input_ids"] + [tok.eos_token_id]
    n = len(ids) // size * size
    rows = [ids[i:i + size] for i in range(0, n, size)]
    if not rows and ids:  # tiny corpus: one short block
        rows = [ids[:size]]
    return Dataset.from_dict({"input_ids": rows, "attention_mask": [[1] * len(r) for r in rows],
                              "labels": rows})


def plot_history(history, speaker, base, out_png):
    tr = [(h["step"], h["loss"]) for h in history if "loss" in h]
    va = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
    if tr:
        ax.plot(*zip(*tr), color="#3b6fd8", lw=1.8, label="train")
    if va:
        ax.plot(*zip(*va), color="#e0702b", lw=1.8, marker="o", ms=3, label="validation")
    ax.set_title(f"{display_name(speaker)}: {base} fine-tuning loss", fontsize=11, loc="left")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy (nats / BPE token)")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def train_one(speaker, args):
    t0 = time.time()
    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.base)
    tok.pad_token = tok.eos_token
    train_docs, val_chunks = train_val_split(speaker)
    train_ds = blocks(train_docs, tok, args.block)
    val_ds = blocks(val_chunks, tok, args.block)
    print(f"[{speaker}] train blocks={len(train_ds)} val blocks={len(val_ds)} (block={args.block})")

    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32)
    steps_per_epoch = math.ceil(len(train_ds) / args.batch)
    eval_every = max(1, steps_per_epoch // 4)
    run_dir = Path(args.work_dir) / speaker
    targs = TrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=max(1, int(0.1 * steps_per_epoch * args.epochs)),
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_available(),
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=eval_every,
        save_strategy="steps",
        save_steps=eval_every,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=default_data_collator)
    trainer.train()
    final = trainer.evaluate()

    out_dir = Path(args.model_dir) / f"{speaker}_gpt2"
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.to(torch.bfloat16).save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    meta = {"speaker": speaker, "base_model": args.base, "epochs": args.epochs, "block_size": args.block,
            "batch_size": args.batch, "learning_rate": args.lr, "train_blocks": len(train_ds),
            "val_blocks": len(val_ds), "best_val_loss": final["eval_loss"],
            "best_val_ppl_per_bpe_token": math.exp(final["eval_loss"]),
            "train_seconds": round(time.time() - t0, 1),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    (out_dir / "catbot_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    history = trainer.state.log_history
    (reports / f"{speaker}_gpt2_loss.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    plot_history(history, speaker, args.base, reports / f"{speaker}_gpt2_loss.png")
    print(f"[{speaker}] best val loss {final['eval_loss']:.3f} -> {out_dir}  ({meta['train_seconds']}s)")
    del trainer, model
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("speakers", nargs="*", help="default: every data/speakers/*.txt")
    ap.add_argument("--base", default="gpt2-medium", help="gpt2-medium | gpt2 | distilgpt2 ...")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-dir", default=ROOT / "model")
    ap.add_argument("--reports-dir", default=ROOT / "reports")
    ap.add_argument("--work-dir", default=ROOT / "gpt2_runs", help="Trainer checkpoints (temporary)")
    args = ap.parse_args()
    for sp in args.speakers or speakers():
        train_one(sp, args)


if __name__ == "__main__":
    main()
