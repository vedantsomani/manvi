"""Clean transcripts, train the Markov model and export it as JSON."""
from pathlib import Path
from clean import clean_text, read_text
from model import build_model, save_json, tokenize

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
OUT = ROOT / "model" / "catbot_model.json"
ORDER = 2

def main():
    files = sorted(RAW.glob("*.txt"))
    files = [p for p in files if p.name != "PUT_TRANSCRIPTS_HERE.txt"]
    if not files:
        raise SystemExit("No transcript .txt files found in data/raw. Add at least one transcript first.")
    CLEAN.mkdir(parents=True, exist_ok=True)
    docs = []
    total_words = 0
    for path in files:
        cleaned = clean_text(read_text(path))
        (CLEAN / path.name).write_text(cleaned, encoding="utf-8")
        tokens = tokenize(cleaned)
        if len(tokens) > ORDER:
            docs.append(tokens)
        total_words += len(cleaned.split())
    if not docs:
        raise SystemExit("Transcript files contain too little usable text.")
    transitions, starts = build_model(docs, ORDER)
    save_json(transitions, starts, OUT, ORDER, source_files=[p.name for p in files],
              word_count=total_words, token_count=sum(map(len, docs)))
    print(f"Trained Markov model from {len(files)} transcript(s)")
    print(f"Words: {total_words:,} | Tokens: {sum(map(len, docs)):,} | States: {len(transitions):,}")
    print(f"Saved model: {OUT}")
    if total_words < 20000:
        print("WARNING: Dataset is small; output may repeat training text. Add more transcripts if possible.")

if __name__ == "__main__":
    main()
