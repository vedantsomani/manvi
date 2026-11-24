"""Speaker corpora and the train/validation split shared by every model and by evaluate.py."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPEAKER_DIR = ROOT / "data" / "speakers"
CHUNK_WORDS = 200   # split unit
VAL_EVERY = 10      # every 10th chunk -> validation (10%)

DISPLAY = {"gandhi": "Mahatma Gandhi", "lincoln": "Abraham Lincoln", "obama": "Barack Obama"}


def speakers(speaker_dir=SPEAKER_DIR):
    return sorted(p.stem for p in Path(speaker_dir).glob("*.txt"))


def display_name(speaker):
    return DISPLAY.get(speaker, speaker.replace("_", " ").title())


def load_documents(speaker, speaker_dir=SPEAKER_DIR):
    """One string per source document (one line each in data/speakers/<speaker>.txt)."""
    text = (Path(speaker_dir) / f"{speaker}.txt").read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def chunks(doc, size=CHUNK_WORDS):
    words = doc.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def train_val_split(speaker, speaker_dir=SPEAKER_DIR):
    """Return (train_docs, val_chunks).

    Each document is cut into 200-word chunks and every 10th chunk overall is held out.
    The train side keeps the surviving chunks of a document together as runs of text,
    so a held-out chunk never leaks into training and n-grams do not bridge the gap.
    """
    train, val, k = [], [], 0
    for doc in load_documents(speaker, speaker_dir):
        run = []
        for c in chunks(doc):
            if k % VAL_EVERY == VAL_EVERY - 1:
                val.append(c)
                if run:
                    train.append(" ".join(run))
                run = []
            else:
                run.append(c)
            k += 1
        if run:
            train.append(" ".join(run))
    return train, val
