"""Step 2: clean raw transcripts.

  data/raw/<speaker>/*.txt  ->  data/speakers/<speaker>.txt   (one cleaned document per line)
  data/raw/*.txt            ->  data/clean/*.txt              (legacy single-speaker layout)
"""
import argparse
import html
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STAGE_WORDS = (r"applause|laughter|laughs|cheers|cheering|music|audience|interruption|"
               r"inaudible|pause|crosstalk|slogans|video|chanting|clapping|translation")
RE_TAGS = re.compile(r"<[^>]+>")
RE_URLS = re.compile(r"https?://\S+|www\.\S+")
RE_SQUARE = re.compile(r"\[[^\]]{0,100}\]")                       # [applause], [inaudible]
RE_PAREN_STAGE = re.compile(rf"\((?=[^)]*(?:{STAGE_WORDS}))[^)]{{0,100}}\)", re.I)
MONTHS = r"(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
RE_DATES = re.compile(
    rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTHS}\.?,?\s+\d{{4}}\b"   # 12 March 2019
    rf"|\b{MONTHS}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b"  # March 12, 2019
    r"|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"                         # 12/03/2019
    r"|\b\d{4}-\d{2}-\d{2}\b",                                    # 2019-03-12
    re.I,
)
QUOTES = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
          "\u2013": "-", "\u2014": " - ", "\u2026": "..."}


def read_text(path: Path) -> str:
    """Fix encoding problems: try UTF-8 first, then Windows-1252, then Latin-1."""
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = RE_TAGS.sub(" ", text)
    text = unicodedata.normalize("NFKC", text)
    for bad, good in QUOTES.items():
        text = text.replace(bad, good)
    text = RE_URLS.sub(" ", text)
    text = RE_SQUARE.sub(" ", text)
    text = RE_PAREN_STAGE.sub(" ", text)
    text = RE_DATES.sub(" ", text)
    text = re.sub(r"_([^_]+)_", lambda m: m.group(1), text)  # Gutenberg _italics_
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")
    return re.sub(r"\s+", " ", text).strip()


def clean_speakers(raw_dir: Path, out_dir: Path) -> dict:
    """Merge each data/raw/<speaker>/ folder into data/speakers/<speaker>.txt."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for folder in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        docs = [clean_text(read_text(f)) for f in sorted(folder.glob("*.txt"))]
        docs = [d for d in docs if d]
        if not docs:
            continue
        (out_dir / f"{folder.name}.txt").write_text("\n".join(docs) + "\n", encoding="utf-8")
        counts[folder.name] = sum(len(d.split()) for d in docs)
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=ROOT / "data" / "raw")
    ap.add_argument("--out", default=ROOT / "data" / "clean", help="output for loose .txt files in --raw")
    ap.add_argument("--speakers-out", default=ROOT / "data" / "speakers")
    args = ap.parse_args()

    raw_dir, out_dir = Path(args.raw), Path(args.out)
    counts = clean_speakers(raw_dir, Path(args.speakers_out))
    for name, words in counts.items():
        flag = "" if words >= 20_000 else "  WARNING: under 20,000 words"
        print(f"{name:10s} {words:8,d} words -> {Path(args.speakers_out) / (name + '.txt')}{flag}")

    files = sorted(raw_dir.glob("*.txt"))
    if not files and not counts:
        raise SystemExit(f"No .txt files found in {raw_dir}. Run fetch_data.py or add transcripts first.")
    if files:
        out_dir.mkdir(parents=True, exist_ok=True)
        total = 0
        for f in files:
            cleaned = clean_text(read_text(f))
            (out_dir / f.name).write_text(cleaned, encoding="utf-8")
            total += len(cleaned.split())
        print(f"Cleaned {len(files)} loose file(s), about {total:,} words -> {out_dir}")


if __name__ == "__main__":
    main()
