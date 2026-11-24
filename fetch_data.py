"""Download public-domain speech transcripts for each speaker into data/raw/<speaker>/.

Sources (see DATA.md for details and licences):
  gandhi  - "Freedom's Battle" (1922), Project Gutenberg #10366
  lincoln - "Noted Speeches of Abraham Lincoln", Gutenberg #48364, plus three speeches
            from "The Papers and Writings of Abraham Lincoln", Gutenberg #3253
  obama   - 14 speeches from the archived Obama White House site (US government works)
  modi    - English texts of PM Narendra Modi's speeches from pmindia.gov.in
  trump   - remarks from the archived Trump White House site (2017-2021)

Modi and Trump URLs are frozen in data/sources/<speaker>.txt so the dataset is
reproducible. `python fetch_data.py --discover modi trump` rebuilds those lists.

Run:  python fetch_data.py modi trump obama   then   python clean.py
"""
import argparse
import html
import re
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
# www.gutenberg.org rate-limits scripts; the PGLAF mirror serves the same files.
GUTENBERG = "https://gutenberg.pglaf.org"
OBAMA = "https://obamawhitehouse.archives.gov/the-press-office/"
UA = {"User-Agent": "CATbot-student-project/1.0 (educational text dataset)"}


def get(url, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8", errors="replace").replace("\r", "")
        except Exception as e:  # network hiccup: retry with backoff
            if i == tries - 1:
                raise RuntimeError(f"Could not download {url}: {e}") from e
            time.sleep(2 * (i + 1))


def gutenberg(book_id):
    s = str(book_id)
    folder = "/".join(s[:-1]) + f"/{s}"
    for name in (f"{s}-0.txt", f"{s}.txt"):
        try:
            return get(f"{GUTENBERG}/{folder}/{name}")
        except RuntimeError:
            continue
    raise RuntimeError(f"Gutenberg book {book_id} not found on mirror")


def between(text, start, end, include_start=True):
    """Text from the line containing `start` up to (not including) `end`."""
    i = text.index(start)
    j = text.index(end, i + len(start))
    return text[i if include_start else i + len(start):j].strip()


# ---------------------------------------------------------------- Gandhi
def fetch_gandhi():
    book = gutenberg(10366)
    # Skip the editor's introduction (by C. Rajagopalachari); keep Gandhi's own writings.
    body = between(book, "II. THE KHILAFAT\n\n\nWHY I HAVE JOINED", "*** END OF THE PROJECT GUTENBERG")
    body = body.replace("II. THE KHILAFAT\n", "", 1)
    return {"freedoms_battle": body}


# ---------------------------------------------------------------- Lincoln
def fetch_lincoln():
    noted = gutenberg(48364)
    papers = gutenberg(3253)
    parts = {
        "cooper_institute": between(noted, "MR. PRESIDENT AND FELLOW-CITIZENS OF NEW YORK",
                                    "\nLINCOLN’S FIRST INAUGURAL ADDRESS"),
        "first_inaugural": between(noted, "FELLOW-CITIZENS OF THE UNITED STATES", "\nLINCOLN’S GETTYSBURG SPEECH"),
        "gettysburg": between(noted, "Fourscore and seven years ago", "\nLINCOLN’S SECOND INAUGURAL ADDRESS"),
        "second_inaugural": between(noted, "FELLOW-COUNTRYMEN:", "\nPROCLAMATION OF EMANCIPATION"),
        "ottawa_debate_reply": between(noted, "MY FELLOW-CITIZENS:--When a man hears himself", "*      *      *"),
        "lyceum_address": between(papers, "As a subject for the remarks of the evening",
                                  "PROTEST IN THE ILLINOIS LEGISLATURE"),
        "peoria_speech": between(papers, "I do not rise to speak now", "REQUEST FOR SENATE SUPPORT"),
        "farewell_springfield": between(papers, "MY FRIENDS:--One who has never been placed",
                                        "REMARKS AT TOLONO"),
    }
    # Editorial narration inside the Peoria text, not Lincoln's words.
    parts["peoria_speech"] = re.sub(r"The audience signified their assent.*?as follows:\n", "",
                                    parts["peoria_speech"], flags=re.S)
    # Drop the trailing all-caps heading line that precedes each `end` marker, if any.
    return {k: re.sub(r"\n[A-Z’' ,.:-]{6,}\n*$", "", v) for k, v in parts.items()}


# ---------------------------------------------------------------- Obama
OBAMA_SPEECHES = [
    "president-barack-obamas-inaugural-address",                                  # 2009 inaugural
    "remarks-president-cairo-university-6-04-09",                                 # 2009 Cairo
    "2011/01/12/remarks-president-barack-obama-memorial-service-victims-shooting-tucson",
    "2011/01/25/remarks-president-state-union-address",
    "2012/01/24/remarks-president-state-union-address",
    "2013/01/21/inaugural-address-president-barack-obama",
    "2013/02/12/remarks-president-state-union-address",
    "2014/01/28/president-barack-obamas-state-union-address",
    "2014/03/26/remarks-president-address-european-youth",
    "2015/01/20/remarks-president-state-union-address-january-20-2015",
    "2015/03/07/remarks-president-50th-anniversary-selma-montgomery-marches",
    "2015/06/26/remarks-president-eulogy-honorable-reverend-clementa-pinckney",
    "2016/03/22/remarks-president-obama-people-cuba",
    "2017/01/10/remarks-president-farewell-address",
]
RE_TIME = re.compile(r"^\s*(END\s+)?\d{1,2}:\d{2}\s*[AP]\.?M\.?.*$", re.I)
RE_LABEL = re.compile(r"^\s*([A-Z][A-Z .'-]{2,40}):\s*")
# Page furniture that sits inside a few transcript bodies.
RE_JUNK = re.compile(r"Flash Player|download as high-quality|FOR IMMEDIATE RELEASE|^\(?[A-Z][a-z]+, ?[A-Z][a-z]+\)?$")
RE_PLACE_PREFIX = re.compile(r"^Selma, Alabama\s+")


def obama_page_text(page):
    start = page.index("field-name-field-forall-body")
    ends = [i for i in (page.find("<footer", start), page.find("</article>", start)) if i > 0] or [len(page)]
    body = page[start:min(ends)]
    body = re.sub(r"<script.*?</script>|<style.*?</style>", " ", body, flags=re.S | re.I)
    # Venue/location lines are centred; pages use either <p> or <div> paragraphs.
    body = re.sub(r'<(p|div)[^>]*class="rtecenter"[^>]*>.*?</(?:p|div)>', " ", body, flags=re.S)
    paras = []
    for block in re.split(r"</p>|</div>|<br\s*/?>", body):
        text = html.unescape(re.sub(r"<[^>]+>", " ", block)).replace("\xa0", " ")
        text = re.sub(r"\s+", " ", text).strip()
        if text in ("END", "###") or text.startswith(("Share This", "Privacy Policy", "Photo Gallery")):
            break  # end of the transcript; the rest is page furniture
        if (not text or RE_TIME.match(text) or "forall-body" in text or "Press Secretary" in text
                or set(text) <= set("_- ") or not re.search(r"[.!?:;,]", text)):
            continue  # headers without sentence punctuation
        if RE_JUNK.search(text):
            continue  # e.g. "(Cairo,Egypt)", "FOR IMMEDIATE RELEASE ..."
        text = RE_PLACE_PREFIX.sub("", text)
        m = RE_LABEL.match(text)
        if m:
            if m.group(1).strip() != "THE PRESIDENT":
                continue  # audience member / introducer, not Obama
            text = text[m.end():]
        paras.append(text)
    return "\n\n".join(paras)


def fetch_obama():
    out = {}
    for slug in OBAMA_SPEECHES:
        name = slug.rstrip("/").split("/")[-1][:60]
        date = "_".join(slug.split("/")[:3]) if slug[:4].isdigit() else "2009"
        out[f"{date}_{name}"] = obama_page_text(get(OBAMA + slug))
        time.sleep(0.5)  # be polite to the archive
    return out


# ---------------------------------------------------------------- shared helpers
SOURCES = ROOT / "data" / "sources"
SPEAKER_LABEL = re.compile(r"^\s*([A-Z][A-Z .'-]{1,40}|Q):\s*")


def paragraphs(fragment):
    """Plain-text paragraphs from an HTML fragment."""
    fragment = re.sub(r"<script.*?</script>|<style.*?</style>", " ", fragment, flags=re.S | re.I)
    out = []
    for block in re.split(r"</p>|</div>|<br\s*/?>", fragment):
        text = html.unescape(re.sub(r"<[^>]+>", " ", block)).replace("\xa0", " ")
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            out.append(text)
    return out


def only_speaker(paras, label="THE PRESIDENT"):
    """Keep paragraphs spoken by `label`. A labelled paragraph switches speaker;
    unlabelled paragraphs belong to whoever spoke last (start: the main speaker)."""
    current, keep = label, []
    for t in paras:
        if re.match(r"^Q[\s:]", t):
            continue  # a reporter's question interrupts but does not change the main speaker
        m = SPEAKER_LABEL.match(t)
        if m:
            current = m.group(1).strip()
            t = t[m.end():]
        if current == label and t:
            keep.append(t)
    return keep


def latin_ratio(text):
    letters = [c for c in text if c.isalpha()]
    return sum(c.isascii() for c in letters) / max(1, len(letters))


def read_sources(speaker):
    path = SOURCES / f"{speaker}.txt"
    if not path.exists():
        raise SystemExit(f"{path} missing. Run: python fetch_data.py --discover {speaker}")
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")]


# ---------------------------------------------------------------- Modi
MODI_BASE = "https://www.pmindia.gov.in/en/news_updates/"
MODI_LIST = "https://www.pmindia.gov.in/en/tag/pmspeech/page/{}/"


def modi_page_text(page):
    i = page.find("class='news-bg'")
    j = page.find('class="tweet-container"', i)
    body = page[i:j if j > 0 else None]
    paras = [t for t in paragraphs(body) if latin_ratio(t) > 0.9]  # drop Hindi/other-script lines
    # Drop the doctype debris some pages carry and one-word salutations without punctuation.
    paras = [t for t in paras if not t.startswith(("<!DOCTYPE", "DOCTYPE")) and "W3C" not in t]
    return "\n\n".join(paras)


def discover_modi(pages=8, min_words=300):
    slugs = []
    for n in range(1, pages + 1):
        listing = get(MODI_LIST.format(n))
        for slug in re.findall(r'news_updates/([a-z0-9-]+)/', listing):
            if slug not in slugs:
                slugs.append(slug)
        time.sleep(0.3)
    keep = []
    for slug in slugs:
        text = modi_page_text(get(MODI_BASE + slug + "/"))
        if len(text.split()) >= min_words:  # English text of a real speech
            keep.append(slug)
        time.sleep(0.3)
    return keep


def fetch_modi():
    out = {}
    for slug in read_sources("modi"):
        out[slug[:80]] = modi_page_text(get(MODI_BASE + slug + "/"))
        time.sleep(0.3)
    return out


# ---------------------------------------------------------------- Trump
TRUMP_BASE = "https://trumpwhitehouse.archives.gov/briefings-statements/"
TRUMP_LIST = "https://trumpwhitehouse.archives.gov/remarks/page/{}/"
# Formats that are mostly Q&A with reporters or other people talking.
TRUMP_SKIP = re.compile(r"departure|roundtable|bilateral|meeting|call|gaggle|briefing|signing|"
                        r"press-conference|exchange|before-|discussion|listening|tour|q-a|interview|town-hall")


def trump_page_text(page):
    i = page.find("page-content__content")
    body = page[i:page.find("</article>", i) if "</article>" in page[i:] else None]
    paras = paragraphs(body[body.find(">") + 1:])
    if "All News" in paras:  # share buttons and breadcrumbs come before the transcript
        paras = paras[paras.index("All News") + 1:]
    out = []
    for t in paras:
        if t in ("END", "###") or t.startswith("END "):
            break
        if t in ("The White House",) or t.startswith("Share on ") or out[-1:] == [t] \
                or re.match(r"^\d{1,2}:\d{2}(\s*[AP]\.?M\.?)?\s", t):
            continue
        if len(t.split()) <= 6 and not re.search(r"[.!?]$", t) and not SPEAKER_LABEL.match(t):
            continue  # venue / city header lines such as "Prescott Regional Airport"
        out.append(t)
    return "\n\n".join(only_speaker(out))


def discover_trump(pages=40, min_words=800, target_words=80_000):
    slugs = []
    for n in range(1, pages + 1):
        for slug in re.findall(r"briefings-statements/(remarks-president-trump-[a-z0-9-]+)/", get(TRUMP_LIST.format(n))):
            if slug not in slugs and not TRUMP_SKIP.search(slug):
                slugs.append(slug)
        time.sleep(0.3)
    keep, total = [], 0
    for slug in slugs:
        try:
            page = get(TRUMP_BASE + slug + "/")
        except RuntimeError:  # a few archived links are dead
            continue
        words = len(trump_page_text(page).split())
        if words >= min_words:
            keep.append(slug)
            total += words
            if total >= target_words:  # keep speakers roughly the same size
                break
        time.sleep(0.3)
    return keep


def fetch_trump():
    out = {}
    for slug in read_sources("trump"):
        out[slug.replace("remarks-president-trump-", "")[:80]] = trump_page_text(get(TRUMP_BASE + slug + "/"))
        time.sleep(0.3)
    return out


SPEAKERS = {"gandhi": fetch_gandhi, "lincoln": fetch_lincoln, "obama": fetch_obama,
            "modi": fetch_modi, "trump": fetch_trump}
DISCOVER = {"modi": discover_modi, "trump": discover_trump}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("speakers", nargs="*", default=["modi", "trump", "obama"],
                    help="any of: " + ", ".join(SPEAKERS))
    ap.add_argument("--discover", action="store_true",
                    help="rebuild data/sources/<speaker>.txt URL lists from the live sites")
    args = ap.parse_args()
    if args.discover:
        SOURCES.mkdir(parents=True, exist_ok=True)
        for sp in args.speakers:
            slugs = DISCOVER[sp]()
            (SOURCES / f"{sp}.txt").write_text("\n".join(slugs) + "\n", encoding="utf-8")
            print(f"{sp}: {len(slugs)} speeches -> {SOURCES / (sp + '.txt')}")
        return
    for sp in args.speakers:
        folder = RAW / sp
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("*.txt"):
            old.unlink()  # this folder is fully regenerated from the sources
        docs = SPEAKERS[sp]()
        words = 0
        for name, text in docs.items():
            if text.strip():
                (folder / f"{name}.txt").write_text(text + "\n", encoding="utf-8")
                words += len(text.split())
        print(f"{sp:8s} {len(docs):3d} document(s) {words:8,d} words -> {folder}")


if __name__ == "__main__":
    main()
