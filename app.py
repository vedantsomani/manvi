"""CATbot: a copy-cat speech generator (Streamlit app).   Run:  streamlit run app.py"""
import html
import json
import random
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

import neural
from copycheck import WINDOW, build_index, copied_mask, copy_rate, longest_copied_run
from corpus import ROOT, display_name, speakers, train_val_split
from model import PUNCT, SENTENCE_END, MarkovModel, detokenize, tokenize

MODEL_DIR = ROOT / "model"
REPORTS = ROOT / "reports"
SPEAKER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
LABEL = "AI-generated text. This is not a real quote from the speaker."

st.set_page_config(page_title="CATbot: copy-cat speech generator", page_icon="🐱", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.cat-out {font-size: 1.08rem; line-height: 1.75; padding: 1.1rem 1.3rem; border-radius: 10px;
          border: 1px solid rgba(128,128,128,.25); background: rgba(128,128,128,.06);}
.cat-out mark {background: rgba(235,104,52,.25); color: inherit; padding: 0 .1em; border-radius: 3px;
               border-bottom: 2px solid #eb6834;}
.cat-out .prompt {font-weight: 600;}
.cat-label {font-size: .8rem; opacity: .7; margin-top: .4rem;}
.cat-footer {font-size: .8rem; opacity: .65; text-align: center; margin-top: 3rem;
             border-top: 1px solid rgba(128,128,128,.2); padding-top: 1rem;}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- cached resources
@st.cache_resource(show_spinner="Loading Markov model ...")
def markov(speaker):
    path = MODEL_DIR / f"{speaker}_markov.json"
    if path.exists():
        return MarkovModel.from_json(path)
    # Fresh clone without model files: train in a second from data/speakers.
    return MarkovModel().fit(train_val_split(speaker)[0])


@st.cache_resource(show_spinner="Indexing training text for the copy check ...")
def copy_index(speaker):
    return build_index(train_val_split(speaker)[0])


@st.cache_resource(show_spinner="Loading fine-tuned GPT-2 (first time only) ...")
def gpt2(speaker):
    return neural.load(speaker)


@st.cache_data
def gpt2_ok(speaker):
    return neural.available(speaker)


SPEAKERS = speakers()
if not SPEAKERS:
    st.error("No speaker data found in data/speakers/. Run `python fetch_data.py && python clean.py`.")
    st.stop()
COLOR = {sp: SPEAKER_COLORS[i % 3] for i, sp in enumerate(SPEAKERS)}


# ---------------------------------------------------------------- helpers
def render(tokens, mask, n_prompt=0):
    """HTML for tokens with copied stretches in <mark> and the prompt in bold."""
    parts, cap = [], True
    for i, (tok, copied) in enumerate(zip(tokens, mask)):
        if tok not in PUNCT and cap:
            tok = tok[:1].upper() + tok[1:]
        if tok not in PUNCT:
            cap = False
        else:
            cap = tok in SENTENCE_END
        piece = html.escape(tok)
        if i < n_prompt:
            piece = f'<span class="prompt">{piece}</span>'
        elif copied:
            piece = f"<mark>{piece}</mark>"
        sep = "" if (tok in PUNCT or i == 0) else " "
        parts.append(sep + piece)
    out = "".join(parts).replace("</mark> <mark>", " ")
    return f'<div class="cat-out">{out}</div><div class="cat-label">🤖 {LABEL}</div>'


def copy_metrics(tokens, speaker, n_prompt=0):
    index = copy_index(speaker)
    body = tokens[n_prompt:] if len(tokens) - n_prompt >= WINDOW else tokens
    return copy_rate(body, index), longest_copied_run(body, index), copied_mask(tokens, index)


def run_markov(speaker, start, length, order, temp, seed):
    m = markov(speaker)
    tokens, trace = m.generate(start, length, order, temp, seed)
    n_prompt = sum(t is None for t in trace)
    return {"kind": "markov", "tokens": tokens, "trace": trace, "n_prompt": n_prompt}


def run_gpt2(speaker, prompt, length, temp, seed):
    model, tok = gpt2(speaker)
    text = neural.generate(model, tok, prompt, max_new_tokens=int(length * 1.5), temperature=temp, seed=seed)
    tokens = tokenize(text)
    n_prompt = len(tokenize(prompt))
    # Trim to the requested length, ending on a sentence boundary when possible.
    cut = tokens[:length + 30]
    ends = [i for i, t in enumerate(cut) if t in SENTENCE_END and i >= length * 0.7]
    tokens = cut[:ends[0] + 1] if ends else cut[:length]
    return {"kind": "gpt2", "tokens": tokens, "trace": None, "n_prompt": n_prompt}


def show_output(res, speaker, title):
    rate, longest, mask = copy_metrics(res["tokens"], speaker, res["n_prompt"])
    st.markdown(f"#### {title}")
    st.markdown(render(res["tokens"], mask, res["n_prompt"]), unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("% copied", f"{rate:.0f}%", help=f"Share of {WINDOW}-word windows found word-for-word "
                                               "in the training speeches (highlighted).")
    c2.metric("Longest copied run", f"{longest} words")
    if res["trace"]:
        steps = [t for t in res["trace"] if t]
        backoff = sum(t["order_used"] < st.session_state.get("order_used_req", 0) for t in steps)
        c3.metric("Backoff steps", f"{backoff}/{len(steps)}",
                  help="Steps where the full context was never seen, so the model used fewer words.")
    else:
        c3.metric("Tokens", len(res["tokens"]))
    st.download_button("Download text", detokenize(res["tokens"]) + f"\n\n[{LABEL}]",
                       file_name=f"catbot_{speaker}_{res['kind']}.txt", key=f"dl_{res['kind']}")


def why_panel(res, speaker):
    steps = [(i, t) for i, t in enumerate(res["trace"]) if t]
    if not steps:
        return
    st.markdown("#### 🔍 Why this word?")
    st.caption("Pick a generated word to see what the Markov chain could have chosen at that step and "
               "how likely each option was after temperature.")
    labels = [f"{k + 1}. {t['chosen']}" for k, (_, t) in enumerate(steps)]
    pick = st.select_slider("Step", options=range(len(steps)), format_func=lambda k: labels[k],
                            value=min(5, len(steps) - 1), label_visibility="collapsed")
    i, t = steps[pick]
    ctx = " ".join(t["context"]) if t["context"] else "(nothing: plain word frequency)"
    req = st.session_state.get("order_used_req", t["order_used"])
    note = "" if t["order_used"] >= req else \
        f" ⚠️ backed off from order {req}: that {req}-word context never appears in the training text."
    st.markdown(f"Context used (order {t['order_used']}): **{html.escape(ctx)}** → chose "
                f"**{html.escape(t['chosen'])}** (p = {t['p']:.2f}, {t['n_options']} possible next "
                f"tokens).{note}")
    top = t["top"][::-1]
    words = [w for w, _ in top]
    fig = go.Figure(go.Bar(
        x=[p for _, p in top], y=words, orientation="h",
        marker_color=[COLOR[speaker] if w == t["chosen"] else "#b9b8b3" for w in words],
        text=[f"{p:.0%}" for _, p in top], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig.update_layout(height=40 + 30 * len(top), margin=dict(l=10, r=40, t=10, b=10),
                      xaxis=dict(showgrid=False, visible=False), yaxis=dict(tickfont=dict(size=14)),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    speaker = st.selectbox("Speaker", SPEAKERS, format_func=display_name)
    has_gpt2 = gpt2_ok(speaker)
    modes = ["Markov chain", "GPT-2 (fine-tuned)", "Side-by-side"] if has_gpt2 else ["Markov chain"]
    mode = st.radio("Model", modes, horizontal=False)
    if not has_gpt2:
        st.caption("GPT-2 weights for this speaker aren't installed, so only the Markov chain is available. "
                   "See the README to add them.")
    order = st.slider("n-gram order (Markov)", 1, 4, 2,
                      help="How many previous words the Markov chain looks at. Higher = more fluent, "
                           "but more copying.")
    temp = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1,
                     help="Below 1 = safer, more predictable words. Above 1 = more surprising choices.")
    length = st.slider("Length (tokens)", 30, 300, 100, 10)
    seed = st.number_input("Random seed", 0, 999_999, 42, help="Same seed + settings = same text.")
    start = st.text_input("Start phrase (optional)", placeholder="e.g. my fellow citizens")
    go_btn = st.button("✨ Generate", type="primary", width="stretch")
    with st.expander("Model details"):
        m = markov(speaker)
        st.write(f"**Training tokens:** {m.n_tokens:,}")
        st.write(f"**Vocabulary:** {m.vocab_size:,}")
        st.write(f"**Order-{m.max_order} contexts:** {len(m.counts[m.max_order]):,}")
        if has_gpt2:
            meta_path = neural.gpt2_path(speaker) / "catbot_meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                st.write(f"**GPT-2:** {meta['base_model']}, {meta['epochs']:g} epochs on {meta['gpu']}")


# ---------------------------------------------------------------- header
st.title("🐱 CATbot")
st.markdown("**A copy-cat speech generator.** CATbot learns how Mahatma Gandhi, Abraham Lincoln and "
            "Barack Obama put words together, then writes new text in their style. It compares a "
            "classic **n-gram Markov chain** with a **fine-tuned GPT-2** and checks how much of the "
            "output is copied word-for-word from the real speeches "
            "(<mark style='background:rgba(235,104,52,.25)'>highlighted</mark>).",
            unsafe_allow_html=True)

tab_gen, tab_game, tab_res, tab_how = st.tabs(["✍️ Generate", "🎯 Guess the speaker", "📊 Results",
                                               "💡 How it works"])

# ---------------------------------------------------------------- generate tab
with tab_gen:
    key = (speaker, mode, order, temp, length, seed, start)
    if go_btn or "gen" not in st.session_state:
        try:
            out = {}
            st.session_state["order_used_req"] = order
            if mode in ("Markov chain", "Side-by-side"):
                out["markov"] = run_markov(speaker, start, length, order, temp, int(seed))
            if mode in ("GPT-2 (fine-tuned)", "Side-by-side"):
                prompt = start.strip() or detokenize(
                    list(random.Random(int(seed)).choice(markov(speaker).starts))[:2])
                with st.spinner("GPT-2 is writing ..."):
                    out["gpt2"] = run_gpt2(speaker, prompt, length, temp, int(seed))
            st.session_state["gen"] = (key, out)
        except Exception as e:  # show the problem in the page, never crash the app
            st.error(f"Could not generate: {e}")
            st.session_state.pop("gen", None)
    if "gen" in st.session_state:
        (k_speaker, k_mode, *_), out = st.session_state["gen"]
        if k_speaker != speaker or k_mode != mode or st.session_state["gen"][0] != key:
            st.info("Settings changed. Press **✨ Generate** to update.")
        if "markov" in out and "gpt2" in out:
            left, right = st.columns(2)
            with left:
                show_output(out["markov"], k_speaker, f"Markov chain (order {st.session_state['order_used_req']})")
            with right:
                show_output(out["gpt2"], k_speaker, "Fine-tuned GPT-2")
            why_panel(out["markov"], k_speaker)
        elif "markov" in out:
            show_output(out["markov"], k_speaker,
                        f"{display_name(k_speaker)}, Markov chain (order {st.session_state['order_used_req']})")
            why_panel(out["markov"], k_speaker)
        else:
            show_output(out["gpt2"], k_speaker, f"{display_name(k_speaker)}, fine-tuned GPT-2")
            st.caption("The *Why this word?* panel is available for the Markov chain.")

# ---------------------------------------------------------------- game tab
with tab_game:
    st.markdown("CATbot writes a short passage as one of the three speakers. **Can you tell who?** "
                "A simple computer judge plays along: it picks the speaker whose Markov model finds "
                "the passage least surprising (lowest perplexity).")
    g = st.session_state.setdefault("game", {"score": 0, "rounds": 0, "judge": 0, "round": None})

    def new_round():
        rng = random.Random()
        sp = rng.choice(SPEAKERS)
        use_gpt2 = gpt2_ok(sp) and rng.random() < 0.5
        s = rng.randrange(1_000_000)
        if use_gpt2:
            prompt = detokenize(list(random.Random(s).choice(markov(sp).starts))[:2])
            res = run_gpt2(sp, prompt, 60, 0.9, s)
        else:
            res = run_markov(sp, "", 60, 2, 0.9, s)
        toks = res["tokens"]
        scores = {c: markov(c).perplexity([toks], 2)[0] for c in SPEAKERS}
        g["round"] = {"speaker": sp, "tokens": toks, "model": "GPT-2" if use_gpt2 else "Markov order 2",
                      "judge": min(scores, key=scores.get), "scores": scores, "guess": None}

    if g["round"] is None:
        new_round()
    r = g["round"]
    st.markdown(f'<div class="cat-out">{html.escape(detokenize(r["tokens"]))}</div>', unsafe_allow_html=True)
    if r["guess"] is None:
        cols = st.columns(len(SPEAKERS))
        for col, sp in zip(cols, SPEAKERS):
            if col.button(display_name(sp), key=f"guess_{sp}", width="stretch"):
                r["guess"] = sp
                g["rounds"] += 1
                g["score"] += sp == r["speaker"]
                g["judge"] += r["judge"] == r["speaker"]
                st.rerun()
    else:
        if r["guess"] == r["speaker"]:
            st.success(f"✅ Correct! It was **{display_name(r['speaker'])}** ({r['model']}).")
        else:
            st.error(f"❌ It was **{display_name(r['speaker'])}** ({r['model']}), "
                     f"not {display_name(r['guess'])}.")
        judge_ok = "✅" if r["judge"] == r["speaker"] else "❌"
        st.caption(f"Computer judge guessed {display_name(r['judge'])} {judge_ok}  ·  perplexities: " +
                   ", ".join(f"{display_name(s)} {v:.0f}" for s, v in r["scores"].items()))
        if st.button("Next passage ▶", type="primary"):
            new_round()
            st.rerun()
    c1, c2 = st.columns(2)
    c1.metric("Your score", f"{g['score']}/{g['rounds']}")
    c2.metric("Computer judge", f"{g['judge']}/{g['rounds']}")

# ---------------------------------------------------------------- results tab
with tab_res:
    res_path = REPORTS / "results.json"
    if not res_path.exists():
        st.info("No evaluation yet. Run `python evaluate.py` to create reports/.")
    else:
        data = json.loads(res_path.read_text(encoding="utf-8"))
        rows = [{"Speaker": display_name(sp), "Model": r["model"],
                 "Perplexity ↓": None if r["perplexity"] is None else round(r["perplexity"], 1),
                 "Distinct-1 ↑": round(r["distinct_1"], 3), "Distinct-2 ↑": round(r["distinct_2"], 3),
                 "Copy rate % ↓": round(r["copy_rate"], 1)}
                for sp in data["speakers"] for r in data["results"][sp]]
        st.markdown("Perplexity is measured on 10% of each speaker's text that no model saw during training. "
                    "Copy rate is the share of 8-word windows in generated text that appear word-for-word "
                    "in the training speeches.")
        st.dataframe(rows, width="stretch", hide_index=True)
        c1, c2 = st.columns(2)
        for col, name in zip((c1, c2, c1, c2), ("perplexity", "copy_rate", "distinct2", "cross_perplexity")):
            p = REPORTS / f"{name}.png"
            if p.exists():
                col.image(str(p), width="stretch")
        losses = sorted(REPORTS.glob("*_gpt2_loss.png"))
        if losses:
            st.markdown("#### GPT-2 fine-tuning curves")
            cols = st.columns(len(losses))
            for col, p in zip(cols, losses):
                col.image(str(p), width="stretch")

# ---------------------------------------------------------------- how it works tab
with tab_how:
    st.markdown(f"""
**1. Data.** Public-domain speeches: Gandhi's *Freedom's Battle* (1922), eight Lincoln speeches (1837–1865)
and fourteen Obama addresses (2009–2017). Between 40k and 74k words per speaker. Every 10th 200-word chunk
is held out for testing.

**2. Markov chain.** For every run of *n* words in the training text, CATbot counts which word came next.
To write, it looks at the last *n* words and picks the next word in proportion to those counts
(adjusted by **temperature**). If those *n* words never appeared together, it **backs off** to the last
*n − 1* words, and so on.

**3. GPT-2.** A 355M-parameter transformer (gpt2-medium), pre-trained on web text and then
**fine-tuned** for 3 epochs on each speaker using an NVIDIA H200 GPU.

**4. Is it just copying?** Every {WINDOW}-word window of the output is checked against the training
text. Higher Markov orders sound more fluent because they copy more: order 4 reproduces most of its
output verbatim. The *Results* tab shows this trade-off in numbers.
""")

st.markdown('<div class="cat-footer">CATbot · student project · built with Python, Streamlit, '
            'Hugging Face Transformers · Generated text is synthetic and must not be presented as a '
            'real quote. Source texts are public domain; see DATA.md.</div>', unsafe_allow_html=True)
