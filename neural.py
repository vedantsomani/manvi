"""Optional fine-tuned GPT-2 models. Every function degrades gracefully when torch,
transformers or the weights are missing, so the app still runs Markov-only."""
import math
from pathlib import Path

from corpus import ROOT

MODEL_DIR = ROOT / "model"


def gpt2_path(speaker):
    return MODEL_DIR / f"{speaker}_gpt2"


def available(speaker):
    p = gpt2_path(speaker)
    if not (p / "config.json").exists():
        return False
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def load(speaker):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(gpt2_path(speaker))
    model = AutoModelForCausalLM.from_pretrained(gpt2_path(speaker), dtype=dtype).to(device).eval()
    return model, tok


def generate(model, tok, prompt, max_new_tokens=160, temperature=1.0, top_p=0.95, seed=0):
    import torch
    torch.manual_seed(seed)
    ids = tok(prompt, return_tensors="pt").input_ids if prompt.strip() else \
        torch.tensor([[tok.eos_token_id]])
    ids = ids.to(model.device)
    with torch.no_grad():
        out = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=temperature > 0,
                             temperature=max(temperature, 1e-5), top_p=top_p,
                             max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0], skip_special_tokens=True)
    return text


def total_nll(model, tok, text):
    """Sum of -log p over the BPE tokens of `text`, conditioning the first on <|endoftext|>."""
    import torch
    ids = [tok.eos_token_id] + tok(text).input_ids
    ids = ids[:1024]
    x = torch.tensor([ids], device=model.device)
    with torch.no_grad():
        logits = model(x).logits.float()
    logp = torch.log_softmax(logits[0, :-1], dim=-1)
    return -logp.gather(1, x[0, 1:].unsqueeze(1)).sum().item()


def perplexity_per_word_token(model, tok, texts, count_tokens):
    """exp(total NLL / number of *Markov* tokens), so it is comparable with the n-gram models."""
    nll = sum(total_nll(model, tok, t) for t in texts)
    n = sum(count_tokens(t) for t in texts)
    return math.exp(nll / max(n, 1))
