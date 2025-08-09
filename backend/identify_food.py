from __future__ import annotations

import concurrent.futures as cf
import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import inflect
import spacy
import torch
from PIL import Image
from transformers import GitForCausalLM, GitProcessor

import openai


os.environ.setdefault(
    "OPENAI_API_KEY",
)

import logging

LOG_LEVEL = os.getenv("STEP2_LOGLEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("step2_food")

_inflect = inflect.engine()
# keep spacy config identical (no lemmatizer, ner, parser)
_nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer", "textcat"])

# Data file lives next to this script
_DATA = Path(__file__).with_name("cross_ref_food.json")

# Captioning
_CAP_TOK: int = 16

# Types
TriggerMap = Dict[str, List[str]]
AdviceMap = Dict[str, Optional[List[str]]]

@lru_cache(maxsize=128)
def _is_food(token: str) -> bool:
    """
    Ask GPT-4o/mini to answer: “Is <token> an edible food item?”
    Returns True on “YES”, False on “NO”.
    Fails open (True) on exceptions, preserving original behavior.
    """
    prompt = (
   
        "Reply with exactly one word: YES or NO (no punctuation).\n\n"
 
        "Say YES if the term names something that can be eaten or drunk "
        "directly OR is commonly used as a culinary ingredient. "
        "This includes category words such as 'cheese', 'bread', 'pasta', "
        "'burger', 'yogurt', 'juice'.\n"
 
        "Say NO if the term is generic ('food', 'dish', 'ingredient'), "
        "a utensil, a quantity word ('bunch'), a colour, or anything non-edible.\n\n"
  
        f"Term: {token}"
    )

    try:
        resp = openai.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        out = (resp.choices[0].message.content or "").strip().upper()
        return out.startswith("Y")
    except Exception as e:
        log.debug("LLM check failed for %r (%s); failing open as True", token, e)
        return True  

@lru_cache(maxsize=1)
def _load_caption() -> Tuple[GitProcessor, GitForCausalLM]:

    proc = GitProcessor.from_pretrained("microsoft/git-large-coco")

    # try 8-bit only if bitsandbytes + CUDA present
    try:
        from transformers import BitsAndBytesConfig

        if torch.cuda.is_available():
            cfg = BitsAndBytesConfig(load_in_8bit=True)
            mdl = GitForCausalLM.from_pretrained(
                "microsoft/git-large-coco",
                device_map="auto",
                quantization_config=cfg,
                torch_dtype=torch.float16,
            )
            return proc, mdl
    except Exception:

        pass

    mdl = GitForCausalLM.from_pretrained(
        "microsoft/git-large-coco",
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    return proc, mdl

def _rgb(im: Image.Image | str | Path) -> Image.Image:
    """Ensure a PIL RGB image from path or Image; behavior identical."""
    return im.convert("RGB") if isinstance(im, Image.Image) else Image.open(im).convert("RGB")

def _sg(word: str) -> str:
    """Singularize the last token of a possibly multi-word phrase (order preserving)."""
    *head, last = word.split()
    last = _inflect.singular_noun(last) or last
    return " ".join([*head, last])

def _caption2foods(text: str) -> List[str]:
    """
    Extract candidate food tokens from a caption:
      • single NOUNs (alphabetic)
      • ADJ + NOUN bigrams
      • singularize final token
      • limit to ≤3 words
      • confirm with _is_food() via LLM (unchanged)
    """
    doc = _nlp(text.lower())
    out: set[str] = set()


    out.update(tok.text for tok in doc if tok.pos_ == "NOUN" and tok.is_alpha)

    # ADJ + NOUN bigrams
    out.update(
        f"{doc[i-1].text} {tok.text}"
        for i, tok in enumerate(doc)
        if tok.pos_ == "NOUN" and i and doc[i - 1].pos_ == "ADJ"
    )

    # ≤3 tokens, singularize tail
    cands = {_sg(w) for w in out if len(w.split()) <= 3}

    # keep only LLM-confirmed foods (preserves original behavior)
    cands = {w for w in cands if _is_food(w)}

    return sorted(cands)

def identify_food_items(img: Image.Image | str | Path) -> List[str]:
    """
    Single-image path/PIL → caption (GIT) → candidate foods (≤3 tokens).
    """
    proc, mdl = _load_caption()
    inputs = proc(images=_rgb(img), return_tensors="pt").to(mdl.device)
    ids = mdl.generate(**inputs, max_new_tokens=_CAP_TOK)
    caption = proc.decode(ids[0], skip_special_tokens=True)
    return _caption2foods(caption)

def identify_food_items_batch(imgs: Iterable[Image.Image | str | Path], batch_size: int = 8) -> List[List[str]]:
    """
    Batched variant; preserves original batching semantics and outputs list per image.
    """
    proc, mdl = _load_caption()
    pil_imgs = [_rgb(i) for i in imgs]
    if not pil_imgs:
        return []

    bag: List[List[str]] = []
    for i in range(0, len(pil_imgs), batch_size):
        batch = pil_imgs[i : i + batch_size]
        inputs = proc(images=batch, return_tensors="pt").to(mdl.device)
        ids = mdl.generate(**inputs, max_new_tokens=_CAP_TOK)
        bag.extend(_caption2foods(t) for t in proc.batch_decode(ids, skip_special_tokens=True))
    return bag

def _canon(k: str) -> str:
    return re.sub(r"[^a-z]", "_", k.lower()).strip("_")

_SPLIT = re.compile(r"[,&;]|\b(?:and|or)\b", re.I)
_TRSH = re.compile(r"\b(certain|various|other|also|like)\b", re.I)

def _split(t: str) -> List[str]:
    t = re.sub(r"[()–—-]", ",", _TRSH.sub("", t))
    return [s.strip().lower() for s in _SPLIT.split(t) if s.strip()]

@lru_cache(maxsize=None)
def _pat(phrase: str) -> re.Pattern[str]:
    parts = phrase.split()
    sg_last = _inflect.singular_noun(parts[-1]) or parts[-1]
    pl_last = _inflect.plural_noun(sg_last) or sg_last
    head = " ".join(parts[:-1])
    head_escaped = f"{re.escape(head)} " if head else ""
    return re.compile(rf"\b{head_escaped}(?:{re.escape(sg_last)}|{re.escape(pl_last)})\b", re.I)

@lru_cache(maxsize=1)
def _db() -> Dict[str, object]:
    """
    Load and compile the trigger DB into regex patterns.
    Structure of cross_ref_food.json is preserved.
    """
    raw = json.load(_DATA.open())
    out: Dict[str, object] = {}

    for k, v in raw.items():
        key = _canon(k)
        if isinstance(v, dict):
            # nested categories → dict[str, list[Pattern]]
            out[key] = {
                c: [_pat(t) for e in foods for t in _split(e) if len(t.split()) <= 3]
                for c, foods in v.items()
            }
        else:
            # flat list → list[Pattern] (order-preserving unique)
            seen_list: List[str] = []
            seen_set: set[str] = set()
            for e in v:
                for t in _split(e):
                    if len(t.split()) <= 3 and t not in seen_set:
                        seen_set.add(t)
                        seen_list.append(t)
            out[key] = [_pat(t) for t in seen_list]

    return out

def _scan_nested(cat: Dict[str, List[re.Pattern[str]]], food: str) -> List[str]:
    return [c for c, ps in cat.items() if any(p.search(food) for p in ps)]

def _scan_list(pats: List[re.Pattern[str]], label: str, food: str) -> List[str]:
    return [label] if any(p.search(food) for p in pats) else []

_TASKS: List[Tuple[str, object, str]] = [
    ("biogenic_amine", _scan_nested, "biogenic_amines"),
    ("salicylate", _scan_list, "salicylates"),
    ("lectin", _scan_list, "lectins"),
    ("metal", _scan_nested, "metals"),
    ("additive", _scan_nested, "artificial_additives"),
    ("other_trigger", _scan_nested, "other_common_triggers"),
]

def _analyse_one(food: str, data: Dict[str, object]) -> Tuple[str, List[str]]:
    hits: List[str] = []
    for lab, fn, key in _TASKS:
        cat = data.get(key, {})
        # preserve original signature dispatch
        if fn is _scan_list:
            hits.extend(_scan_list(cat, lab, food))  # type: ignore[arg-type]
        else:
            hits.extend(_scan_nested(cat, food))     # type: ignore[arg-type]

    # de-dup preserve order
    seen: set[str] = set()
    uniq: List[str] = []
    for h in hits:
        if h not in seen:
            seen.add(h)
            uniq.append(h)
    return food, uniq

def analyse_triggers(names: Iterable[str] | str, parallel: bool = True, max_workers: Optional[int] = None) -> Dict[str, List[str]]:
    """
    Map input food names → trigger categories using the compiled DB.
    Preserves all original semantics including parallel behavior and normalization.
    """
    foods: Tuple[str, ...]
    if isinstance(names, str):
        foods = tuple(x for x in (n.strip() for n in names.split(",")) if x)
    else:
        foods = tuple(str(f).strip() for f in names)

    foods = tuple(_sg(f.lower()) for f in foods if f)
    if not foods:
        return {}

    data = _db()
    res: Dict[str, List[str]] = {}

    if parallel and len(foods) > 1:
        with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_analyse_one, x, data) for x in foods]
            for f in cf.as_completed(futs):
                k, h = f.result()
                res[k] = h
    else:
        for f in foods:
            _, h = _analyse_one(f, data)
            res[f] = h

    return res
