# vision.py
# -----------------------------------------------------------------------------
# Returns a PLAIN LIST of food/ingredient tokens from an image.
# No trigger scanning or DB logic here — just vision -> caption -> tokens.
#
# Public API:
#   - identify_food_items(img) -> List[str]
#   - identify_food_items_batch(imgs, batch_size=8) -> List[List[str]]
#
# Behavior:
#   1) (Optional) Ask vision model if image is FOOD vs NOT_FOOD (cheap guard).
#   2) Caption with Ollama (llama3.2-vision or your chosen vision model).
#   3) Parse caption into short, singularized tokens (≤ 3 words), dedup, order-preserving.
# -----------------------------------------------------------------------------

from __future__ import annotations

import base64
import logging
import os
import re
import sys
from io import BytesIO
from pathlib import Path
from typing import Iterable, List, Union, Optional

import inflect
import requests
import spacy
import subprocess
from PIL import Image

# --------------------------------------------------------------------------------------
# Env / logging
# --------------------------------------------------------------------------------------

LOG_LEVEL = os.getenv("STEP2_LOGLEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("step2_food")

# --------------------------------------------------------------------------------------
# NLP helpers
# --------------------------------------------------------------------------------------

_inflect = inflect.engine()
# keep spaCy config minimal (token POS tagging only)
_nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer", "textcat"])

def _sg(phrase: str) -> str:
    """Singularize the last word of a phrase, preserving the rest."""
    phrase = phrase.strip()
    if not phrase:
        return phrase
    parts = phrase.split()
    last = _inflect.singular_noun(parts[-1]) or parts[-1]
    return " ".join([*parts[:-1], last])

# --------------------------------------------------------------------------------------
# Ollama Vision configuration
# --------------------------------------------------------------------------------------

_CAP_TOK: int = int(os.getenv("VISION_CAP_TOK", "16"))  # small, concise captions

_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
_LLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llama3.2-vision:latest")

# Warmup toggles (disable by setting 0)
_WARMUP_OLLAMA = os.getenv("WARMUP_OLLAMA", "1") != "0"

# Request timeout (ms → s)
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "10000000"))

# --------------------------------------------------------------------------------------
# Image utilities
# --------------------------------------------------------------------------------------

def _rgb(im: Image.Image | str | Path) -> Image.Image:
    """Ensure a PIL RGB image from path or Image."""
    if isinstance(im, Image.Image):
        return im.convert("RGB")
    return Image.open(im).convert("RGB")

def _jpeg_b64_for_vision(img: Image.Image, max_side: int = 768, quality: int = 85) -> str:
    """
    Downscale the longer side to ≤ max_side and encode as JPEG base64 string.
    Keeps payloads small to avoid timeouts.
    """
    w, h = img.size
    m = max(w, h)
    im = img
    if m > max_side:
        scale = max_side / float(m)
        im = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# --------------------------------------------------------------------------------------
# Vision model calls
# --------------------------------------------------------------------------------------

def _llama_caption_prompt() -> str:
    return (
        "List all visible foods and ingredients in the image. "
        "Be specific: include individual components (e.g., bun, beef patty, cheese, lettuce, tomato). "
        "Output as a short comma-separated list of ingredients only, no full sentences."
    )

def _warmup_vision_cpu() -> None:
    """
    Warm up the vision model on CPU so a different model (e.g., Dolphin) can keep the GPU.
    Spawn once; non-blocking. Safe to fail.
    """
    if not _WARMUP_OLLAMA:
        return
    try:
        subprocess.Popen(
            ["ollama", "run", "--cpu", _LLAMA_VISION_MODEL, "warmup"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log.info("Warmup requested for vision model on CPU: %s", _LLAMA_VISION_MODEL)
    except Exception as e:
        log.warning("Vision warmup failed (non-fatal): %s", e)

# Call warmup once at import
_warmup_vision_cpu()

def _vision_is_food(img_b64: str) -> Optional[bool]:
    """
    Ask the vision model to classify the image as FOOD or NOT_FOOD.
    Returns True/False, or None on error/timeouts.
    """
    url = f"{_OLLAMA_URL}/api/chat"
    prompt = (
        "Answer with exactly one word: FOOD or NOT_FOOD.\n"
        "FOOD = cooked/raw meals, dishes, snacks, drinks, packaged food.\n"
        "NOT_FOOD = ingredient label text, barcodes, documents, scenery, people, objects."
    )
    payload = {
        "model": _LLAMA_VISION_MODEL,
        "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
        "stream": False,
        "options": {
            "num_predict": 3,      # tiny
            "keep_alive": "30m",
            "num_gpu": 0,          # force CPU for this model
        },
    }
    try:
        r = requests.post(url, json=payload, timeout=max(1, OLLAMA_TIMEOUT // 100))
        r.raise_for_status()
        out = (r.json().get("message", {}).get("content") or "").strip().upper()
        log.debug("Vision FOOD/NOT_FOOD raw: %r", out)
        if out.startswith("FOOD"):
            return True
        if out.startswith("NOT_FOOD"):
            return False
    except requests.exceptions.ReadTimeout:
        log.warning("Vision FOOD/NOT_FOOD timed out")
    except Exception as e:
        log.error("Vision FOOD/NOT_FOOD failed: %s", e)
    return None

def _ollama_chat_caption(img_b64: str, num_predict: int = _CAP_TOK) -> str:
    """
    Ask the vision model for a concise, comma-separated ingredient list.
    Returns raw text (we’ll parse into tokens).
    """
    url = f"{_OLLAMA_URL}/api/chat"
    payload = {
        "model": _LLAMA_VISION_MODEL,
        "messages": [{
            "role": "user",
            "content": _llama_caption_prompt(),
            "images": [img_b64],
        }],
        "stream": False,
        "options": {
            "num_predict": int(num_predict),
            "keep_alive": "30m",
            "num_gpu": 0,  # run on CPU
        },
    }
    for attempt in range(3):
        try:
            r = requests.post(url, json=payload, timeout=max(3, OLLAMA_TIMEOUT // 10))
            r.raise_for_status()
            content = (r.json().get("message", {}).get("content") or "").strip()
            log.debug("Vision caption raw: %r", content)
            return content
        except requests.exceptions.ReadTimeout:
            log.warning("Vision caption timed out (attempt %d/3)", attempt + 1)
        except Exception as e:
            log.error("Ollama caption call failed: %s", e)
            break
    return ""

# --------------------------------------------------------------------------------------
# Caption → tokens
# --------------------------------------------------------------------------------------

def _caption2foods(text: str) -> List[str]:
    """
    Turn a caption into short tokens:
      - Try comma-split (expected),
      - If too few, fall back to spaCy (NOUNs and ADJ+NOUN bigrams),
      - Keep tokens ≤ 3 words,
      - Singularize the last word,
      - Deduplicate while preserving order.
    """
    if not text:
        return []

    parts = [p.strip().lower() for p in text.split(",") if p.strip()]
    tokens: List[str] = []

    seen = set()
    for p in parts:
        if len(p.split()) <= 3:
            s = _sg(p)
            if s and s not in seen:
                seen.add(s)
                tokens.append(s)

    # Fallback for sentence-like captions
    if len(tokens) < 2:
        doc = _nlp(text.lower())
        candidates: List[str] = []
        for i, tok in enumerate(doc):
            if tok.pos_ == "NOUN" and tok.is_alpha:
                candidates.append(tok.text)
                if i and doc[i - 1].pos_ == "ADJ":
                    candidates.append(f"{doc[i - 1].text} {tok.text}")

        seen2 = set()
        for w in candidates:
            if len(w.split()) <= 3:
                s = _sg(w)
                if s and s not in seen2:
                    seen2.add(s)
                    tokens.append(s)

    return tokens

# --------------------------------------------------------------------------------------
# Public API (PLAIN LIST ONLY)
# --------------------------------------------------------------------------------------

def identify_food_items(img: Image.Image | str | Path) -> List[str]:
    """
    Photo → (optional FOOD/NOT_FOOD) → caption → tokens.
    Always returns a plain list of ingredient/food tokens (lowercase).
    Returns [] on failure or NOT_FOOD classification.
    """
    pil = _rgb(img)
    b64 = _jpeg_b64_for_vision(pil, max_side=768, quality=85)

    is_food = _vision_is_food(b64)
    if is_food is False:
        log.info("Vision says NOT_FOOD; skipping caption")
        return []
    # If None (error), still attempt caption once.

    caption = _ollama_chat_caption(b64, num_predict=_CAP_TOK)
    tokens = _caption2foods(caption) if caption else []
    log.info("identify_food_items → %d tokens", len(tokens))
    return tokens
