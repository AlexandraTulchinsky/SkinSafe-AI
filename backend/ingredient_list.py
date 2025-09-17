"""
Simple ingredient-label OCR pipeline:
1. EasyOCR first (fast, accurate on packaging)
2. Tesseract fallback if EZ-OCR output looks like gibberish
3. Vision model fallback (Ollama) if both fail
"""

from __future__ import annotations
import os, re, sys, time, logging, base64
from pathlib import Path
from typing import List, Union
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    import easyocr
except Exception:
    easyocr = None

_MIN_SIDE = int(os.getenv("OCR_MIN_SIDE", "800"))
_MIN_GOOD_TOK = int(os.getenv("OCR_MIN_GOOD_TOK", "3"))
_TESS_TIMEOUT_S = max(0.1, float(os.getenv("OCR_TESS_TIMEOUT_MS", "1000")) / 1000.0)

# Vision model settings
_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llama3.2-vision:latest")
_OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "10000"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("ingredient_ocr")

_LANG = "eng"
_PSM = "--psm 6"
_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ,.;:()[]-/+%&'"

_SEPARATORS = r"[•·\*\+|;∙●◦/:]"

# utils
def _ensure_bgr(img: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
    if isinstance(img, (str, Path)):
        bgr = cv2.imread(str(img))
        if bgr is None:
            raise FileNotFoundError(f"Image not found: {img}")
        return bgr
    if isinstance(img, Image.Image):
        return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    if isinstance(img, np.ndarray):
        return img
    raise TypeError(f"Unsupported type: {type(img)}")

def _resize(bgr: np.ndarray, min_side: int = _MIN_SIDE) -> np.ndarray:
    h, w = bgr.shape[:2]
    s = min(h, w)
    if s >= min_side: return bgr
    scale = float(min_side) / float(s)
    return cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

def _normalize_text(s: str) -> str:
    s = s.replace("ﬁ", "fi").replace("ﬂ", "fl")
    s = s.replace("’", "'").replace("`", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = s.replace("—", "-").replace("–", "-")
    return re.sub(r"[ \t]+", " ", s)

def _clean_and_split(block: str) -> List[str]:
    s = _normalize_text(block)
    s = re.sub(_SEPARATORS, ",", s)
    s = s.replace("(", ",").replace(")", ",")
    s = re.sub(r"\band\/or\b", ",", s, flags=re.I)
    s = re.sub(r"\s+(?:and|or)\s+", ",", s, flags=re.I)
    s = re.sub(r"\s+", " ", s)

    parts = re.split(r"[,\n]", s)
    seen, out = set(), []
    for p in parts:
        t = p.strip().lower().strip(" .;:")
        if not t: continue
        if len(t) < 2: continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def _tokens_look_valid(tokens: List[str], min_good: int = _MIN_GOOD_TOK) -> bool:
    return sum(1 for t in tokens if re.search(r"[A-Za-z]{3,}", t)) >= min_good

#  OCR engines 
_EASYREADER = None
def _easyocr_tokens(bgr: np.ndarray) -> List[str]:
    global _EASYREADER
    if easyocr is None: return []
    if _EASYREADER is None:
        _EASYREADER = easyocr.Reader(["en"], gpu=True)
    try:
        out = _EASYREADER.readtext(bgr, detail=0, paragraph=True)
        txt = _normalize_text(" ".join([r for r in out if r and r.strip()]))
        print(f"[RAW EASY] {txt}", flush=True)
        return _clean_and_split(txt)
    except Exception as e:
        log.warning("EasyOCR failed: %s", e)
        return []

def _tesseract_tokens(bgr: np.ndarray) -> List[str]:
    if pytesseract is None: return []
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cfg = f'{_PSM} --oem 1 -c preserve_interword_spaces=1 -c user_defined_dpi=300 -c tessedit_char_whitelist={_WHITELIST}'
    try:
        txt = pytesseract.image_to_string(rgb, lang=_LANG, config=cfg, timeout=_TESS_TIMEOUT_S)
        txt = _normalize_text(txt or "")
        print(f"[RAW TESS] {txt}", flush=True)
        return _clean_and_split(txt)
    except Exception as e:
        log.warning("Tesseract failed: %s", e)
        return []

def _jpeg_b64_for_vision(img: np.ndarray, max_side: int = 768, quality: int = 85) -> str:
    h, w = img.shape[:2]
    m = max(h, w)
    if m > max_side:
        scale = max_side / float(m)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])[1]
    return base64.b64encode(buf).decode("utf-8")

def _vision_tokens(bgr: np.ndarray) -> List[str]:
    b64 = _jpeg_b64_for_vision(bgr)
    url = f"{_OLLAMA_URL.rstrip('/')}/api/chat"
    prompt = (
        "Extract all ingredients from this image. "
        "Return as a comma-separated list of short tokens."
    )
    payload = {
        "model": _VISION_MODEL,
        "messages": [{"role": "user", "content": prompt, "images": [b64]}],
        "stream": False,
        "options": {"num_predict": 64, "num_gpu": 0},
    }
    try:
        r = requests.post(url, json=payload, timeout=_OLLAMA_TIMEOUT)
        r.raise_for_status()
        txt = (r.json().get("message", {}).get("content") or "").strip()
        print(f"[RAW VISION] {txt}", flush=True)
        return _clean_and_split(txt)
    except Exception as e:
        log.warning("Vision fallback failed: %s", e)
        return []

# API
def extract_ingredients_label(img: Union[str, Path, np.ndarray, Image.Image]) -> List[str]:
    bgr = _ensure_bgr(img)
    bgr = _resize(bgr, _MIN_SIDE)

    # Step 1: EasyOCR
    tokens = _easyocr_tokens(bgr)
    if _tokens_look_valid(tokens):
        log.info("EasyOCR accepted: %d tokens", len(tokens))
        return tokens

    # Step 2: Tesseract fallback
    tokens = _tesseract_tokens(bgr)
    if _tokens_look_valid(tokens):
        log.info("Tesseract accepted: %d tokens", len(tokens))
        return tokens

    # Step 3: Vision fallback
    tokens = _vision_tokens(bgr)
    if _tokens_look_valid(tokens):
        log.info("Vision accepted: %d tokens", len(tokens))
        return tokens

    log.warning("No valid OCR result")
    return []
