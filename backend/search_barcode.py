from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union, Optional

import cv2  
import easyocr  
import numpy as np
import requests

log = logging.getLogger(__name__)

FOOD_API_TEMPLATE: str = "https://world.openfoodfacts.org/api/v2/product/{code}.json"
HTTP_TIMEOUT_S: int = 20

# OCR config
OCR_LANGS: Sequence[str] = ("en",)

# Regexes
INGR_HEADER = re.compile(r"ingredients?\s*[:\-]?\s*", re.I)
SPLIT_REGEX = re.compile(r"[;,•·]|(?<!\d)\.(?!\d)")
PAREN_CLEAN = re.compile(r"\([^)]*\)")

# Types
BGRImage = np.ndarray

_DETECTOR = cv2.barcode.BarcodeDetector()
_READER = easyocr.Reader(list(OCR_LANGS)) 


def _clean(text: str) -> str:
    """Normalize ingredient tokens (trim + lowercase)."""
    return text.strip().lower()


def _split_ingredient_line(s: str) -> List[str]:
    """Split a free-text ingredient string into tokens."""
    s = PAREN_CLEAN.sub("", s)
    return [_clean(p) for p in SPLIT_REGEX.split(s) if _clean(p)]


def _ensure_bgr(img: Union[str, Path, BGRImage]) -> BGRImage:
    """
    Accept a filesystem path or a NumPy image and return a BGR NumPy image.
    Raises FileNotFoundError on invalid paths. Keeps behavior identical for arrays.
    """
    if isinstance(img, (str, Path)):
        bgr = cv2.imread(str(img))
        if bgr is None:
            raise FileNotFoundError(img)
        return bgr
    return img  


def _from_open_food_facts(code: str, *, debug: bool = False) -> List[str]:
    """
    Look up ingredients for an EAN/UPC code via Open Food Facts.
    Returns [] on not found or any error (identical to original behavior).
    """
    url = FOOD_API_TEMPLATE.format(code=code)
    try:
        js = requests.get(url, timeout=HTTP_TIMEOUT_S).json()
        if js.get("status") != 1:
            return []  

        prod = js["product"]

        if isinstance(prod.get("ingredients"), list):
            raw = [i.get("text", "") for i in prod["ingredients"]]
            cleaned = [_clean(t) for t in raw if _clean(t)]
            if debug and cleaned:
                log.info("[DEBUG] Ingredients pulled via OpenFoodFacts: %s", cleaned)
            return cleaned

        # Fallback: free-text fields
        txt = prod.get("ingredients_text") or prod.get("ingredients_text_en", "")
        ing = _split_ingredient_line(txt)
        if debug and ing:
            log.info("[DEBUG] Ingredients pulled via OpenFoodFacts: %s", ing)
        return ing
    except Exception:
        # Preserve original silent-failure semantics
        return []

def _ocr_ingredients(bgr_img: BGRImage) -> List[str]:
    """
    OCR the image and try to isolate the ingredient line(s).
    Note: EasyOCR expects RGB; convert from BGR.
    """
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    text = " ".join(_READER.readtext(rgb, detail=0))
    lines: List[str] = []
    for ln in text.splitlines():
        m = INGR_HEADER.match(ln)
        lines.append(ln[m.end():] if m else ln)
    return _split_ingredient_line(" ".join(lines))


def _valid_ean13(code: str) -> bool:
    """Validate an EAN-13 string (includes UPC-A normalized with a leading 0)."""
    if len(code) != 13 or not code.isdigit():
        return False
    s = sum((3 if i % 2 else 1) * int(d) for i, d in enumerate(code[:-1]))
    return (10 - s % 10) % 10 == int(code[-1])


def _decode_barcode(bgr_img: BGRImage, *, debug: bool = False) -> Optional[str]:
    """
    Try to detect and decode a barcode at 0°, 90°, 180°, 270° rotations.
    Returns a normalized EAN-13 string or None. Keeps messages/format for DEBUG logs.
    """
    def _try(mat: BGRImage) -> Optional[str]:
        res = _DETECTOR.detectAndDecode(mat)

        # OpenCV returns different shapes across versions; handle both.
        if not isinstance(res, tuple):
            return None

        ok, infos = (res[0], res[1]) if len(res) == 4 else (bool(res[0]), res[0])
        if not ok or not infos:
            return None

        raw = infos[0] if isinstance(infos, (list, tuple)) else infos
        code = re.sub(r"\D", "", raw)

        # Normalize UPC-A (12 digits) to EAN-13 by left-padding a zero
        if len(code) == 12:
            code = "0" + code

        return code if _valid_ean13(code) else None

    for k in range(4):  # 0°, 90°, 180°, 270°
        rotated = np.rot90(bgr_img, k).copy()
        code = _try(rotated)
        if code:
            if debug:
                log.info("[DEBUG] Barcode detected (%d°): %s", k * 90, code)
            return code
    return None


def extract_ingredients(image: Union[str, Path, BGRImage], *, debug: bool = False) -> List[str]:
    """
    Extract a clean ingredient list from either:
      • A barcode (via Open Food Facts), or
      • OCR of the label text (fallback).

    Parameters
    ----------
    image : Union[str, Path, np.ndarray]
        Either a path to an image file or a NumPy BGR image (OpenCV style).
    debug : bool
        If True, emit the same DEBUG prints as in the original implementation.

    Returns
    -------
    List[str]
        A list of normalized ingredient tokens (lowercased, trimmed).
    """
    bgr = _ensure_bgr(image)

    # 1) Try barcode → OFF API
    code = _decode_barcode(bgr, debug=debug)
    if code:
        ingredients = _from_open_food_facts(code, debug=debug)
        if ingredients:
            return ingredients

    # 2) Fallback: OCR
    return _ocr_ingredients(bgr)

