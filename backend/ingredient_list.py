from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
from PIL import Image
import easyocr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("ingredient_ocr")

_DEFAULT_LANGS: Sequence[str] = ("en", "fr")
_STOP_AT: Sequence[str] = [
    "contains", "ingrédients",  # English / French
    "nutrition", "valeur",      # nutrition panels
    "allergen", "allergy", "Ingredients"
]
_SEPARATORS = r"[•·\*\+|;]"  # bullets, mid-dots, unusual delimiters

# EasyOCR Reader singleton (lazy load)
_ocr_reader: Optional[easyocr.Reader] = None


def ocr_easy(src: Union[str, Path, np.ndarray, Image.Image],
             lang: Sequence[str] = _DEFAULT_LANGS) -> str:
    global _ocr_reader
    if _ocr_reader is None:
        log.debug("Initializing EasyOCR Reader with languages: %s", lang)
        _ocr_reader = easyocr.Reader(lang, gpu=False)

    if isinstance(src, (str, Path)):
        result = _ocr_reader.readtext(str(src), detail=0, paragraph=True)
    else:
        result = _ocr_reader.readtext(src, detail=0, paragraph=True)

    return "\n".join(result)


def extract_block(txt: str) -> Optional[str]:
    """
    Find the substring that follows 'Ingredients:' / 'Ingredients',
    stopping at the first stop word (e.g., 'Contains', 'Nutrition').

    Returns None if no ingredient header is found.
    """
    m = re.search(r"ingredients?\s*[:\-]?\s*", txt, re.IGNORECASE)
    if not m:
        return None

    sub = txt[m.end():]
    for stop_word in _STOP_AT:
        m2 = re.search(stop_word, sub, re.IGNORECASE)
        if m2:
            sub = sub[:m2.start()]
            break
    return sub


def clean_split(block: str) -> List[str]:
    
    block = re.sub(_SEPARATORS, ",", block)

    block = re.sub(r"[\(\[]", ",", block)
    block = re.sub(r"[\)\]]", ",", block)

    block = re.sub(r"\band\/or\b", ",", block, flags=re.IGNORECASE)
    block = re.sub(r"\band\b", ",", block, flags=re.IGNORECASE)

    parts = re.split(r"[,\n]", block)

    tokens = [p.strip().lower() for p in parts if p.strip()]

    seen, final = set(), []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            final.append(t)

    return final


def extract_ingredients_label(img_or_path: Union[str, Path, np.ndarray, Image.Image]) -> List[str]:
    raw_text = ocr_easy(img_or_path)
    block = extract_block(raw_text)
    return clean_split(block) if block else []

