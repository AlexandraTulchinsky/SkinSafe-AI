"""
Flow:
  1) CLEAN  → keep only ingredient noun-phrases; remove qualifiers/headers/percentages/clauses.
  2) SPLIT  → split multi-ingredients; condense single noisy ones; light typo-correct; no inventions.
  3) VERIFY → strict boolean gate (drop uncertain/garbled/headers/clauses/numbers).
  4) CANONICAL_DEDUP → conservative LLM pass to collapse family variants & remove duplicates.
  5) DB lookup → exact match; else n-gram/regex sub-phrase fallback.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

import logging
import requests

try:
    from pymongo import MongoClient
except Exception:
    MongoClient = None  

from pydantic import BaseModel, Field


LOG_LEVEL = os.getenv("TINY_RECO_LOGLEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("tiny_reco")


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "dolphin-llama3:latest")

def _ollama_chat(prompt: str, num_predict: int = 256, temperature: float = 0.0) -> str:
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"num_predict": num_predict, "temperature": temperature, "keep_alive": "30m"},
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return (data.get("message") or {}).get("content", "").strip()


MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = os.getenv("MONGO_DB", "eczema_db")
MONGO_COLL = os.getenv("MONGO_COLL", "ingredients")

def _get_mongo_collection():
    if MongoClient is None:
        log.warning("pymongo not available; DB lookups will be skipped.")
        return None
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)  # type: ignore
        col = client[MONGO_DB][MONGO_COLL]
        client.admin.command("ping")
        return col
    except Exception as e:
        log.warning("Mongo connection failed: %s", e)
        return None


class IngredientAdvice(BaseModel):
    safe: List[str] = Field(default_factory=list)
    avoid: List[str] = Field(default_factory=list)

class RecommendationOutput(BaseModel):
    product: str
    ingredients: IngredientAdvice


def _dedupe_preserve(seq: List[str]) -> List[str]:
    out, seen = [], set()
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _lookup_triggers_exact(col, name: str) -> Optional[List[str]]:
    if col is None:
        return None
    key = _norm(name)
    try:
        doc = col.find_one({"ingredient": key}, {"triggers": 1})
        if doc and isinstance(doc.get("triggers"), list):
            return [t.strip().lower() for t in doc["triggers"] if isinstance(t, str)]
        doc = col.find_one({"examples": key}, {"triggers": 1})
        if doc and isinstance(doc.get("triggers"), list):
            return [t.strip().lower() for t in doc["triggers"] if isinstance(t, str)]
    except Exception as e:
        log.warning("Mongo exact lookup error for %r: %s", name, e)
    return None

def _lookup_triggers_fallback(col, name: str) -> Optional[List[str]]:
    """
    Fallback when exact match fails:
      - Try longest->shortest n-grams from the token.
      - For each candidate, try exact ('ingredient'/'examples').
      - If still nothing, try case-insensitive regex whole-word match on 'ingredient' and 'examples'.
    The first match wins.
    """
    if col is None:
        return None
    words = [w for w in _norm(name).split(" ") if w]
    tried = set()

    def _try_candidate(cand: str) -> Optional[List[str]]:
        key = cand.strip()
        if not key or key in tried:
            return None
        tried.add(key)
        try:
            # exact candidate
            doc = col.find_one({"ingredient": key}, {"triggers": 1})
            if doc and isinstance(doc.get("triggers"), list):
                return [t.strip().lower() for t in doc["triggers"] if isinstance(t, str)]
            doc = col.find_one({"examples": key}, {"triggers": 1})
            if doc and isinstance(doc.get("triggers"), list):
                return [t.strip().lower() for t in doc["triggers"] if isinstance(t, str)]
            # regex whole-word candidate (case-insensitive)
            rx = {"$regex": rf"(?:^|\s){re.escape(key)}(?:\s|$)", "$options": "i"}
            doc = col.find_one({"ingredient": rx}, {"triggers": 1})
            if doc and isinstance(doc.get("triggers"), list):
                return [t.strip().lower() for t in doc["triggers"] if isinstance(t, str)]
            doc = col.find_one({"examples": rx}, {"triggers": 1})
            if doc and isinstance(doc.get("triggers"), list):
                return [t.strip().lower() for t in doc["triggers"] if isinstance(t, str)]
        except Exception as e:
            log.warning("Mongo fallback lookup error for %r (cand=%r): %s", name, key, e)
        return None

    # try full phrase first, then n-grams (len -> 1)
    for L in range(len(words), 0, -1):
        for i in range(0, len(words) - L + 1):
            cand = " ".join(words[i:i+L])
            hits = _try_candidate(cand)
            if hits:
                return hits
    return None

def _lookup_triggers(col, name: str) -> Optional[List[str]]:
    """
    Public DB lookup: exact first, then fallback to sub-word/phrase heuristics.
    """
    hits = _lookup_triggers_exact(col, name)
    if hits:
        return hits
    return _lookup_triggers_fallback(col, name)

# -----------------------------------------------------------------------------
# LLM passes — STRICT base prompts (CLEAN, SPLIT, VERIFY) + conservative CANONICAL_DEDUP
# -----------------------------------------------------------------------------
def _llm_clean(raw_lines: List[str]) -> List[str]:
    """
    STRICT CLEAN: keep only ingredient-like tokens; strip qualifiers/headers/percentages/clauses.
    """
    raw_json = json.dumps(raw_lines, ensure_ascii=False)
    prompt = (
        "You are cleaning OCR output from packaged-food labels.\n"
        "Return ONLY a JSON array of tokens that are standalone ingredient names or short ingredient noun-phrases (1–3 words).\n"
        "\n"
        "Rules:\n"
        "- Keep tokens that clearly name a food/ingredient/additive.\n"
        "- If a token has qualifiers/marketing (e.g., reduced, natural, pure, low-fat), drop the qualifier and keep only the ingredient noun if present.\n"
        "- Drop section headers (e.g., ingredients, allergen information), numbers/percentages/measurements, preparation/storage notes, and relational/claim phrases (e.g., contains, may contain, from, with, minimum, processed in).\n"
        "- Drop whole sentences/clauses. If a single clear ingredient noun-phrase can be extracted from a clause, output only that noun-phrase.\n"
        "- Be conservative: if you cannot identify a valid ingredient noun-phrase, drop it.\n"
        "- Output MUST be a JSON array of strings only. No prose, no code fences.\n"
        f"\nINPUT: {raw_json}\nOUTPUT:"
    )
    try:
        raw = _ollama_chat(prompt, 192, 0.0)
        s = raw.strip()
        if s.startswith("```"):
            s = s.strip("`").strip()
        arr = json.loads(s[s.find('['):s.rfind(']')+1])
        if not isinstance(arr, list):
            return []
        return _dedupe_preserve([str(x).strip() for x in arr if isinstance(x, str) and str(x).strip()])
    except Exception as e:
        log.warning("LLM clean failed: %s", e)
        return []

def _llm_split_and_fix(tokens: List[str]) -> List[str]:
    """
    STRICT SPLIT & FIX:
      - Always split glued multi-food tokens into separate items.
      - Condense single noisy ones into a short noun-phrase.
      - Correct common OCR typos.
      - Drop junk if it can't be fixed confidently.
    """
    raw_json = json.dumps(tokens, ensure_ascii=False)

    prompt = (
        "You are refining ingredient tokens from OCR. Return ONE flat JSON array of short ingredient tokens.\n"
        "\n"
        "Rules:\n"
        "- If a token contains multiple food/ingredient words jammed together (e.g., 'lactose whey skim milk powder'), "
        "ALWAYS split into separate valid items in order (['lactose','whey','milk powder']).\n"
        "- If a token mixes foods with other words (e.g., 'cheese cultures salt enzymes'), split into clean separate items "
        "(['cheese cultures','salt','enzymes']).\n"
        "- If a token is ONE ingredient but noisy/long, remove extra words and reduce to a valid noun-phrase (≤2 words if possible). "
        "Example: 'sunflower oul' → 'sunflower oil'.\n"
        "- Correct minor OCR spelling errors (e.g., 'onio'→'onion', 'cawola'→'canola').\n"
        "- Do not invent new ingredients. If you cannot confidently extract valid food terms, DROP that item.\n"
        "- Output MUST be ONLY a JSON array of strings (no prose, no code fences).\n"
        f"\nINPUT: {raw_json}\nOUTPUT:"
    )

    try:
        raw = _ollama_chat(prompt, 300, 0.0).strip()
        if raw.startswith("```"):
            raw = raw.strip("`").strip()
        arr = json.loads(raw[raw.find('['):raw.rfind(']')+1])
        if not isinstance(arr, list):
            return []
        cleaned = [str(x).strip() for x in arr if isinstance(x, str) and str(x).strip()]
        return _dedupe_preserve(cleaned)
    except Exception as e:
        log.warning("LLM split_or_condense failed: %s", e)
        return []

def _llm_verify_one(token: str) -> bool:
    """
    STRICT VERIFY:
      - Accept only clearly valid short ingredient noun-phrases.
      - Reject long/junky phrases unless they are well-known additive terms.
    """
    token = (token or "").strip()
    if not token:
        return False

    prompt = (
        "You will receive ONE token from a food ingredient list. Return EXACTLY one JSON boolean: true or false.\n"
        "\n"
        "Accept (true) only if:\n"
        "- It is a valid standalone ingredient or additive noun-phrase (head noun with ≤2 modifiers).\n"
        "- It may be multi-word ONLY if it is a well-known food additive (e.g., 'monosodium glutamate', 'calcium carbonate').\n"
        "\n"
        "Reject (false) if:\n"
        "- The token is more than 3 words long and not a standard additive name.\n"
        "- It looks like a glued mashup of multiple ingredients.\n"
        "- It is a section header, number/percentage, relational/claim phrase (contains, from, with, minimum), adjective-only, or garbled OCR fragment.\n"
        "- If uncertain, return false.\n"
        f"\nINPUT: {json.dumps(token, ensure_ascii=False)}\nOUTPUT:"
    )

    try:
        raw = _ollama_chat(prompt, 12, 0.0).strip().lower()
        if raw.startswith("true"):
            return True
        if raw.startswith("false"):
            return False
        try:
            val = json.loads(raw)
            return bool(val) is True
        except Exception:
            return False
    except Exception:
        return False

def _llm_canonicalize_and_dedupe(tokens: List[str]) -> List[str]:
    """
    STRICT FINAL PASS: collapse obvious intra-family variants into a single canonical token and dedupe.
    Decisive behavior: when a base family and its variants both appear, output ONLY the base form.
    Keep technical additives (e.g., cheese cultures, enzymes) distinct from the food family.

    Returns: list of canonical tokens, first-occurrence order preserved.
    """
    if not tokens:
        return []
    raw_json = json.dumps(tokens, ensure_ascii=False)

    prompt = (
        "You will receive a JSON array of verified ingredient tokens.\n"
        "Task: collapse redundant variants by merging tokens that denote the same base ingredient family into a single canonical token, "
        "then remove duplicates while preserving first-occurrence order.\n"
        "\n"
        "Decision rules (strict & decisive):\n"
        "- If multiple tokens are variants/forms of the SAME base food family, output ONLY the base form and DROP the variants. "
        "Do not keep both the base and its variants.\n"
        "- Determine the base by the simplest, most general head noun of the family (e.g., milk, cheese). "
        "Examples of common variant patterns include fat-level modifiers (skim/whole), process/physical-form modifiers (powder, condensed, evaporated), "
        "and named subtypes (e.g., cheddar, parmesan for cheese). These should collapse to the base.\n"
        "- Keep technical additives or processing agents that are NOT the food itself (e.g., cheese cultures, enzymes) as separate tokens; do NOT collapse them into the food base.\n"
        "- Do NOT invent new tokens. If you cannot confidently identify a base family for a token, keep the token as-is.\n"
        "- Output MUST be ONLY a JSON array of strings (no prose, no code fences).\n"
        "\n"
        "Pattern guidance (not exhaustive):\n"
        "INPUT: [\"skim milk\", \"milk\"]\nOUTPUT: [\"milk\"]\n"
        "INPUT: [\"milk powder\", \"skimmed milk\", \"whole milk\"]\nOUTPUT: [\"milk\"]\n"
        "INPUT: [\"cheddar cheese\", \"cheese cultures\"]\nOUTPUT: [\"cheese\", \"cheese cultures\"]\n"
        "INPUT: [\"parmesan cheese\", \"cheddar cheese\"]\nOUTPUT: [\"cheese\"]\n"
        "INPUT: [\"milk\", \"cheese\"]\nOUTPUT: [\"milk\", \"cheese\"]\n"
        "\n"
        f"INPUT: {raw_json}\nOUTPUT:"
    )

    try:
        raw = _ollama_chat(prompt, 180, 0.0).strip()
        if raw.startswith("```"):
            raw = raw.strip("`").strip()
        arr = json.loads(raw[raw.find('['):raw.rfind(']')+1])
        if not isinstance(arr, list):
            return _dedupe_preserve(tokens)
        return _dedupe_preserve([str(x).strip() for x in arr if isinstance(x, str) and str(x).strip()])
    except Exception as e:
        log.warning("LLM canonical_dedup failed: %s", e)
        return _dedupe_preserve(tokens)

# Orchestration
def _preclean_with_llm_list(raw_items: List[str]) -> List[str]:
    """
    Base strict pipeline + final conservative canonical dedupe:
      1) CLEAN → 2) SPLIT → 3) VERIFY → 4) CANONICAL_DEDUP
    """
    step1 = _llm_clean(raw_items)
    step2 = _llm_split_and_fix(step1)
    verified = [t for t in step2 if _llm_verify_one(t)]
    canonical = _llm_canonicalize_and_dedupe(verified)
    cleaned = _dedupe_preserve(canonical)

    log.info("LLM clean tokens: %s", step1)
    log.info("LLM split_or_condense tokens: %s", step2)
    log.info("LLM verified tokens: %s", verified)
    log.info("LLM canonical_dedup tokens: %s", cleaned)
    return cleaned

# API
def recommend_from_plain_list(ingredients: List[str]) -> Dict[str, Any]:
    """
    End-to-end:
      - LLM: clean → split → verify → canonical_dedup (strict base + conservative dedupe)
      - DB: exact match then fallback sub-word/phrase
      - Shape: unchanged for frontend
    """
    product = ", ".join(ingredients)  # keep original messy list for UI

    if not ingredients:
        return {"product": "", "ingredients": {"safe": [], "avoid": []}}

    cleaned = _preclean_with_llm_list(ingredients)
    if not cleaned:
        return {"product": product, "ingredients": {"safe": [], "avoid": []}}

    col = _get_mongo_collection()
    safe: List[str] = []
    avoid: List[str] = []

    for name in cleaned:
        trigs = _lookup_triggers(col, name)
        if trigs:
            avoid.append(f"{name} ({', '.join(trigs)})")
        else:
            safe.append(name)

    return RecommendationOutput(
        product=product,
        ingredients=IngredientAdvice(safe=safe, avoid=avoid)
    ).dict()
