# -*- coding: utf-8 -*-
"""
recommend.py — DB classifier with LLM CLEAN + SPLIT_OR_CONDENSE + VERIFY (+ final CANONICAL_DEDUP)
Base = original strict 3-stage pipeline; adds a conservative final dedupe/canonicalization step.
Also keeps improved DB lookup: exact match first, then n-gram/regex sub-phrase fallback.

Flow:
  1) CLEAN  → keep only ingredient noun-phrases; remove qualifiers/headers/percentages/clauses.
  2) SPLIT  → split multi-ingredients; condense single noisy ones; light typo-correct; no inventions.
  3) VERIFY → strict boolean gate (drop uncertain/garbled/headers/clauses/numbers).
  4) CANONICAL_DEDUP → conservative LLM pass to collapse family variants & remove duplicates.
  5) DB lookup → exact match; else n-gram/regex sub-phrase fallback.

Env:
  OLLAMA_BASE_URL, OLLAMA_MODEL
  MONGO_URI (default: mongodb://localhost:27017/)
  MONGO_DB  (default: eczema_db)
  MONGO_COLL (default: ingredients)
  TINY_RECO_LOGLEVEL (default: INFO)
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
    MongoClient = None  # type: ignore

from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOG_LEVEL = os.getenv("TINY_RECO_LOGLEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("tiny_reco")

# -----------------------------------------------------------------------------
# Ollama config
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Mongo config
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Pydantic models
# -----------------------------------------------------------------------------
class IngredientAdvice(BaseModel):
    safe: List[str] = Field(default_factory=list)
    avoid: List[str] = Field(default_factory=list)

class RecommendationOutput(BaseModel):
    product: str
    ingredients: IngredientAdvice

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _dedupe_preserve(seq: List[str]) -> List[str]:
    out, seen = [], set()
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

# -----------------------------------------------------------------------------
# DB lookups (exact + sub-word/phrase fallback)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
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









# # -*- coding: utf-8 -*-
# """
# recommend.py — DB classifier with LLM CLEAN + SPLIT_OR_CONDENSE + VERIFY

# Goals (prompt-tuned):
#   • Strip qualifiers/marketing (e.g., "reduced cocoa" → "cocoa").
#   • Split adjacent distinct ingredients (e.g., "sugar glucose syrup" → ["sugar","glucose syrup"]).
#   • Canonicalize family variants (e.g., "skim milk", "milk powder", "milk chocolate" → "milk"),
#     while keeping distinct families separate (e.g., "egg powder" ≠ "milk").
#   • Drop headers/percentages/clauses ("ingredients", "4% minimum", "contains ...").

# Flow:
#   1) CLEAN → keep only ingredient noun-phrases; remove qualifiers/headers/percentages/clauses.
#   2) SPLIT_OR_CONDENSE → split multi-ingredients; condense single noisy ones; canonicalize & dedupe families.
#   3) VERIFY → strict boolean: accept only short ingredient noun-phrases.

# Env:
#   OLLAMA_BASE_URL, OLLAMA_MODEL
#   MONGO_URI (default: mongodb://localhost:27017/)
#   MONGO_DB  (default: eczema_db)
#   MONGO_COLL(default: ingredients)
#   TINY_RECO_LOGLEVEL (default: INFO)
# """

# from __future__ import annotations

# import json
# import os
# import re
# import sys
# import subprocess
# from typing import Any, Dict, List, Optional

# import logging
# import requests

# try:
#     from pymongo import MongoClient
# except Exception:
#     MongoClient = None  # type: ignore

# from pydantic import BaseModel, Field

# # -----------------------------------------------------------------------------
# # Logging
# # -----------------------------------------------------------------------------
# LOG_LEVEL = os.getenv("TINY_RECO_LOGLEVEL", "INFO").upper()
# logging.basicConfig(
#     level=LOG_LEVEL,
#     format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
#     stream=sys.stderr,
# )
# log = logging.getLogger("tiny_reco")

# # -----------------------------------------------------------------------------
# # Ollama config
# # -----------------------------------------------------------------------------
# OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
# OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "dolphin-llama3:latest")

# def _ollama_chat(prompt: str, num_predict: int = 256, temperature: float = 0.0) -> str:
#     url = f"{OLLAMA_BASE_URL}/api/chat"
#     payload = {
#         "model": OLLAMA_MODEL,
#         "messages": [{"role": "user", "content": prompt}],
#         "stream": False,
#         "options": {"num_predict": num_predict, "temperature": temperature, "keep_alive": "30m"},
#     }
#     r = requests.post(url, json=payload, timeout=120)
#     r.raise_for_status()
#     data = r.json()
#     return (data.get("message") or {}).get("content", "").strip()

# # -----------------------------------------------------------------------------
# # Mongo config
# # -----------------------------------------------------------------------------
# MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
# MONGO_DB = os.getenv("MONGO_DB", "eczema_db")
# MONGO_COLL = os.getenv("MONGO_COLL", "ingredients")

# def _get_mongo_collection():
#     if MongoClient is None:
#         log.warning("pymongo not available; DB lookups will be skipped.")
#         return None
#     try:
#         client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)  # type: ignore
#         col = client[MONGO_DB][MONGO_COLL]
#         client.admin.command("ping")
#         return col
#     except Exception as e:
#         log.warning("Mongo connection failed: %s", e)
#         return None

# # -----------------------------------------------------------------------------
# # Pydantic models
# # -----------------------------------------------------------------------------
# class IngredientAdvice(BaseModel):
#     safe: List[str] = Field(default_factory=list)
#     avoid: List[str] = Field(default_factory=list)

# class RecommendationOutput(BaseModel):
#     product: str
#     ingredients: IngredientAdvice

# # -----------------------------------------------------------------------------
# # Helpers
# # -----------------------------------------------------------------------------
# def _dedupe_preserve(seq: List[str]) -> List[str]:
#     out, seen = [], set()
#     for x in seq:
#         if x not in seen:
#             seen.add(x)
#             out.append(x)
#     return out

# def _lookup_triggers(col, name: str) -> Optional[List[str]]:
#     if col is None:
#         return None
#     key = re.sub(r"\s+", " ", name.strip().lower())
#     try:
#         doc = col.find_one({"ingredient": key}, {"triggers": 1})
#         if doc and isinstance(doc.get("triggers"), list):
#             return [t.strip().lower() for t in doc["triggers"] if isinstance(t, str)]
#         doc = col.find_one({"examples": key}, {"triggers": 1})
#         if doc and isinstance(doc.get("triggers"), list):
#             return [t.strip().lower() for t in doc["triggers"] if isinstance(t, str)]
#     except Exception as e:
#         log.warning("Mongo lookup error for %r: %s", name, e)
#     return None

# # -----------------------------------------------------------------------------
# # LLM passes — PROMPTS UPDATED WITH TARGETED FEW-SHOT
# # -----------------------------------------------------------------------------
# def _llm_clean(raw_lines: List[str]) -> List[str]:
#     """
#     Keep only ingredient-like tokens; strip qualifiers/headers/percentages/clauses.
#     """
#     raw_json = json.dumps(raw_lines, ensure_ascii=False)

#     prompt = (
#         "You are cleaning OCR output from packaged-food labels.\n"
#         "Return ONLY a JSON array of tokens that are standalone ingredient names or short ingredient noun-phrases (1–3 words).\n"
#         "\n"
#         "Rules:\n"
#         "- Keep tokens that clearly name a food/ingredient/additive (e.g., milk, sugar, glucose syrup, cocoa, vanilla extract, sunflower oil, whey powder).\n"
#         "- If a token has qualifiers/marketing (reduced, natural, pure, low-fat), drop the qualifier and keep only the ingredient noun if present (e.g., 'reduced cocoa' -> 'cocoa').\n"
#         "- Drop section headers (ingredients, allergen information), numbers/percentages/measurements, preparation/storage notes, and relational/claim phrases (contains, may contain, from, with, minimum, processed in).\n"
#         "- Drop whole sentences/clauses. If a single clear ingredient noun-phrase can be extracted from a clause, output only that noun-phrase.\n"
#         "- Be conservative: if you cannot identify a valid ingredient noun-phrase, drop it.\n"
#         "- Output MUST be a JSON array of strings only. No prose, no code fences.\n"
#         "\n"
#         "Examples (guidance):\n"
#         "INPUT: [\"INGREDIENTS:\", \"Skimmed Milk\", \"Milk Chocolate contains milk solids\", \"4% minimum\", \"Reduced Cocoa\", \"Natural Vanilla Extract\"]\n"
#         "OUTPUT: [\"skimmed milk\", \"milk\", \"cocoa\", \"vanilla extract\"]\n"
#         "\n"
#         f"INPUT: {raw_json}\n"
#         "OUTPUT:"
#     )

#     try:
#         raw = _ollama_chat(prompt, 224, 0.0)
#         s = raw.strip()
#         if s.startswith("```"):
#             s = s.strip("`").strip()
#         arr = json.loads(s[s.find('['):s.rfind(']')+1])
#         if not isinstance(arr, list):
#             return []
#         return _dedupe_preserve([str(x).strip() for x in arr if isinstance(x, str) and str(x).strip()])
#     except Exception as e:
#         log.warning("LLM clean failed: %s", e)
#         return []

# def _llm_split_and_fix(tokens: List[str]) -> List[str]:
#     """
#     Split adjacent distinct ingredients; condense single noisy ones; canonicalize family variants;
#     correct minor typos; dedupe by canonical form. Output a single flat JSON array.
#     """
#     raw_json = json.dumps(tokens, ensure_ascii=False)

#     prompt = (
#         "You are refining ingredient tokens. Return ONE flat JSON array of canonical ingredient tokens.\n"
#         "\n"
#         "Do all of the following:\n"
#         "1) SPLIT: If a token contains two or more distinct ingredient names side-by-side, split them into separate items in order.\n"
#         "   - Example: \"sugar glucose syrup\" -> [\"sugar\", \"glucose syrup\"]\n"
#         "   - Example: \"glucose syrups butter\" -> [\"glucose syrup\", \"butter\"]\n"
#         "2) CONDENSE: If a token is one ingredient but noisy/long, remove relational/qualifier words and reduce to its core noun-phrase (ideally ≤2 words) without changing meaning.\n"
#         "   - Example: \"milk lactose protein from whey\" -> [\"whey protein\"]\n"
#         "   - Example: \"sunflower qul\" -> [\"sunflower oil\"]\n"
#         "3) CANONICALIZE & DEDUPE families: Merge synonymous or hierarchical variants into a single base ingredient when they belong to the same family.\n"
#         "   - Collapse variants like \"skim milk\", \"skimmed milk\", \"milk powder\", \"milk chocolate\", \"chocolate milk\" into \"milk\".\n"
#         "   - Keep distinct families separate (e.g., \"egg powder\" stays distinct from \"milk\").\n"
#         "4) QUALITY: Correct minor spelling errors; do not invent ingredients; drop items if a valid ingredient cannot be recovered.\n"
#         "5) FORMAT: Output MUST be a JSON array of strings only. No prose, no code fences.\n"
#         "\n"
#         "Guidance examples (not exhaustive):\n"
#         "INPUT: [\"skimmed milk powder sugar\", \"glucose syrups butter\", \"reduced cocoa\", \"natural vanilla extract\"]\n"
#         "OUTPUT: [\"milk\", \"sugar\", \"glucose syrup\", \"butter\", \"cocoa\", \"vanilla extract\"]\n"
#         "\n"
#         f"INPUT: {raw_json}\n"
#         "OUTPUT:"
#     )

#     try:
#         raw = _ollama_chat(prompt, 320, 0.0).strip()
#         if raw.startswith("```"):
#             raw = raw.strip("`").strip()
#         arr = json.loads(raw[raw.find('['):raw.rfind(']')+1])
#         if not isinstance(arr, list):
#             return []
#         cleaned = [str(x).strip() for x in arr if isinstance(x, str) and str(x).strip()]
#         return _dedupe_preserve(cleaned)
#     except Exception as e:
#         log.warning("LLM split_or_condense failed: %s", e)
#         return []

# def _llm_verify_one(token: str) -> bool:
#     """
#     Strict boolean gate for ingredient noun-phrases (1–3 words), after canonicalization.
#     """
#     token = (token or "").strip()
#     if not token:
#         return False

#     prompt = (
#         "You will receive ONE token from a food ingredient list. Return EXACTLY one JSON boolean: true or false.\n"
#         "\n"
#         "Return true only if the token is a valid standalone ingredient noun-phrase (food/ingredient/additive head noun with ≤2 modifiers),\n"
#         "such as sugar, milk, glucose syrup, cocoa, vanilla extract, sunflower oil, whey powder, soy/soya, barley extract, egg powder.\n"
#         "Reject headers, numbers/percentages, sentences/clauses, relational/claim phrases (contains, from, with, minimum), adjective-only text, and non-words.\n"
#         "\n"
#         "Examples:\n"
#         "  \"ingredients\" -> false\n"
#         "  \"4% minimum\" -> false\n"
#         "  \"skimmed milk powder sugar\" -> false (two ingredients combined)\n"
#         "  \"milk. natural vanilla extract. chocolate contains milk solids\" -> false (clause)\n"
#         "  \"glucose syrup\" -> true\n"
#         "  \"milk\" -> true\n"
#         "  \"vanilla extract\" -> true\n"
#         "\n"
#         f"INPUT: {json.dumps(token, ensure_ascii=False)}\n"
#         "OUTPUT:"
#     )

#     try:
#         raw = _ollama_chat(prompt, 16, 0.0).strip().lower()
#         if raw.startswith("true"):
#             return True
#         if raw.startswith("false"):
#             return False
#         try:
#             val = json.loads(raw)
#             return bool(val) is True
#         except Exception:
#             return False
#     except Exception:
#         return False

# # -----------------------------------------------------------------------------
# # Orchestration
# # -----------------------------------------------------------------------------
# def _preclean_with_llm_list(raw_items: List[str]) -> List[str]:
#     step1 = _llm_clean(raw_items)
#     step2 = _llm_split_and_fix(step1)
#     verified = [t for t in step2 if _llm_verify_one(t)]
#     cleaned = _dedupe_preserve(verified)

#     log.info("LLM clean tokens: %s", step1)
#     log.info("LLM split_or_condense tokens: %s", step2)
#     log.info("Verified tokens for DB: %s", cleaned)
#     return cleaned

# # -----------------------------------------------------------------------------
# # Public API
# # -----------------------------------------------------------------------------
# def recommend_from_plain_list(ingredients: List[str]) -> Dict[str, Any]:
#     product = ", ".join(ingredients)

#     if not ingredients:
#         return {"product": "", "ingredients": {"safe": [], "avoid": []}}

#     cleaned = _preclean_with_llm_list(ingredients)
#     if not cleaned:
#         return {"product": product, "ingredients": {"safe": [], "avoid": []}}

#     col = _get_mongo_collection()
#     safe: List[str] = []
#     avoid: List[str] = []

#     for name in cleaned:
#         trigs = _lookup_triggers(col, name)
#         if trigs:
#             avoid.append(f"{name} ({', '.join(trigs)})")
#         else:
#             safe.append(name)

#     return RecommendationOutput(
#         product=product,
#         ingredients=IngredientAdvice(safe=safe, avoid=avoid)
#     ).dict()


# # -*- coding: utf-8 -*-
# """
# recommend.py — DB classifier with LLM CLEAN + SPLIT_OR_CONDENSE + VERIFY

# Flow:
#   1) Input: raw OCR list (strings; may include long/glued phrases).
#   2) LLM CLEAN (array → array):
#        - drop headers/numbers/clauses/noise; KEEP only standalone ingredient noun-phrases
#        - if a token mixes an ingredient with relational words, EXTRACT only the ingredient words
#        - be conservative: if uncertain, DROP
#   3) LLM SPLIT_OR_CONDENSE (array → array):
#        - For each element:
#            • if it actually contains MULTIPLE ingredients → split into separate items
#            • else if it is ONE ingredient but long/noisy → strip non-ingredient words and condense to ≤2 words USING ONLY words from the token
#            • fix small misspellings (minor edits), no inventions; if you can't form a valid ingredient noun-phrase → DROP
#        - Outcome: flat array of short tokens (ideally ≤2 words), no junk
#   4) LLM VERIFY (per token):
#        - confirm each token is a standalone ingredient/food/additive noun-phrase; return strict boolean
#        - be conservative; if not clearly an ingredient → false
#   5) DB lookup (ingredient/examples):
#        - triggers found → avoid: "name (t1, t2, ...)"
#        - else → safe: "name"
#   6) Return unchanged frontend shape.

# Env:
#   OLLAMA_BASE_URL, OLLAMA_MODEL
#   MONGO_URI (default: mongodb://localhost:27017/)
#   MONGO_DB  (default: eczema_db)
#   MONGO_COLL(default: ingredients)
#   TINY_RECO_LOGLEVEL (default: INFO)
# """

# from __future__ import annotations

# import json
# import os
# import re
# import sys
# import time
# import subprocess
# from typing import Any, Dict, List, Optional

# import logging
# import requests

# try:
#     from pymongo import MongoClient
# except Exception:
#     MongoClient = None  # type: ignore

# from pydantic import BaseModel, Field

# # -----------------------------------------------------------------------------
# # Logging
# # -----------------------------------------------------------------------------
# LOG_LEVEL = os.getenv("TINY_RECO_LOGLEVEL", "INFO").upper()
# logging.basicConfig(
#     level=LOG_LEVEL,
#     format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
#     stream=sys.stderr,
# )
# log = logging.getLogger("tiny_reco")

# # -----------------------------------------------------------------------------
# # Ollama config
# # -----------------------------------------------------------------------------
# OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
# OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "dolphin-llama3:latest")

# def _ollama_chat(prompt: str, num_predict: int = 256, temperature: float = 0.0) -> str:
#     url = f"{OLLAMA_BASE_URL}/api/chat"
#     payload = {
#         "model": OLLAMA_MODEL,
#         "messages": [{"role": "user", "content": prompt}],
#         "stream": False,
#         "options": {"num_predict": num_predict, "temperature": temperature, "keep_alive": "30m"},
#     }
#     r = requests.post(url, json=payload, timeout=120)
#     r.raise_for_status()
#     data = r.json()
#     return (data.get("message") or {}).get("content", "").strip()

# # -----------------------------------------------------------------------------
# # Mongo config
# # -----------------------------------------------------------------------------
# MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
# MONGO_DB = os.getenv("MONGO_DB", "eczema_db")
# MONGO_COLL = os.getenv("MONGO_COLL", "ingredients")

# def _get_mongo_collection():
#     if MongoClient is None:
#         log.warning("pymongo not available; DB lookups will be skipped.")
#         return None
#     try:
#         client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)  # type: ignore
#         col = client[MONGO_DB][MONGO_COLL]
#         client.admin.command("ping")
#         return col
#     except Exception as e:
#         log.warning("Mongo connection failed: %s", e)
#         return None

# # -----------------------------------------------------------------------------
# # Pydantic models
# # -----------------------------------------------------------------------------
# class IngredientAdvice(BaseModel):
#     safe: List[str] = Field(default_factory=list)
#     avoid: List[str] = Field(default_factory=list)

# class RecommendationOutput(BaseModel):
#     product: str
#     ingredients: IngredientAdvice

# # -----------------------------------------------------------------------------
# # Helpers
# # -----------------------------------------------------------------------------
# def _dedupe_preserve(seq: List[str]) -> List[str]:
#     out, seen = [], set()
#     for x in seq:
#         if x not in seen:
#             seen.add(x)
#             out.append(x)
#     return out

# def _lookup_triggers(col, name: str) -> Optional[List[str]]:
#     """
#     Look up triggers by exact ingredient match first, then by examples.
#     Returns list of triggers or None if not found / no DB.
#     """
#     if col is None:
#         return None
#     key = re.sub(r"\s+", " ", name.strip().lower())
#     try:
#         doc = col.find_one({"ingredient": key}, {"triggers": 1})
#         if doc and isinstance(doc.get("triggers"), list):
#             return [t.strip().lower() for t in doc["triggers"] if isinstance(t, str)]
#         doc = col.find_one({"examples": key}, {"triggers": 1})
#         if doc and isinstance(doc.get("triggers"), list):
#             return [t.strip().lower() for t in doc["triggers"] if isinstance(t, str)]
#     except Exception as e:
#         log.warning("Mongo lookup error for %r: %s", name, e)
#     return None

# # -----------------------------------------------------------------------------
# # LLM passes (CLEAN unchanged; SPLIT + VERIFY prompts rewritten)
# # -----------------------------------------------------------------------------
# def _llm_clean(raw_lines: List[str]) -> List[str]:
#     """
#     LLM Clean: keep only ingredient-like tokens; allow extracting the ingredient
#     noun phrase from within a noisy token. Be conservative; drop if uncertain.
#     """
#     raw_json = json.dumps(raw_lines, ensure_ascii=False)

#     prompt = (
#         "You are an expert at cleaning OCR from packaged-food ingredient lists.\n"
#         "Task: Given a JSON array of raw OCR tokens, return ONLY a JSON array of tokens that are\n"
#         "standalone ingredient names or short ingredient noun-phrases.\n"
#         "\n"
#         "STRICT RULES:\n"
#         "1) KEEP tokens that clearly name a food/ingredient/additive (e.g., 'sugar', 'milk',\n"
#         "   'whey protein', 'vanilla extract', 'soy lecithin', 'cocoa butter', 'skimmed milk powder').\n"
#         "2) If a token includes an ingredient plus extra relational/qualifier words, OUTPUT only the\n"
#         "   ingredient words and discard the rest (e.g., 'from milk' → 'milk').\n"
#         "3) DROP section headers, numbers/percentages/measurements, country-of-origin, storage/prep\n"
#         "   instructions, allergen/disclaimer/relational phrases (e.g., 'contains', 'may contain',\n"
#         "   'with', 'from', 'made of', 'processed in', 'minimum', 'at least'), marketing adjectives\n"
#         "   without a noun head ('natural', 'reduced', etc.), and any garbled/non-word fragments.\n"
#         "4) Be conservative: if you are not confident the token is a valid ingredient noun-phrase,\n"
#         "   DROP it rather than guessing.\n"
#         "5) Output MUST be ONLY a JSON array of strings. No prose. No code fences.\n"
#         "\n"
#         f"REAL INPUT: {raw_json}\n"
#         "REAL OUTPUT:"
#     )

#     try:
#         raw = _ollama_chat(prompt, 192, 0.0)
#         s = raw.strip()
#         if s.startswith("```"):
#             s = s.strip("`").strip()
#         arr = json.loads(s[s.find("["):s.rfind("]")+1])
#         if not isinstance(arr, list):
#             return []
#         return _dedupe_preserve([str(x).strip() for x in arr if isinstance(x, str) and str(x).strip()])
#     except Exception as e:
#         log.warning("LLM clean failed: %s", e)
#         return []

# def _llm_split_and_fix(tokens: List[str]) -> List[str]:
#     """
#     Unified LLM: For EACH input token:
#       - If it contains MULTIPLE ingredients → split into SEPARATE items.
#       - ELSE if it is ONE ingredient but long/noisy → remove non-ingredient words and
#         CONDENSE to ≤2 words USING ONLY words from the token.
#       - Fix minor typos (a few character edits). Do not invent or guess new ingredients.
#       - If you cannot confidently form a valid standalone ingredient noun-phrase → DROP it.
#     Output MUST be a single flat JSON array of strings.
#     """
#     raw_json = json.dumps(tokens, ensure_ascii=False)

#     prompt = (
#         "You will receive a JSON array of tokens from OCR (already roughly cleaned). "
#         "Return ONE flat JSON array of short ingredient tokens.\n"
#         "\n"
#         "Instructions (strict):\n"
#         "- Identify ingredient noun-phrases: a food/ingredient head noun optionally preceded by one or two modifiers. "
#         "- If a token contains two or more distinct ingredient noun-phrases, ALWAYS split them into separate items in the original order. "
#         "Do not glue multiple ingredients into a single token.\n"
#         "- Only condense when the token clearly describes ONE ingredient: remove relational/qualifier words and keep the ingredient noun-phrase; "
#         "reduce to ≤2 words when possible without changing meaning. Never condense by merging two different ingredients.\n"
#         "- Correct minor spelling errors. Do not invent ingredients. If a well-formed ingredient noun-phrase cannot be recovered confidently, drop that item.\n"
#         "- Output MUST be a JSON array of strings only (no prose, no code fences).\n"
#         "\n"
#         f"INPUT:\n{raw_json}\n\nOUTPUT:"
#     )

#     try:
#         raw = _ollama_chat(prompt, 256, 0.0).strip()
#         if raw.startswith("```"):
#             raw = raw.strip("`").strip()
#         arr = json.loads(raw[raw.find("["):raw.rfind("]")+1])
#         if not isinstance(arr, list):
#             return []
#         cleaned = [str(x).strip() for x in arr if isinstance(x, str) and str(x).strip()]
#         return _dedupe_preserve(cleaned)
#     except Exception as e:
#         log.warning("LLM split_or_condense failed: %s", e)
#         return []

# def _llm_verify_one(token: str) -> bool:
#     """
#     Confirm token is a standalone ingredient/food/additive noun-phrase.
#     Output MUST be a single JSON boolean: true/false.
#     """
#     token = (token or "").strip()
#     if not token:
#         return False

#     prompt = (
#         "You will receive ONE token from a packaged-food ingredient list. "
#         "Return EXACTLY one JSON boolean (true/false) with no explanation.\n"
#         "\n"
#         "Return true if and only if the token is a well-formed standalone ingredient noun-phrase "
#         "(a food/ingredient/additive head noun, optionally with one or two modifiers). "
#         "Multi-word compound ingredients are valid (e.g., modifier + head), including dairy/grain/plant compounds, powders, oils, extracts, proteins, starches, leaveners, emulsifiers, and similar classes.\n"
#         "\n"
#         "Return false if the token is a section header; a relational/qualifier phrase; adjective-only or marketing-only text; "
#         "a fragment or non-word; or not clearly an ingredient noun-phrase. "
#         "Do not reject solely because the ingredient has multiple words if it is well-formed.\n"
#         "\n"
#         f"INPUT: {json.dumps(token, ensure_ascii=False)}\n"
#         "OUTPUT:"
#     )

#     try:
#         raw = _llama_chat_result = _ollama_chat(prompt, 8, 0.0).strip().lower()
#         if raw.startswith("true"):
#             return True
#         if raw.startswith("false"):
#             return False
#         # Fallback JSON parse
#         try:
#             val = json.loads(raw)
#             return bool(val) is True
#         except Exception:
#             return False
#     except Exception:
#         return False

# # -----------------------------------------------------------------------------
# # Orchestration
# # -----------------------------------------------------------------------------
# def _preclean_with_llm_list(raw_items: List[str]) -> List[str]:
#     """
#     1) LLM clean (drop junk / extract ingredient noun-phrases)
#     2) LLM split_or_condense (split multi-ingredients OR condense one ingredient, drop uncertain)
#     3) LLM verify per token (strict boolean)
#     """
#     step1 = _llm_clean(raw_items)
#     step2 = _llm_split_and_fix(step1)
#     verified = [t for t in step2 if _llm_verify_one(t)]
#     cleaned = _dedupe_preserve(verified)

#     # DEBUG snapshots before DB
#     log.info("LLM clean tokens: %s", step1)
#     log.info("LLM split_or_condense tokens: %s", step2)
#     log.info("Verified tokens for DB: %s", cleaned)
#     return cleaned

# # -----------------------------------------------------------------------------
# # Public API
# # -----------------------------------------------------------------------------
# def recommend_from_plain_list(ingredients: List[str]) -> Dict[str, Any]:
#     """
#     1) LLM clean → split_or_condense → verify → cleaned tokens.
#     2) DB lookup per token:
#          - triggers found → avoid: \"name (t1, t2, ...)\"
#          - else → safe: \"name\"
#     3) Return frontend shape exactly.
#     """
#     product = ", ".join(ingredients)  # keep original messy list as-is for UI

#     if not ingredients:
#         return {"product": "", "ingredients": {"safe": [], "avoid": []}}

#     cleaned = _preclean_with_llm_list(ingredients)
#     if not cleaned:
#         return {"product": product, "ingredients": {"safe": [], "avoid": []}}

#     col = _get_mongo_collection()
#     safe: List[str] = []
#     avoid: List[str] = []

#     for name in cleaned:
#         trigs = _lookup_triggers(col, name)
#         if trigs:
#             avoid.append(f"{name} ({', '.join(trigs)})")
#         else:
#             safe.append(name)

#     return RecommendationOutput(
#         product=product,
#         ingredients=IngredientAdvice(safe=safe, avoid=avoid)
#     ).dict()



# # -*- coding: utf-8 -*-
# """
# recommend.py — DB classifier with LLM split+spell-fix on raw OCR

# Flow:
#   1) Input: a PLAIN LIST of raw OCR strings (may be long, glued, comma/semicolon separated).
#   2) LLM SPLIT+FIX (single pass):
#        - Iterate through the raw text; keep only foods/drinks/ingredients (e.g., milk counts).
#        - Split combined sequences into separate items.
#        - Apply minimal spelling corrections (small edits).
#        - DROP headers/clauses/numbers/noise (e.g., 'ingredients', 'contains ...', '%', 'minimum').
#        - RETURN a JSON array of ingredient tokens (strings).
#   3) Optional LLM VERIFY per token (boolean) to reduce hallucinations.
#   4) DB lookup for each surviving token:
#        - triggers found (ingredient/examples) → avoid: "name (t1, t2, ...)"
#        - else → safe: "name"
#   5) Return unchanged frontend shape:
#        {
#          "product": "<original list joined by commas (uncleaned)>",
#          "ingredients": {"safe": [...], "avoid": [...]}
#        }

# Env:
#   OLLAMA_BASE_URL, OLLAMA_MODEL, WARMUP_OLLAMA
#   MONGO_URI (default: mongodb://localhost:27017/)
#   MONGO_DB  (default: eczema_db)
#   MONGO_COLL(default: ingredients)
#   TINY_RECO_LOGLEVEL (default: INFO)
# """

# from __future__ import annotations

# import json
# import os
# import re
# import sys
# import time
# import subprocess
# from typing import Any, Dict, List, Optional

# import logging
# import requests

# try:
#     from pymongo import MongoClient
# except Exception:
#     MongoClient = None  # type: ignore

# from pydantic import BaseModel, Field

# # -----------------------------------------------------------------------------
# # Logging
# # -----------------------------------------------------------------------------
# LOG_LEVEL = os.getenv("TINY_RECO_LOGLEVEL", "INFO").upper()
# logging.basicConfig(
#     level=LOG_LEVEL,
#     format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
#     stream=sys.stderr,
# )
# log = logging.getLogger("tiny_reco")

# # -----------------------------------------------------------------------------
# # Ollama config (used for split+fix and verify)
# # -----------------------------------------------------------------------------
# OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
# OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "dolphin-llama3:latest")
# _WARMUP_OLLAMA = os.getenv("WARMUP_OLLAMA", "1") != "0"

# def _warmup_dolphin() -> None:
#     if not _WARMUP_OLLAMA:
#         return
#     try:
#         subprocess.Popen(
#             ["ollama", "run", OLLAMA_MODEL, "warmup"],
#             stdout=subprocess.DEVNULL,
#             stderr=subprocess.DEVNULL,
#         )
#         log.info("Warmup requested: %s", OLLAMA_MODEL)
#     except Exception as e:
#         log.warning("Warmup failed (non-fatal): %s", e)

# _warmup_dolphin()

# # -----------------------------------------------------------------------------
# # Mongo config
# # -----------------------------------------------------------------------------
# MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
# MONGO_DB = os.getenv("MONGO_DB", "eczema_db")
# MONGO_COLL = os.getenv("MONGO_COLL", "ingredients")

# def _get_mongo_collection():
#     if MongoClient is None:
#         log.warning("pymongo not available; DB lookups will be skipped.")
#         return None
#     try:
#         client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)  # type: ignore
#         col = client[MONGO_DB][MONGO_COLL]
#         client.admin.command("ping")
#         return col
#     except Exception as e:
#         log.warning("Mongo connection failed (%s). DB lookups will be skipped.", e)
#         return None

# # -----------------------------------------------------------------------------
# # Pydantic models
# # -----------------------------------------------------------------------------
# class IngredientAdvice(BaseModel):
#     safe: List[str] = Field(default_factory=list)
#     avoid: List[str] = Field(default_factory=list)

#     class Config:
#         extra = "ignore"
#         allow_population_by_field_name = True
#         populate_by_name = True

# class RecommendationOutput(BaseModel):
#     product: str
#     ingredients: IngredientAdvice

# # -----------------------------------------------------------------------------
# # Utilities
# # -----------------------------------------------------------------------------
# def _norm(s: str) -> str:
#     return re.sub(r"\s+", " ", (s or "").strip().lower())

# def _dedupe_preserve(seq: List[str]) -> List[str]:
#     out: List[str] = []
#     seen = set()
#     for x in seq:
#         if x not in seen:
#             seen.add(x)
#             out.append(x)
#     return out

# def _lookup_triggers(col, name: str) -> Optional[List[str]]:
#     """
#     Look up triggers by exact ingredient match first, then by examples.
#     Returns list of triggers or None if not found / no DB.
#     """
#     if col is None:
#         return None
#     key = _norm(name)
#     try:
#         doc = col.find_one({"ingredient": key}, {"triggers": 1})
#         if doc and isinstance(doc.get("triggers"), list):
#             return sorted({t.strip().lower() for t in doc["triggers"] if isinstance(t, str)})
#         doc = col.find_one({"examples": key}, {"triggers": 1})
#         if doc and isinstance(doc.get("triggers"), list):
#             return sorted({t.strip().lower() for t in doc["triggers"] if isinstance(t, str)})
#     except Exception as e:
#         log.warning("Mongo lookup error for %r: %s", name, e)
#     return None

# # -----------------------------------------------------------------------------
# # Core LLM calls
# # -----------------------------------------------------------------------------
# def _ollama_chat(prompt: str, num_predict: int = 256, temperature: float = 0.0) -> str:
#     url = f"{OLLAMA_BASE_URL}/api/chat"
#     payload = {
#         "model": OLLAMA_MODEL,
#         "messages": [{"role": "user", "content": prompt}],
#         "stream": False,
#         "options": {
#             "num_predict": int(num_predict),
#             "temperature": float(temperature),
#             "keep_alive": "30m",
#         },
#     }
#     r = requests.post(url, json=payload, timeout=120)
#     r.raise_for_status()
#     data = r.json()
#     msg = data.get("message") or {}
#     return (msg.get("content") or "").strip()

# def _llm_split_and_fix(raw_lines: List[str]) -> List[str]:
#     """
#     One-pass LLM: given the RAW OCR list (array of strings), return a JSON array
#     of *only* ingredient/food/drink tokens:
#       - split combined sequences into separate items
#       - fix minor misspellings
#       - drop headers/clauses/numbers/noise
#       - do not invent new items
#     """
#     # Join as JSON array for precise grounding
#     raw_json = json.dumps(raw_lines, ensure_ascii=False)

#     prompt = (
#         "You will receive a JSON array of RAW OCR lines from an ingredient label.\n"
#         "Task: produce ONLY a JSON array of CLEAN ingredient/food/drink tokens.\n"
#         "Rules:\n"
#         "1) Iterate through the text; KEEP ONLY items that are foods/drinks/ingredients "
#         "(e.g., 'milk','butter','sugar','vanilla extract','skimmed milk powder').\n"
#         "2) You MAY SPLIT combined sequences into separate items (e.g., "
#         "'sugar glucose syrup skimmed milk powde' → ['sugar','glucose syrup','skimmed milk powder']).\n"
#         "3) Apply MINIMAL spelling correction; keep meaning; prefer canonical names; small edits only.\n"
#         "4) DROP headers/clauses/noise and sentences such as 'ingredients', 'contains ...', percentages, "
#         "and anything that is not a standalone ingredient/food term.\n"
#         "5) DO NOT invent items that are not obviously present; if unsure, drop.\n"
#         "6) Output MUST be ONLY a JSON array of strings. No prose. No code fences.\n\n"
#         f"INPUT:\n{raw_json}\n\n"
#         "OUTPUT:\n"
#     )

#     try:
#         raw = _ollama_chat(prompt, num_predict=256, temperature=0.0).strip()
#         if raw.startswith("```"):
#             raw = raw.strip("`").strip()
#         # Try to parse JSON array
#         try:
#             arr = json.loads(raw)
#         except Exception:
#             # try to find the first [ ... ] region
#             start, end = raw.find("["), raw.rfind("]") + 1
#             s = raw[start:end] if start != -1 and end != -1 else "[]"
#             arr = json.loads(s)
#         if not isinstance(arr, list):
#             return []
#         # keep only clean non-empty strings; preserve order, dedupe
#         cleaned = [str(x).strip() for x in arr if isinstance(x, str) and str(x).strip()]
#         return _dedupe_preserve(cleaned)
#     except Exception as e:
#         log.warning("LLM split+fix failed (%s); returning empty.", e)
#         return []

# def _llm_verify_one(token: str) -> bool:
#     """
#     Second pass: confirm token is indeed a food/ingredient (boolean).
#     """
#     token = (token or "").strip()
#     if not token:
#         return False

#     prompt = (
#         "You will receive ONE token.\n"
#         "Answer with exactly ONE JSON boolean:\n"
#         "  true  → token is an ingredient/food term (e.g., 'milk','butter','sugar','vanilla extract','skimmed milk powder')\n"
#         "  false → otherwise (headers like 'ingredients', clauses like 'contains milk', or noise)\n"
#         "No prose. No code fences.\n\n"
#         f"INPUT: {json.dumps(token, ensure_ascii=False)}\n"
#         "OUTPUT:"
#     )
#     try:
#         raw = _ollama_chat(prompt, num_predict=8, temperature=0.0).strip()
#         if raw.startswith("```"):
#             raw = raw.strip("`").strip()
#         if raw.lower().startswith("true"):
#             return True
#         if raw.lower().startswith("false"):
#             return False
#         try:
#             val = json.loads(raw.lower())
#             return bool(val) is True
#         except Exception:
#             return False
#     except Exception as e:
#         log.warning("LLM verify failed for %r: %s", token, e)
#         return False

# # -----------------------------------------------------------------------------
# # Orchestration
# # -----------------------------------------------------------------------------
# def _preclean_with_llm_list(raw_items: List[str]) -> List[str]:
#     """
#     Give the RAW OCR list to LLM to split+fix; then run a per-token verify pass.
#     """
#     # 1) LLM split+fix
#     tokens = _llm_split_and_fix(raw_items)

#     # 2) Verify tokens individually (drop if LLM says false)
#     verified: List[str] = []
#     for t in tokens:
#         if _llm_verify_one(t):
#             verified.append(t)

#     cleaned = _dedupe_preserve(verified)

#     # DEBUG: show what we will send to MongoDB
#     log.info("LLM split+fix tokens: %s", tokens)
#     log.info("Verified tokens for DB lookup: %s", cleaned)
#     return cleaned

# # -----------------------------------------------------------------------------
# # Public API — DB ONLY classification (LLM used for split+fix + verify)
# # -----------------------------------------------------------------------------
# def recommend_from_plain_list(ingredients: List[str]) -> Dict[str, Any]:
#     """
#     1) LLM split+fix+verify → cleaned tokens.
#     2) DB lookup per token:
#          - triggers found → avoid: "name (t1, t2, ...)"
#          - else → safe: "name"
#     3) Return frontend shape exactly.
#     """
#     product = ", ".join(ingredients)  # keep original messy list

#     if not ingredients:
#         return {"product": "", "ingredients": {"safe": [], "avoid": []}}

#     cleaned = _preclean_with_llm_list(ingredients)
#     if not cleaned:
#         return {"product": product, "ingredients": {"safe": [], "avoid": []}}

#     col = _get_mongo_collection()
#     safe: List[str] = []
#     avoid: List[str] = []

#     for name in cleaned:
#         trigs = _lookup_triggers(col, name)
#         if trigs:
#             avoid.append(f"{name} ({', '.join(trigs)})")
#         else:
#             safe.append(name)

#     advice = IngredientAdvice(safe=safe, avoid=avoid)
#     output = RecommendationOutput(product=product, ingredients=advice)
#     return output.dict()
