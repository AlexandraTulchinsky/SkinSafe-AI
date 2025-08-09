from __future__ import annotations

import os

os.environ.setdefault(
    "OPENAI_API_KEY",
)

import openai
client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

os.environ["PYTORCH_SDP_DISABLE_FLASH_ATTENTION"] = "1"
os.environ["PYTORCH_SDP_DISABLE_MEM_EFFICIENT_ATTENTION"] = "1"

import json
import re
import sys
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import torch
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    logging as hf_logging,
)

import logging

LOG_LEVEL = os.getenv("TINY_RECO_LOGLEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("tiny_reco")
if LOG_LEVEL != "DEBUG":
    hf_logging.set_verbosity_error()

_MODEL_ID = "gpt-4o"

@lru_cache(maxsize=1)
def _load_llm() -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Download GPT-4o and keep it on the CPU in float32. (Unused here; retained for parity.)"""
    log.info("Loading GPT-4o (%s)…", _MODEL_ID)
    kwargs = {"device_map": "cpu", "torch_dtype": torch.float32, "low_cpu_mem_usage": True}
    model = AutoModelForCausalLM.from_pretrained(_MODEL_ID, **kwargs)
    tok = AutoTokenizer.from_pretrained(_MODEL_ID)
    tok.pad_token = tok.pad_token or tok.eos_token
    log.info("GPT-4o loaded successfully (CPU).")
    return tok, model

class IngredientAdvice(BaseModel):
    safe: List[str] = Field(default_factory=list)
    avoid: List[str] = Field(default_factory=list)

    class Config:
        extra = "ignore"
        allow_population_by_field_name = True
        populate_by_name = True

class RecommendationOutput(BaseModel):
    product: str
    ingredients: IngredientAdvice

def _prompt(triggers: Dict[str, Optional[List[str]]]) -> str:
    """
    Build an instruction for GPT-4o.

    • If cats is a list      → we already know the trigger categories.
    • If cats is []          → explicitly “no known triggers”.
    • If cats is None        → model must decide whether it triggers eczema.
    """
    ing_lines: List[str] = []
    for ing, cats in triggers.items():
        if cats is None:                   
            ing_lines.append(f"- {ing}")
        else:                              
            label = ", ".join(cats) if cats else "none"
            ing_lines.append(f"- {ing}: {label}")

    ing_block = "\n".join(ing_lines) or "- none"

    return (
        "You are a meticulous dietary-safety assistant.\n\n"
        f"Here is a list of ingredients:\n{ing_block}\n\n"
        "For each ingredient decide if it can trigger eczema.\n"
        "Return exactly **one JSON object** with two keys:\n"
        "• `safe`  – ingredients that do **NOT** trigger eczema\n"
        "• `avoid` – strings like \"ingredient (trigger1, trigger2, …)\" "
        "for those that **DO** trigger eczema.\n\n"
        "**RULES**:\n"
        "1. If an ingredient triggers eczema → put it in `avoid` and list the "
        "trigger categories in parentheses.\n"
        "2. Ingredients without triggers → put them in `safe`.\n"
        "3. Output only the JSON – no prose, no back-ticks, no comments.\n\n"
        'Example:\n'
        '{"safe":["water"],'
        '"avoid":["milk (dairy)","tomato (histamine, salicylate)"]}\n'
    )

_JSON_TAG_RE = re.compile(r"<json>(.*?)</json>", flags=re.S)

def _now_ms() -> int:
    return int(time.time() * 1000)

def _extract_json_payload(raw_txt: str) -> str:
    """
    Extract a JSON object from raw LLM text.
    Behavior is identical to the original:
      1) try <json>...</json>
      2) else take substring from first '{' to last '}'
      3) else return empty string
    """
    m = _JSON_TAG_RE.search(raw_txt)
    if m:
        return m.group(1)

    start, end = raw_txt.find("{"), raw_txt.rfind("}") + 1
    return raw_txt[start:end] if start != -1 and end != -1 else ""

def _normalize_advice_keys(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Lower-case top-level keys to match Pydantic fields (same as original intent)."""
    return {str(k).lower(): v for k, v in obj.items()}

def recommend_ingredients(triggers: Dict[str, Optional[List[str]]]) -> Dict[str, Any]:
    """
    Args
    ----
    triggers: mapping ingredient -> list of trigger categories
              (or None to force the model to decide)

    Returns
    -------
    {
      "product": str,
      "ingredients": {"safe": [...], "avoid": [...]},
    }
    """
    if not triggers:
        return {"product": "", "ingredients": {"safe": [], "avoid": []}}

    product = ", ".join(triggers.keys())

    prompt = _prompt(triggers)
    log.debug("Prompt:\n%s", prompt)

    raw_txt = ""
    try:
        t0_ms = _now_ms()
        rsp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},  
            temperature=0,                        
            max_tokens=256,
        )
        raw_txt = rsp.choices[0].message.content
        log.debug("Raw model output (%d ms):\n%s", _now_ms() - t0_ms, raw_txt)

        if raw_txt.startswith("```") and raw_txt.endswith("```"):
            lines = raw_txt.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw_txt = "\n".join(lines).strip()
            log.debug("After stripping fences:\n%s", raw_txt)

        payload = _extract_json_payload(raw_txt)
        if not payload.strip().startswith("{"):
            raise ValueError(f"No JSON payload found; extracted payload: {payload!r}")

        advice_dict = json.loads(payload)
        advice = IngredientAdvice(**_normalize_advice_keys(advice_dict))

    except Exception as exc:
        log.error("JSON parse failed: %s", exc)
        advice = IngredientAdvice(
            safe=[],
            avoid=[f"{ing} (unknown)" for ing in triggers.keys()],
        )

    output = RecommendationOutput(product=product, ingredients=advice)
    log.debug("Final response: %s", output.dict())
    return output.dict()

def recommend_from_plain_list(ingredients: List[str]) -> Dict[str, Any]:
    """
    Same output as `recommend_ingredients`, but you pass a *flat* list such as
        ["water", "salicylic acid", "butylene glycol", …]

    Behaviour
    ---------
    • Builds a dummy trigger-map {ing: [] …}.
      That tells GPT-4o: “no pre-known triggers – you decide which ones are safe
      and which ones to avoid.”
    • Calls `recommend_ingredients()` internally and returns its JSON.
    """
    if not ingredients:
        return {"product": "", "ingredients": {"safe": [], "avoid": []}}

    triggers: Dict[str, Optional[List[str]]] = {ing: None for ing in ingredients}
    return recommend_ingredients(triggers)
