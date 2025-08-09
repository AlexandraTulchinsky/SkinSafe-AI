from __future__ import annotations

import base64
import io
import logging
import time
import uuid
from typing import Tuple

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

from ingredient_list import extract_ingredients_label
from classify_item import classify_image_with_clip
from identify_food import identify_food_items, analyse_triggers
from recommend import recommend_ingredients, recommend_from_plain_list
from search_barcode import extract_ingredients

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("diet-ai")

app = FastAPI(title="Diet AI Analyzer", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    image: str = Field(..., description="Image as a Data URL (e.g., data:image/jpeg;base64,...)")

CLASS_FOOD = "a photo of a food item"
CLASS_LIST = "a photo of ingredient list"
CLASS_BARCODE = "barcode"


def _decode_data_url(data_url: str) -> Image.Image:
    """
    Minimal, robust Data URL decoder for images; returns a PIL RGB image.
    Logic matches the original flow (no transformations beyond RGB convert).
    """
    _hdr, b64 = data_url.split(",", 1)
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    return img


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1_000)


def _to_cv2_rgb_to_bgr(img: Image.Image) -> np.ndarray:
    """Convert PIL RGB → OpenCV BGR. Only used in branches that need OpenCV."""
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


@app.post("/analyze")
async def analyze_image(req: AnalyzeRequest):
    """
      1) Classify image with CLIP
      2) If food → identify items → analyze triggers → LLM recommend
      3) If barcode → OpenCV → lookup ingredients → LLM recommend
      4) If ingredient list photo → OpenCV → OCR → LLM recommend
    All returns include processing_time_ms.
    """
    t0 = time.perf_counter()
    req_id = uuid.uuid4().hex[:8]

    try:
        img = _decode_data_url(req.image)

        result = classify_image_with_clip(img)
        log.info("[%s] CLIP classification: %s", req_id, result)

        # ── (1) Food photo branch ───────────────────────────
        if result == CLASS_FOOD:
            items = identify_food_items(img)
            log.info("[%s] Food items: %s", req_id, items)

            triggers = analyse_triggers(items)
            log.info("[%s] Triggers:   %s", req_id, triggers)

            llm_out = recommend_ingredients(triggers) 
            return {
                "success": True,
                **llm_out,
                "processing_time_ms": _ms(t0),
            }

        # ── (2) Barcode branch ──────────────────────────────
        if result == CLASS_BARCODE:
            opencv_img = _to_cv2_rgb_to_bgr(img)
            ingredients = extract_ingredients(opencv_img)
            log.info("[%s] Ingredients pulled: %s", req_id, ingredients)

            llm_out = recommend_from_plain_list(ingredients)
            log.info("[%s] LLM output: %s", req_id, llm_out)

            return {
                "success": True,
                **llm_out,
                "processing_time_ms": _ms(t0),
            }

        # ── (3) Ingredient-list photo branch ────────────────
        if result == CLASS_LIST:
            opencv_img = _to_cv2_rgb_to_bgr(img)
            ingredients = extract_ingredients_label(opencv_img)
            log.info("[%s] Ingredients pulled: %s", req_id, ingredients)

            llm_out = recommend_from_plain_list(ingredients)
            log.info("[%s] LLM output: %s", req_id, llm_out)

            return {
                "success": True,
                **llm_out,
                "processing_time_ms": _ms(t0),
            }

        return {
            "success": True,
            **llm_out, 
            "processing_time_ms": _ms(t0),
        }

    except Exception as exc:
        log.error("[%s] /analyze failed: %s", req_id, exc, exc_info=True)
        return {"success": False, "error": str(exc)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

