# main.py
from __future__ import annotations

import base64
import io
import logging
import os
import re
import time
import uuid
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

from classify_item import classify_image_with_clip

from detect_and_crop import detect_and_crop
from identify_food import identify_food_items
from ingredient_list import extract_ingredients_label
from recommend import recommend_from_plain_list
from search_barcode import extract_ingredients

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ezeczema-ai")

app = FastAPI(title="Ezeczema Analyzer", version="1.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASS_BARCODE = "barcode"
CLASS_LIST = "ingredients"
CLASS_FOOD = "food"
CLASS_NOT_FOOD = "not_food"

CLIP_LABEL_COOKED = "photo of cooked food on a plate"
CLIP_LABEL_PANEL = "ingredients text panel on food packaging"
CLIP_LABEL_BARCODE = "black and white barcode on packaging"
CLIP_LABEL_OTHER = "other type of packaging or logo"

class AnalyzeRequest(BaseModel):
    image: str = Field(..., description="Image as a Data URL (e.g., data:image/jpeg;base64,...)")

# Utils
def _decode_data_url(data_url: str) -> Image.Image:
    """Decode base64 Data URL to a PIL RGB image."""
    _hdr, b64 = data_url.split(",", 1)
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    return img

def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1_000)

def _to_cv2_rgb_to_bgr(img: Image.Image) -> np.ndarray:
    """Convert PIL RGB → OpenCV BGR (for YOLO/OpenCV)."""
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def _tokens_look_valid(tokens: list[str]) -> bool:
    """
    Require at least 6 tokens that look like real words (>=3 letters).
    Prevents passing OCR garbage (from photos) to the recommender.
    """
    return sum(1 for t in tokens if re.search(r"[A-Za-z]{3,}", t)) >= 6

# Route
@app.post("/analyze")
async def analyze_image(req: AnalyzeRequest):
    t0 = time.perf_counter()
    req_id = uuid.uuid4().hex[:8]

    try:
        # 0) Decode input
        img = _decode_data_url(req.image)

        # 0.a) CLIP gate — decide panel/barcode vs everything else
        clip_t0 = time.perf_counter()
        pred_label = classify_image_with_clip(img)
        log.info("[%s] CLIP predicted: %s (t=%d ms)", req_id, pred_label, _ms(clip_t0))

        is_panel_or_barcode = pred_label in (CLIP_LABEL_PANEL, CLIP_LABEL_BARCODE)

        if not is_panel_or_barcode:
            # Treat as non-panel image → go straight to Vision
            food_items = identify_food_items(img)
            llm_out = recommend_from_plain_list(food_items)
            log.info("[%s] Response: %s", req_id, llm_out)
            return {
                "success": True,
                "detected_class": CLASS_FOOD,
                **llm_out,
                "processing_time_ms": _ms(t0),
                "raw_food_items": food_items,
                "stage": "clip-other->vision",
                "clip_label": pred_label,
            }

        # 1) Panel or barcode (per CLIP) → try YOLO to find precise crop(s)
        opencv_img = _to_cv2_rgb_to_bgr(img)
        crops = detect_and_crop(opencv_img, target_classes=[CLASS_BARCODE, CLASS_LIST])

        if crops:
            # Support either (cls, crop) or (cls, crop, conf)
            rec = crops[0]
            if len(rec) == 3:
                cls, crop, conf = rec
                try:
                    conf = float(conf)
                except Exception:
                    conf = None
            else:
                cls, crop = rec
                conf = None

            if crop is None or getattr(crop, "size", 0) == 0:
                log.warning("[%s] Empty crop returned for class=%s", req_id, cls)
                return {"success": False, "error": "Empty crop", "processing_time_ms": _ms(t0)}

            # Save crop for debugging (best-effort)
            out_dir = os.path.join(os.getcwd(), "debug_crops")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"crop_{cls}_{req_id}.jpg")
            try:
                cv2.imwrite(out_path, crop)
            except Exception as e:
                log.debug("[%s] Failed to save crop for debug: %s", req_id, e)

            # 1.a) Barcode branch
            if cls == CLASS_BARCODE:
                ingredients = extract_ingredients(crop)
                if not ingredients:  # fallback to OCR if lookup failed
                    ingredients = extract_ingredients_label(crop)
                llm_out = recommend_from_plain_list(ingredients)
                log.info("[%s] Response: %s", req_id, llm_out)
                return {
                    "success": True,
                    "detected_class": cls,
                    **llm_out,
                    "processing_time_ms": _ms(t0),
                    "stage": "clip-panel->yolo->barcode",
                    "clip_label": pred_label,
                }

            # 1.b) Ingredient panel branch
            if cls == CLASS_LIST:
                # (Optional) distrust weak 'ingredients' detections
                if conf is not None and conf < 0.95:
                    log.info("[%s] YOLO ingredients conf=%.2f < 0.95; using Vision", req_id, conf)
                    food_items = identify_food_items(img)
                    llm_out = recommend_from_plain_list(food_items)
                    log.info("[%s] Response: %s", req_id, llm_out)
                    return {
                        "success": True,
                        "detected_class": CLASS_FOOD,
                        **llm_out,
                        "processing_time_ms": _ms(t0),
                        "raw_food_items": food_items,
                        "stage": "clip-panel->yolo-lowconf->vision",
                        "clip_label": pred_label,
                    }

                tokens = extract_ingredients_label(crop)

                # Quality gate: if OCR looks like junk → treat as food photo
                if not _tokens_look_valid(tokens):
                    log.info("[%s] OCR looked poor; falling back to Vision", req_id)
                    food_items = identify_food_items(img)
                    llm_out = recommend_from_plain_list(food_items)
                    log.info("[%s] Response: %s", req_id, llm_out)
                    return {
                        "success": True,
                        "detected_class": CLASS_FOOD,
                        **llm_out,
                        "processing_time_ms": _ms(t0),
                        "raw_food_items": food_items,
                        "stage": "clip-panel->yolo->ocr-bad->vision",
                        "clip_label": pred_label,
                    }

                # OCR looked good → proceed
                llm_out = recommend_from_plain_list(tokens)
                log.info("[%s] Response: %s", req_id, llm_out)
                return {
                    "success": True,
                    "detected_class": cls,
                    **llm_out,
                    "processing_time_ms": _ms(t0),
                    "stage": "clip-panel->yolo->ocr->classify",
                    "clip_label": pred_label,
                }

        # 2) CLIP said panel/barcode but YOLO found nothing → Vision fallback
        food_items = identify_food_items(img)
        log.info("[%s] Food items: %s", req_id, food_items)
        if food_items:
            llm_out = recommend_from_plain_list(food_items)
            log.info("[%s] Response: %s", req_id, llm_out)
            return {
                "success": True,
                "detected_class": CLASS_FOOD,
                 **llm_out,
                "processing_time_ms": _ms(t0),
                "raw_food_items": food_items,
                "stage": "clip-panel->no-panels->vision",
                "clip_label": pred_label,
            }

        # 3) Nothing recognized
        return {
            "success": True,
            "detected_class": CLASS_NOT_FOOD,
            "ingredients": [],
            "processing_time_ms": _ms(t0),
            "stage": "unrecognized",
            "clip_label": pred_label,
        }

    except Exception as exc:
        log.error("[%s] /analyze failed: %s", req_id, exc, exc_info=True)
        return {"success": False, "error": str(exc), "processing_time_ms": _ms(t0)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

