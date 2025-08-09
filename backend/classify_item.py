from __future__ import annotations

import logging
from typing import List

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

log = logging.getLogger("clip_classifier")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

_MODEL_ID = "openai/clip-vit-base-patch32"
_LABELS: List[str] = [
    "a photo of a food item",
    "a photo of ingredient list",
    "barcode",
    "a person",
    "other object",
]

_model: CLIPModel | None = None
_processor: CLIPProcessor | None = None

def _load_clip() -> tuple[CLIPProcessor, CLIPModel]:
    """
    Load the CLIP model and processor once (CPU-only).
    """
    global _model, _processor
    if _model is None or _processor is None:
        log.info("Loading CLIP model: %s", _MODEL_ID)
        _model = CLIPModel.from_pretrained(_MODEL_ID)
        _processor = CLIPProcessor.from_pretrained(_MODEL_ID)
        log.info("CLIP model loaded successfully.")
    return _processor, _model

def classify_image_with_clip(image: Image.Image) -> str:

    processor, model = _load_clip()

    inputs = processor(text=_LABELS, images=image, return_tensors="pt", padding=True)
    with torch.inference_mode():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)
        predicted_label = _LABELS[probs.argmax()]
        log.debug("CLIP probabilities: %s", probs.tolist())

    return predicted_label
