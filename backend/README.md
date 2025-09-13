## Ezeczema Analyzer — Backend

FastAPI service that takes one image and returns ingredient advice (`safe` vs `avoid`) for eczema-prone users.

### How it works 
CLIP gates the image type (food vs ingredients panel vs barcode). For panels/barcodes, YOLO crops the region; barcodes are looked up on Open Food Facts, panels are OCR’d (EasyOCR → Tesseract → vision fallback). For food photos or poor OCR, a vision LLM lists visible foods. Tokens are cleaned by an LLM and classified against a local MongoDB of triggers into `safe` and `avoid`.

### Run
1) Requirements: Python 3.10+, Ollama (`dolphin-llama3:latest` and `llama3.2-vision:latest`), optional Tesseract, optional MongoDB.
2) Install:
```bash
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r ..\requirements.txt
```
3) YOLO: set `MODEL_PATH` in `detect_and_crop.py` to your YOLOv8 weights (`best.pt`).
4) Start API:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### API
- POST `/analyze`
  - Body: `{ "image": "data:image/jpeg;base64,<...>" }`
  - Response (keys): `success`, `detected_class`, `ingredients.safe`, `ingredients.avoid`, `processing_time_ms`, `stage`

### Config (env vars)
- `OLLAMA_BASE_URL` (default `http://localhost:11434`), `OLLAMA_MODEL` (`dolphin-llama3:latest`), `OLLAMA_VISION_MODEL` (`llama3.2-vision:latest`)
- `MONGO_URI` (`mongodb://localhost:27017/`), `MONGO_DB` (`eczema_db`), `MONGO_COLL` (`ingredients`)
- OCR tuning: `OCR_MIN_SIDE`, `OCR_MIN_GOOD_TOK`, `OCR_TESS_TIMEOUT_MS`

### Seed Mongo (optional)
```bash
python seed_mongo.py --json ..\cross_ref_food.json --db eczema_db --collection ingredients --drop --sample 3
```

### Notes
- Debug crops are saved to `backend/debug_crops/`.
- EasyOCR GPU optional; change to `gpu=False` in `ingredient_list.py` if needed.
- First run may download model weights (transformers).


