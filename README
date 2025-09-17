# SkinSafe AI 🧴🤖

Instantly analyze product ingredient lists with **multimodal AI** to determine if they’re safe for **eczema** and **malassezia**-prone skin.

**Tech stack:**
- **Backend:** Python 3.10+, FastAPI, Uvicorn, MongoDB, Ollama (for Llama models), Tesseract 
- **AI/ML Models:** CLIP (image type gating), YOLOv8 (object detection/cropping), Llama 3 (LLM, vision LLM), EasyOCR, Tesseract OCR
- **Frontend:** Next.js 15, React, Tailwind CSS, Radix UI, Node.js 18+/20+
- **Other:** OpenCV, EasyOCR, Open Food Facts API, Transformers, requests, numpy, cv2

Models used: CLIP, Llama 3, YOLOv8, EasyOCR, Tesseract

### What it does
- Scans images/barcodes → extracts ingredients.
- Gates with CLIP, crops panels/barcodes with YOLO, OCR via EasyOCR/Tesseract.
- Uses vision LLM for food photos and an LLM + MongoDB to return Safe/Avoid.

### Quick start
- Backend
  - Requirements: Python 3.10+, Ollama models (`dolphin-llama3:latest`, `llama3.2-vision:latest`), optional Tesseract/MongoDB.
  - Commands (PowerShell):
    ```bash
    cd backend
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r ..\requirements.txt
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
    ```
- Frontend
  - Requirements: Node 18+ or 20+.
  - Commands:
    ```bash
    cd frontend
    npm install
    npm run dev
    # open http://localhost:3000
    ```

### API
- POST `http://localhost:8000/analyze`
  - Body: `{ "image": "data:image/jpeg;base64,<...>" }`
  - Returns: `ingredients.safe`, `ingredients.avoid`, `detected_class`, `processing_time_ms`.

### More detail
- See `backend/README.md` for pipeline, env vars, and Mongo seed.
- See `frontend/README.md` for dev/build and API URL.

### Project structure (key files)
- `backend/main.py` — FastAPI API & orchestration
- `backend/classify_item.py` — CLIP image-type gate
- `backend/detect_and_crop.py` — YOLO panel/barcode detection
- `backend/ingredient_list.py` — OCR pipeline for ingredient panels
- `backend/search_barcode.py` — Barcode decode + Open Food Facts lookup
- `backend/identify_food.py` — Vision LLM tokens from food photos
- `backend/recommend.py` — LLM cleaning + MongoDB trigger classification
- `frontend/app/page.tsx` — UI (camera/upload) calling `/analyze`

