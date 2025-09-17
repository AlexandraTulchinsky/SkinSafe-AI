from pathlib import Path
import cv2
from ultralytics import YOLO
import os

# Path to your trained YOLO model
MODEL_PATH = os.getenv(
    "YOLO_MODEL_PATH",
    "runs/detect/train/weights/best.pt"  
)

# Initialize YOLO
model = YOLO(MODEL_PATH)
print("[YOLO DEBUG] model.names =", model.names, flush=True)
def detect_and_crop(image_bgr, target_classes=None, conf=0.1):
    """
    Run YOLO detection, crop detected regions, and return list of crops.
    """
    results = model.predict(source=image_bgr, conf=0.1, imgsz=768, verbose=False)
    print(f"[YOLO DEBUG] raw result boxes: {results[0].boxes}", flush=True)

    crops = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls.item())
            cls_name = model.names[cls_id]
            conf_score = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Debug logging for every detection
            print(f"[YOLO DEBUG] Detected {cls_name} (conf={conf_score:.2f}) "
                  f"at box=({x1},{y1},{x2},{y2})", flush=True)

            if (target_classes is None) or (cls_name in target_classes):
                if x2 > x1 and y2 > y1:  # ensure valid crop
                    crop = image_bgr[y1:y2, x1:x2]
                    crops.append((cls_name, crop))
                else:
                    print(f"[YOLO DEBUG] Skipped invalid box for {cls_name}", flush=True)

    if not crops:
        print("[YOLO DEBUG] No crops returned after filtering.", flush=True)

    return crops
