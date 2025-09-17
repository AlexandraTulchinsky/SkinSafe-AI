"""
SkinSafeAI — multi-class object detection (ingredient_list + barcode)

Local training script that uses an existing YOLO-format dataset directory (default: ./SkinSafeAI-2).

Example:
    python train_skinsafeai_local.py \
        --data_root SkinSafeAI-2 \
        --pretrained yolov8n.pt \
        --epochs 50 --imgsz 640 --batch 16 --device 0
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, List


try:
    import yaml
except Exception:
    print("Missing dependency 'pyyaml'. Install with: pip install pyyaml", file=sys.stderr)
    raise

try:
    import torch
except Exception:
    print("Missing dependency 'torch'. Install PyTorch that matches your CUDA runtime.", file=sys.stderr)
    raise

try:
    from ultralytics import YOLO
except Exception:
    print("Missing dependency 'ultralytics'. Install with: pip install ultralytics", file=sys.stderr)
    raise


# -------------------------
# Helper utilities
# -------------------------
def _is_windows() -> bool:
    return os.name == "nt"

def _workers_default() -> int:
    return 0 if _is_windows() else 8


def _nearest_multiple_of_32(n: int) -> int:
    return int(round(n / 32)) * 32 or 32

def _cuda_preflight(requested_device: str | int) -> str:

    dev_str = str(requested_device).strip().lower()

    if dev_str == "cpu":
        print("Running on CPU; training will be slow.")
        return "cpu"

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    cuda_avail = torch.cuda.is_available()
    n_gpus = torch.cuda.device_count() if cuda_avail else 0
    cuda_ver = getattr(torch.version, "cuda", None)
    print(f"CUDA preflight: available={cuda_avail}  device_count={n_gpus}  torch.version.cuda={cuda_ver}")

    if not cuda_avail:
        print(
            "CUDA is not available in your current PyTorch build, but a GPU device was requested.\n"
            "Install a CUDA-enabled PyTorch wheel that matches your system, e.g.:\n"
            "  pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio\n",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        _ = torch.cuda.get_device_name(0)
    except Exception as e:
        print(f"Could not query CUDA device 0: {e}", file=sys.stderr)
        sys.exit(1)

    torch.backends.cudnn.benchmark = True 
    return dev_str


def find_data_yaml(dataset_root: Path) -> Path:
 
    direct = dataset_root / "data.yaml"
    if direct.exists():
        return direct
    for p in dataset_root.rglob("data.yaml"):
        return p
    raise FileNotFoundError(
        f"Could not find data.yaml under {dataset_root}. Ensure your dataset follows the YOLO layout."
    )


def read_classes_from_yaml(data_yaml: Path) -> List[str]:
    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    if not isinstance(names, list):
        names = list(names)
    return [str(n) for n in names]


def find_images_dir(dataset_root: Path, split: str) -> Optional[Path]:
    cand = dataset_root / split / "images"
    return cand if cand.exists() else None


def check_dataset_integrity(dataset_root: Path, strict: bool = True) -> None:

    seg_like: list[Path] = []
    missing_images: list[Path] = []

    for split in ("train", "valid", "test"):
        labels_dir = dataset_root / split / "labels"
        images_dir = dataset_root / split / "images"
        if not labels_dir.exists() or not images_dir.exists():
            continue

        for label_file in labels_dir.glob("*.txt"):
            try:
                with open(label_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) > 5:
                            seg_like.append(label_file)
                            break
                        try:
                            _ = [float(x) for x in parts]
                        except ValueError:
                            seg_like.append(label_file)
                            break
            except Exception:
                seg_like.append(label_file)
                continue

         
            stem = label_file.stem
            if not any((images_dir / f"{stem}{ext}").exists() for ext in (".jpg", ".jpeg", ".png")):
                missing_images.append(label_file)

    if seg_like:
        msg = f"Found {len(seg_like)} label files that look like segmentation (polygons) instead of detection.\n" \
              f"Example: {seg_like[0]}"
        if strict:
            print(msg, file=sys.stderr)
            sys.exit(1)
        else:
            print("Warning: " + msg)

    if missing_images:
        msg = f"Found {len(missing_images)} label files without a matching image.\n" \
              f"Example: {missing_images[0]}"
        if strict:
            print(msg, file=sys.stderr)
            sys.exit(1)
        else:
            print("Warning: " + msg)


def train_yolo(
    data_yaml: Path,
    pretrained: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str | int,
    workers: int,
    resume: bool,
) -> Path:
    """
    Train YOLO and return path to best weights.
    """
    device_str = _cuda_preflight(device)

    if device_str != "cpu":
        try:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"Training on GPU device={device_str} ({gpu_name})")
        except Exception:
            print(f"Training on GPU device={device_str}")
    else:
        print("Proceeding on CPU.")

    print(f"\nUsing data.yaml: {data_yaml}")
    print(f"Pretrained weights: {pretrained}")
    print(f"epochs={epochs}, imgsz={imgsz}, batch={batch}, workers={workers}, device={device_str}, resume={resume}\n")

    model = YOLO(pretrained)
    try:
        results = model.train(
            data=str(data_yaml),
            epochs=int(epochs),
            imgsz=int(imgsz),
            batch=int(batch),
            workers=int(workers),
            device=str(device_str),
            save=True,          
            verbose=True,
            amp=True,            
            resume=resume,
        )
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(
                "CUDA out of memory.",
                file=sys.stderr,
            )
        raise

    save_dir = Path(getattr(results, "save_dir", Path("runs") / "detect" / "train"))
    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"best.pt not found at {best_pt}. Check training logs above.")

    print(f"Training complete. Best weights: {best_pt}")
    print(f"Run artifacts directory: {save_dir}")
    return best_pt


def validate_model(best_weights: Path, data_yaml: Path, imgsz: int, device: str | int) -> None:
    device_str = str(device)
    print(f"Validating {best_weights.name} on device={device_str} ...")
    model = YOLO(str(best_weights))
    _ = model.val(data=str(data_yaml), imgsz=int(imgsz), device=device_str)


def quick_predict(best_weights: Path, dataset_root: Path, conf: float = 0.25, device: str | int = "0") -> None:

    test_dir = find_images_dir(dataset_root, "test") or find_images_dir(dataset_root, "valid")
    if not test_dir:
        print("No test/images or valid/images directory found; skipping quick prediction.")
        return

    print(f"Running a quick prediction on: {test_dir} (device={device})")
    model = YOLO(str(best_weights))
    model.predict(
        source=str(test_dir),
        conf=conf,
        save=True,
        save_txt=False,
        save_conf=True,
        verbose=False,
        device=str(device),
    )
    print("Prediction complete. See runs/detect/predict*/")



# Entry point
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO locally on SkinSafeAI (barcode + ingredient_list).")

    # Dataset location
    p.add_argument(
        "--data_root",
        type=str,
        default="SkinSafeAI-2",
        help="Path to dataset root directory (containing data.yaml somewhere within).",
    )
    p.add_argument(
        "--data_yaml",
        type=str,
        default=None,
        help="Optional explicit path to data.yaml; if omitted, the script searches under --data_root.",
    )

    # Training hyperparameters
    p.add_argument("--pretrained", default="yolov8s.pt",
                   help="YOLO pretrained weights")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=768)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0", help="GPU index (e.g., 0), '0,1' for multi-GPU, or 'cpu'")
    p.add_argument("--workers", type=int, default=_workers_default())
    p.add_argument("--resume", action="store_true", help="Resume training if a prior run exists in the default path.")
    p.add_argument("--no_strict_integrity", action="store_true",
                   help="Do not exit on integrity violations (segmentation labels, missing images).")

    return p.parse_args()


def main() -> None:
  
    logging.basicConfig(
        filename="train_log.txt",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("Starting training run")

    args = parse_args()

    if args.batch <= 0:
        sys.exit("Batch size must be > 0")
    if args.imgsz <= 0:
        sys.exit("imgsz must be > 0")
    if args.imgsz % 32 != 0:
        recommended = _nearest_multiple_of_32(args.imgsz)
        print(f"imgsz {args.imgsz} is not a multiple of 32; using {recommended} instead for optimal performance.")
        args.imgsz = recommended

    dataset_root = Path(args.data_root).expanduser().resolve()
    if not dataset_root.exists():
        print(f"Dataset root not found: {dataset_root}", file=sys.stderr)
        sys.exit(1)

    if args.data_yaml:
        data_yaml = Path(args.data_yaml).expanduser().resolve()
        if not data_yaml.exists():
            print(f"data.yaml not found at {data_yaml}", file=sys.stderr)
            sys.exit(1)
    else:
        data_yaml = find_data_yaml(dataset_root)

    check_dataset_integrity(dataset_root, strict=not args.no_strict_integrity)

    class_names = read_classes_from_yaml(data_yaml)
    print(f"Classes in dataset: {class_names} (nc={len(class_names)})")
    if len(class_names) < 2:
        print("Warning: fewer than 2 classes were found. Verify dataset class names in data.yaml.", file=sys.stderr)

    # Train
    best_weights = train_yolo(
        data_yaml=data_yaml,
        pretrained=args.pretrained,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        resume=args.resume,
    )

    validate_model(best_weights, data_yaml, imgsz=args.imgsz, device=args.device)

    quick_predict(best_weights, dataset_root, device=args.device)


if __name__ == "__main__":
    main()
