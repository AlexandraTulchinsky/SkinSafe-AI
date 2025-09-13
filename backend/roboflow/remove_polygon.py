#!/usr/bin/env python3
import os
from pathlib import Path

# Root of your dataset
DATASET_ROOT = Path("C:/Users/alexandra.tulchinsky/OneDrive - Ross Video/Documents/personal_project/backend_v0/roboflow/SkinSafeAI-2")

def is_segmentation_label(label_path: Path) -> bool:
    """Return True if any line has more than 5 values (YOLO seg format)."""
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 5:
                return True
    return False

def cleanup_split(split: str) -> int:
    """Check labels in one split (train/valid/test), remove seg files + images. Return count removed."""
    split_dir = DATASET_ROOT / split
    labels_dir = split_dir / "labels"
    images_dir = split_dir / "images"
    removed = 0

    for label_file in labels_dir.glob("*.txt"):
        if is_segmentation_label(label_file):
            # remove corresponding image (try jpg/png/jpeg)
            stem = label_file.stem
            found_img = False
            for ext in [".jpg", ".jpeg", ".png"]:
                img_file = images_dir / f"{stem}{ext}"
                if img_file.exists():
                    img_file.unlink()
                    found_img = True
                    break
            # remove the label file
            label_file.unlink()
            removed += 1
            print(f"Removed {label_file} {'and image' if found_img else '(no image found)'}")

    return removed

def main():
    total_removed = {}
    for split in ["train", "valid", "test"]:
        removed = cleanup_split(split)
        total_removed[split] = removed

    print("\n=== Cleanup Summary ===")
    for split, count in total_removed.items():
        print(f"{split}: {count} files removed")
    print("=======================")

if __name__ == "__main__":
    main()
