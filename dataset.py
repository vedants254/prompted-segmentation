"""
PromptedSegDataset — loads taping (bbox) and crack (polygon) annotations,
producing positive + negative text-conditioned pairs.

Every image yields TWO samples per epoch:
  positive — matching prompt  + real mask
  negative — opposite prompt  + all-zero mask

This forces the model to attend to the text prompt instead of
learning a single "detect any defect" behaviour.
"""

import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── Mask parsers ──────────────────────────────────────────────────────────────

def bbox_mask(label_path: Path, img_w: int, img_h: int) -> np.ndarray:
    """YOLO detection format  →  filled-rectangle binary mask.

    Each line: class cx cy bw bh   (all normalised 0-1)
    Multiple boxes → union of all filled rectangles.
    """
    mask = np.zeros((img_h, img_w), dtype=np.float32)
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cx, cy, bw, bh = map(float, parts[1:5])
            x1 = max(0, int((cx - bw / 2) * img_w))
            y1 = max(0, int((cy - bh / 2) * img_h))
            x2 = min(img_w, int((cx + bw / 2) * img_w))
            y2 = min(img_h, int((cy + bh / 2) * img_h))
            mask[y1:y2, x1:x2] = 1.0
    return mask


def polygon_mask(label_path: Path, img_w: int, img_h: int) -> np.ndarray:
    """YOLOv8-seg format  →  filled-polygon binary mask.

    Each line: class x1 y1 x2 y2 … (normalised polygon vertices)
    Multiple polygons → union of all filled regions.
    """
    mask_img = Image.new("L", (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask_img)
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:          # class + at least 3 points
                continue
            coords = list(map(float, parts[1:]))
            points = [
                (coords[i] * img_w, coords[i + 1] * img_h)
                for i in range(0, len(coords) - 1, 2)
            ]
            if len(points) >= 3:
                draw.polygon(points, fill=255)
    return np.array(mask_img, dtype=np.float32) / 255.0


# ── Dataset ───────────────────────────────────────────────────────────────────

class PromptedSegDataset(Dataset):
    """
    Parameters
    ----------
    taping_dir, cracks_dir : Path
        Root dirs containing  {split}/images/  and  {split}/labels/
    split : str
        "train", "valid", or "test"
    processor : CLIPSegProcessor
        Handles image normalisation + text tokenisation.
    taping_prompts, crack_prompts : list[str]
        Prompt pools for each defect type.
    augment : bool
        Enable horizontal-flip augmentation (train only).
    deterministic_prompt : bool
        When True, always use the first prompt (for reproducible eval).
    """

    def __init__(
        self,
        taping_dir: Path,
        cracks_dir: Path,
        split: str,
        processor,
        taping_prompts: list,
        crack_prompts: list,
        augment: bool = False,
        deterministic_prompt: bool = False,
    ):
        self.processor = processor
        self.taping_prompts = taping_prompts
        self.crack_prompts = crack_prompts
        self.augment = augment
        self.deterministic_prompt = deterministic_prompt

        # Collect (image_path, label_path, dataset_type) triples
        self.samples: list[tuple[Path, Path, str]] = []
        for data_dir, dtype in [(taping_dir, "taping"), (cracks_dir, "cracks")]:
            img_dir = data_dir / split / "images"
            lbl_dir = data_dir / split / "labels"
            if not img_dir.exists() or not lbl_dir.exists():
                continue
            for img_path in sorted(img_dir.iterdir()):
                if img_path.suffix.lower() not in IMG_EXTENSIONS:
                    continue
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                if lbl_path.exists() and lbl_path.stat().st_size > 0:
                    self.samples.append((img_path, lbl_path, dtype))

    # 2× samples: first half = positives, second half = negatives
    def __len__(self):
        return len(self.samples) * 2

    def __getitem__(self, idx):
        is_negative = idx >= len(self.samples)
        real_idx = idx % len(self.samples)
        img_path, lbl_path, dtype = self.samples[real_idx]

        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        # ── Build mask + pick prompt ──────────────────────────────────────
        if is_negative:
            # Wrong prompt → empty mask
            prompt = self._pick_prompt("cracks" if dtype == "taping" else "taping")
            mask = np.zeros((h, w), dtype=np.float32)
        else:
            # Correct prompt → real mask
            prompt = self._pick_prompt(dtype)
            mask = (
                bbox_mask(lbl_path, w, h)
                if dtype == "taping"
                else polygon_mask(lbl_path, w, h)
            )

        # ── Augmentation (spatial — applied to both image and mask) ───────
        if self.augment and random.random() > 0.5:
            image = TF.hflip(image)
            mask = np.fliplr(mask).copy()

        # ── Processor: image normalisation + text tokenisation ────────────
        inputs = self.processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True,
        )

        # ── Resize mask to 352×352 (model output resolution) ─────────────
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        mask_352 = mask_pil.resize((352, 352), Image.NEAREST)
        mask_tensor = torch.from_numpy(np.array(mask_352)).float() / 255.0

        return {
            "pixel_values":  inputs["pixel_values"].squeeze(0),
            "input_ids":     inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "mask":          mask_tensor,
            "image_path":    str(img_path),
            "prompt":        prompt,
            "original_size": (h, w),
            "dataset_type":  dtype,
        }

    def _pick_prompt(self, dtype: str) -> str:
        pool = self.taping_prompts if dtype == "taping" else self.crack_prompts
        return pool[0] if self.deterministic_prompt else random.choice(pool)


# ── Collate ───────────────────────────────────────────────────────────────────

def collate_fn(batch: list[dict]) -> dict:
    """Stack tensors, keep metadata as lists."""
    return {
        "pixel_values":  torch.stack([b["pixel_values"]  for b in batch]),
        "input_ids":     torch.stack([b["input_ids"]     for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "mask":          torch.stack([b["mask"]          for b in batch]),
        "image_path":    [b["image_path"]    for b in batch],
        "prompt":        [b["prompt"]        for b in batch],
        "original_size": [b["original_size"] for b in batch],
        "dataset_type":  [b["dataset_type"]  for b in batch],
    }
