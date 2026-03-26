"""
Analyze prediction probabilities at full resolution to check if
threshold=0.5 is optimal. No retraining needed — uses saved checkpoint.

Usage: python analyze_probs.py
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

import config
from dataset import bbox_mask, polygon_mask

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PROMPTS = {
    "taping": config.TAPING_PROMPTS[0],
    "cracks": config.CRACK_PROMPTS[0],
}


def analyze():
    # Load model from checkpoint
    processor = CLIPSegProcessor.from_pretrained(config.MODEL_NAME)
    model = CLIPSegForImageSegmentation.from_pretrained(config.MODEL_NAME)
    ckpt = config.CHECKPOINT_DIR / "best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=config.DEVICE))
    model.to(config.DEVICE)
    model.eval()
    print(f"Loaded: {ckpt}")

    # Collect probability stats per class
    stats = {"taping": {"gt_pos_probs": [], "gt_neg_probs": []},
             "cracks": {"gt_pos_probs": [], "gt_neg_probs": []}}

    for data_dir, dtype in [(config.TAPING_DIR, "taping"), (config.CRACKS_DIR, "cracks")]:
        img_dir = data_dir / "valid" / "images"
        lbl_dir = data_dir / "valid" / "labels"
        if not img_dir.exists():
            continue

        count = 0
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXTENSIONS:
                continue
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists() or lbl_path.stat().st_size == 0:
                continue

            image = Image.open(img_path).convert("RGB")
            w, h = image.size

            # GT mask at full resolution
            gt = bbox_mask(lbl_path, w, h) if dtype == "taping" else polygon_mask(lbl_path, w, h)
            gt_tensor = torch.from_numpy(gt)

            # Model prediction
            inputs = processor(text=[PROMPTS[dtype]], images=[image],
                               return_tensors="pt", padding="max_length", max_length=77, truncation=True)
            with torch.no_grad():
                logits = model(pixel_values=inputs["pixel_values"].to(config.DEVICE),
                               input_ids=inputs["input_ids"].to(config.DEVICE),
                               attention_mask=inputs["attention_mask"].to(config.DEVICE)).logits

            # Upsample to full resolution (same as predict.py)
            logits_up = F.interpolate(
                logits.unsqueeze(1), size=(h, w),
                mode="bilinear", align_corners=False,
            ).squeeze().cpu()
            probs = torch.sigmoid(logits_up)

            # Collect probabilities for GT-positive and GT-negative pixels
            pos_mask = gt_tensor > 0.5
            neg_mask = gt_tensor < 0.5
            if pos_mask.sum() > 0:
                stats[dtype]["gt_pos_probs"].extend(probs[pos_mask].numpy().tolist())
            if neg_mask.sum() > 0:
                # Sample negatives (too many otherwise)
                neg_probs = probs[neg_mask].numpy()
                stats[dtype]["gt_neg_probs"].extend(
                    np.random.choice(neg_probs, size=min(1000, len(neg_probs)), replace=False).tolist()
                )

            count += 1
            if count >= 50:  # analyze 50 images per class
                break

        print(f"\n{'='*60}")
        print(f"  {dtype.upper()} — probability analysis ({count} images)")
        print(f"{'='*60}")

        pos = np.array(stats[dtype]["gt_pos_probs"])
        if len(pos) > 0:
            print(f"\n  GT-POSITIVE pixels (should be predicted as 1):")
            print(f"    Mean probability:  {pos.mean():.4f}")
            print(f"    Median:            {np.median(pos):.4f}")
            print(f"    Std:               {pos.std():.4f}")
            print(f"    Min:               {pos.min():.4f}")
            print(f"    Max:               {pos.max():.4f}")
            print(f"")
            for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
                captured = (pos >= t).mean() * 100
                print(f"    Threshold={t:.1f} → captures {captured:.1f}% of GT-positive pixels")

        neg = np.array(stats[dtype]["gt_neg_probs"])
        if len(neg) > 0:
            print(f"\n  GT-NEGATIVE pixels (should be predicted as 0):")
            print(f"    Mean probability:  {neg.mean():.4f}")
            print(f"    Median:            {np.median(neg):.4f}")
            for t in [0.3, 0.4, 0.5]:
                false_pos = (neg >= t).mean() * 100
                print(f"    Threshold={t:.1f} → false positive rate: {false_pos:.1f}%")

    # Suggest optimal threshold
    print(f"\n{'='*60}")
    print(f"  THRESHOLD SWEEP (mIoU estimate)")
    print(f"{'='*60}")
    for dtype in ["taping", "cracks"]:
        pos = np.array(stats[dtype]["gt_pos_probs"])
        neg = np.array(stats[dtype]["gt_neg_probs"])
        if len(pos) == 0:
            continue
        print(f"\n  {dtype}:")
        best_t, best_iou = 0.5, 0.0
        for t in np.arange(0.1, 0.8, 0.05):
            tp = (pos >= t).sum()
            fn = (pos < t).sum()
            fp = (neg >= t).sum()
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
            marker = ""
            if iou > best_iou:
                best_iou = iou
                best_t = t
                marker = "  ← best"
            print(f"    t={t:.2f}  IoU={iou:.4f}  TP%={tp/len(pos)*100:.1f}  FP%={fp/len(neg)*100:.1f}{marker}")
        print(f"    → Optimal threshold: {best_t:.2f} (IoU={best_iou:.4f})")


if __name__ == "__main__":
    analyze()
