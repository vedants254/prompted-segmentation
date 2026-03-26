"""
Generate prediction masks on the test set, compute metrics, save visualisations.

Usage:
    python predict.py              # evaluate on test split
    python predict.py --split valid  # evaluate on validation split

Outputs:
    outputs/predictions/   — binary PNG masks  ({id}__segment_*.png)
    outputs/visualizations/ — side-by-side comparison grids
"""

import argparse
import json
import time
from pathlib import Path 

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

import config
from dataset import bbox_mask, polygon_mask

# ── Canonical prompts (one per defect type, used at inference) ────────────────
PROMPTS = {
    "taping": config.TAPING_PROMPTS[0],   # "segment taping area"
    "cracks": config.CRACK_PROMPTS[0],     # "segment crack"
}
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── Test dataset ──────────────────────────────────────────────────────────────

class TestDataset(Dataset):
    """
    Each test image paired with its MATCHING prompt only.
    Taping images → "segment taping area" + GT bbox mask
    Crack images  → "segment crack"       + GT polygon mask

    Metrics are evaluated per-dataset as the rubric requires:
    mIoU & Dice on "segment taping area" and "segment crack" separately.
    """

    def __init__(self, taping_dir, cracks_dir, split, processor):
        self.processor = processor
        self.items = []   # (img_path, prompt, lbl_path, dtype)

        for data_dir, dtype in [(taping_dir, "taping"), (cracks_dir, "cracks")]:
            img_dir = data_dir / split / "images"
            lbl_dir = data_dir / split / "labels"
            if not img_dir.exists():
                continue
            for img_path in sorted(img_dir.iterdir()):
                if img_path.suffix.lower() not in IMG_EXTENSIONS:
                    continue
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                has_lbl = lbl_path.exists() and lbl_path.stat().st_size > 0
                if has_lbl:
                    self.items.append((img_path, PROMPTS[dtype], lbl_path, dtype))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, prompt, lbl_path, dtype = self.items[idx]
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        gt = bbox_mask(lbl_path, w, h) if dtype == "taping" else polygon_mask(lbl_path, w, h)

        inputs = self.processor(
            text=[prompt], images=[image],
            return_tensors="pt", padding="max_length", max_length=77, truncation=True,
        )

        return {
            "pixel_values":  inputs["pixel_values"].squeeze(0),
            "input_ids":     inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "gt_mask":       torch.from_numpy(gt),
            "image_path":    str(img_path),
            "prompt":        prompt,
            "original_size": (h, w),
            "dataset_type":  dtype,
        }


def test_collate(batch):
    return {
        "pixel_values":  torch.stack([b["pixel_values"]  for b in batch]),
        "input_ids":     torch.stack([b["input_ids"]     for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "gt_mask":       [b["gt_mask"]       for b in batch],     # different sizes
        "image_path":    [b["image_path"]    for b in batch],
        "prompt":        [b["prompt"]        for b in batch],
        "original_size": [b["original_size"] for b in batch],
        "dataset_type":  [b["dataset_type"]  for b in batch],
    }


# ── Visualisation helper ──────────────────────────────────────────────────────

def save_comparison(img_path, gt_np, pred_np, prompt, save_path):
    """orig | GT | pred  side-by-side."""
    img = Image.open(img_path).convert("RGB")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[1].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth")
    axes[2].imshow(pred_np, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Prediction")
    fig.suptitle(f'Prompt: "{prompt}"', fontsize=12)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def predict(split: str = "test"):
    config.PRED_DIR.mkdir(parents=True, exist_ok=True)
    config.VIS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────
    processor = CLIPSegProcessor.from_pretrained(config.MODEL_NAME)
    model = CLIPSegForImageSegmentation.from_pretrained(config.MODEL_NAME)

    ckpt = config.CHECKPOINT_DIR / "best.pt"
    if not ckpt.exists():
        print(f"Checkpoint not found at {ckpt} — using pretrained weights (zero-shot).")
    else:
        model.load_state_dict(torch.load(ckpt, map_location=config.DEVICE))
        print(f"Loaded checkpoint: {ckpt}")

    model.to(config.DEVICE)
    model.eval()

    # Model size
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Model size : {param_bytes / 1e6:.1f} MB")

    # ── Data ──────────────────────────────────────────────────────────────
    ds = TestDataset(config.TAPING_DIR, config.CRACKS_DIR, split, processor)
    loader = DataLoader(
        ds, batch_size=1, shuffle=False, collate_fn=test_collate,
        num_workers=config.NUM_WORKERS,
    )
    print(f"Test items : {len(ds)} images  (taping + cracks, each with matching prompt)")

    # ── Inference ─────────────────────────────────────────────────────────
    # Per-dataset metrics — evaluated separately as rubric requires
    per_dataset = {
        "taping": {"iou": [], "dice": []},
        "cracks": {"iou": [], "dice": []},
    }
    vis_candidates = {}      # dtype → list of (img_path, gt, pred, prompt)
    inference_times = []

    with torch.no_grad():
        for batch in loader:
            pv  = batch["pixel_values"].to(config.DEVICE)
            ids = batch["input_ids"].to(config.DEVICE)
            am  = batch["attention_mask"].to(config.DEVICE)

            if config.DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            logits = model(pixel_values=pv, input_ids=ids, attention_mask=am).logits

            if config.DEVICE.type == "cuda":
                torch.cuda.synchronize()
            inference_times.append(time.perf_counter() - t0)

            # Upsample to original image size
            h, w = batch["original_size"][0]
            logits_up = F.interpolate(
                logits.unsqueeze(1), size=(h, w),
                mode="bilinear", align_corners=False,
            ).squeeze()

            pred = (torch.sigmoid(logits_up) > config.THRESHOLD).float().cpu()
            gt   = batch["gt_mask"][0]

            # ── Save prediction mask ──────────────────────────────────
            img_id = Path(batch["image_path"][0]).stem
            prompt_slug = batch["prompt"][0].replace(" ", "_")
            mask_out = (pred.numpy() * 255).astype(np.uint8)
            Image.fromarray(mask_out).save(config.PRED_DIR / f"{img_id}__{prompt_slug}.png")

            # ── Metrics (per dataset) ─────────────────────────────────
            tp = (pred * gt).sum().float()
            fp = (pred * (1 - gt)).sum().float()
            fn = ((1 - pred) * gt).sum().float()

            if tp + fp + fn == 0:
                iou, dice = 1.0, 1.0
            else:
                iou  = (tp / (tp + fp + fn)).item()
                dice = (2 * tp / (2 * tp + fp + fn)).item()

            dtype = batch["dataset_type"][0]
            per_dataset[dtype]["iou"].append(iou)
            per_dataset[dtype]["dice"].append(dice)

            # Collect vis candidates (up to 2 per type)
            vis_candidates.setdefault(dtype, [])
            if len(vis_candidates[dtype]) < 2:
                vis_candidates[dtype].append(
                    (batch["image_path"][0], gt.numpy(), pred.numpy(), batch["prompt"][0])
                )

    # ── Report ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  TEST RESULTS  ({split} split)")
    print(f"{'='*60}")

    all_ious, all_dices = [], []
    prompt_labels = {"taping": PROMPTS["taping"], "cracks": PROMPTS["cracks"]}
    for dtype in ["taping", "cracks"]:
        ious = per_dataset[dtype]["iou"]
        dcs  = per_dataset[dtype]["dice"]
        if not ious:
            print(f"  {prompt_labels[dtype]:30s}  (no test data)")
            continue
        all_ious.extend(ious)
        all_dices.extend(dcs)
        print(
            f"  {prompt_labels[dtype]:30s}  "
            f"mIoU={np.mean(ious):.4f}  Dice={np.mean(dcs):.4f}  (n={len(ious)})"
        )

    if all_ious:
        print(f"  {'OVERALL':30s}  mIoU={np.mean(all_ious):.4f}  Dice={np.mean(all_dices):.4f}")
    print(f"\n  Avg inference : {np.mean(inference_times)*1000:.1f} ms/image")
    print(f"  Predictions   : {config.PRED_DIR}/")

    # ── Save metrics to JSON ──────────────────────────────────────────────
    results = {}
    for dtype in ["taping", "cracks"]:
        ious = per_dataset[dtype]["iou"]
        dcs  = per_dataset[dtype]["dice"]
        if ious:
            results[prompt_labels[dtype]] = {
                "mIoU": round(float(np.mean(ious)), 4),
                "Dice": round(float(np.mean(dcs)), 4),
                "n_samples": len(ious),
            }
    if all_ious:
        results["overall"] = {
            "mIoU": round(float(np.mean(all_ious)), 4),
            "Dice": round(float(np.mean(all_dices)), 4),
        }
    results["avg_inference_ms"] = round(float(np.mean(inference_times) * 1000), 1)

    metrics_path = Path("metrics.json")
    metrics_path.write_text(json.dumps(results, indent=2))
    print(f"  Metrics JSON  : {metrics_path}")

    # ── Visualisations (3–4 examples as required by rubric) ───────────────
    vis_count = 0
    for dtype, examples in vis_candidates.items():
        for img_path, gt_np, pred_np, prompt in examples:
            vis_count += 1
            fname = f"example_{vis_count}_{dtype}.png"
            save_comparison(img_path, gt_np, pred_np, prompt, config.VIS_DIR / fname)
    print(f"  Visualisations: {vis_count} saved to {config.VIS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["test", "valid", "train"])
    args = parser.parse_args()
    predict(args.split)
