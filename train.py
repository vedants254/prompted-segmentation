"""
Fine-tune CLIPSeg decoder on taping + crack datasets.

Usage:
    python train.py

Expects data in:
    data/taping/{train,valid}/images/  +  labels/
    data/cracks/{train,valid}/images/  +  labels/
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

import config
from dataset import PromptedSegDataset, collate_fn


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Loss ──────────────────────────────────────────────────────────────────────

def focal_tversky_loss(
    logits: torch.Tensor, targets: torch.Tensor,
    alpha: float = 0.3, beta: float = 0.7, gamma: float = 1.33, smooth: float = 1.0,
):
    """Focal Tversky loss — penalises missed pixels (FN) more than false alarms (FP),
    and focuses gradient on hard samples via the focal exponent γ."""
    probs = torch.sigmoid(logits)
    tp = (probs * targets).sum(dim=(-2, -1))
    fp = (probs * (1 - targets)).sum(dim=(-2, -1))
    fn = ((1 - probs) * targets).sum(dim=(-2, -1))
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return ((1.0 - tversky) ** gamma).mean()


def combined_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    ft = focal_tversky_loss(logits, targets)
    return config.BCE_WEIGHT * bce + config.DICE_WEIGHT * ft


# ── Metrics ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def per_sample_metrics(logits: torch.Tensor, targets: torch.Tensor, dtypes: list):
    """Return per-dataset IoU and Dice lists."""
    preds = (torch.sigmoid(logits) > config.THRESHOLD).float()
    results = {"taping": {"iou": [], "dice": []}, "cracks": {"iou": [], "dice": []}}
    for p, t, dt in zip(preds, targets, dtypes):
        tp = (p * t).sum()
        fp = (p * (1 - t)).sum()
        fn = ((1 - p) * t).sum()
        if tp + fp + fn == 0:
            iou, dice = 1.0, 1.0
        else:
            iou = (tp / (tp + fp + fn)).item()
            dice = (2 * tp / (2 * tp + fp + fn)).item()
        results[dt]["iou"].append(iou)
        results[dt]["dice"].append(dice)
    return results


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    set_seed(config.SEED)
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Model ─────────────────────────────────────────────────────────────
    processor = CLIPSegProcessor.from_pretrained(config.MODEL_NAME)
    model = CLIPSegForImageSegmentation.from_pretrained(config.MODEL_NAME)
    model.to(config.DEVICE)

    # Freeze CLIP backbone — only train the decoder
    for p in model.clip.parameters():
        p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Device     : {config.DEVICE}")
    print(f"Parameters : {trainable:,} trainable / {total:,} total")

    # ── Data ──────────────────────────────────────────────────────────────
    train_ds = PromptedSegDataset(
        config.TAPING_DIR, config.CRACKS_DIR, "train",
        processor, config.TAPING_PROMPTS, config.CRACK_PROMPTS,
        augment=True,
    )
    val_ds = PromptedSegDataset(
        config.TAPING_DIR, config.CRACKS_DIR, "valid",
        processor, config.TAPING_PROMPTS, config.CRACK_PROMPTS,
        augment=False, deterministic_prompt=True,
    )
    print(f"Train      : {len(train_ds)} samples  ({len(train_ds)//2} images × 2)")
    print(f"Valid      : {len(val_ds)} samples  ({len(val_ds)//2} images × 2)")

    if len(train_ds) == 0:
        raise RuntimeError(
            "No training samples found.  Check that data/taping/train/ and "
            "data/cracks/train/ contain images/ and labels/ subdirectories."
        )

    # ── Weighted sampler to fix class imbalance (820 taping vs 5164 cracks) ──
    # Each sample gets inverse-frequency weight based on its source dataset.
    # Positive and negative from the same image share the same weight.
    n_base = len(train_ds.samples)  # number of real images (before 2x)
    n_taping = sum(1 for _, _, dt in train_ds.samples if dt == "taping")
    n_cracks = sum(1 for _, _, dt in train_ds.samples if dt == "cracks")
    weight_per_type = {"taping": 1.0 / max(n_taping, 1), "cracks": 1.0 / max(n_cracks, 1)}

    sample_weights = []
    for i in range(len(train_ds)):
        real_idx = i % n_base
        _, _, dtype = train_ds.samples[real_idx]
        sample_weights.append(weight_per_type[dtype])

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_ds),
        replacement=True,
    )
    print(f"Sampler    : taping={n_taping} (w={weight_per_type['taping']:.6f})  "
          f"cracks={n_cracks} (w={weight_per_type['cracks']:.6f})")

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, sampler=sampler,
        collate_fn=collate_fn, num_workers=config.NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=config.NUM_WORKERS, pin_memory=True,
    )

    # ── Optimiser & scheduler ─────────────────────────────────────────────
    optimiser = torch.optim.AdamW(
        model.decoder.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=config.NUM_EPOCHS,
    )

    # ── Loop ──────────────────────────────────────────────────────────────
    best_min_miou = 0.0
    best_snapshot = {}
    history = []
    t_start = time.time()

    for epoch in range(1, config.NUM_EPOCHS + 1):
        # ── train ─────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{config.NUM_EPOCHS}", leave=False)
        for batch in pbar:
            pv    = batch["pixel_values"].to(config.DEVICE)
            ids   = batch["input_ids"].to(config.DEVICE)
            am    = batch["attention_mask"].to(config.DEVICE)
            masks = batch["mask"].to(config.DEVICE)

            logits = model(pixel_values=pv, input_ids=ids, attention_mask=am).logits

            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(
                    logits.unsqueeze(1), size=masks.shape[-2:],
                    mode="bilinear", align_corners=False,
                ).squeeze(1)

            loss = combined_loss(logits, masks)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=1.0)
            optimiser.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_train_loss = running_loss / len(train_loader)

        # ── validate (per-prompt mIoU & Dice) ─────────────────────────
        model.eval()
        val_metrics = {"taping": {"iou": [], "dice": []}, "cracks": {"iou": [], "dice": []}}
        with torch.no_grad():
            for batch in val_loader:
                pv    = batch["pixel_values"].to(config.DEVICE)
                ids   = batch["input_ids"].to(config.DEVICE)
                am    = batch["attention_mask"].to(config.DEVICE)
                masks = batch["mask"].to(config.DEVICE)

                logits = model(
                    pixel_values=pv, input_ids=ids, attention_mask=am,
                ).logits

                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = F.interpolate(
                        logits.unsqueeze(1), size=masks.shape[-2:],
                        mode="bilinear", align_corners=False,
                    ).squeeze(1)

                batch_metrics = per_sample_metrics(logits, masks, batch["dataset_type"])
                for dt in ["taping", "cracks"]:
                    val_metrics[dt]["iou"].extend(batch_metrics[dt]["iou"])
                    val_metrics[dt]["dice"].extend(batch_metrics[dt]["dice"])

        # Per-prompt averages
        tap_miou = np.mean(val_metrics["taping"]["iou"]) if val_metrics["taping"]["iou"] else 0.0
        tap_dice = np.mean(val_metrics["taping"]["dice"]) if val_metrics["taping"]["dice"] else 0.0
        crk_miou = np.mean(val_metrics["cracks"]["iou"]) if val_metrics["cracks"]["iou"] else 0.0
        crk_dice = np.mean(val_metrics["cracks"]["dice"]) if val_metrics["cracks"]["dice"] else 0.0
        min_miou = min(tap_miou, crk_miou)
        mean_dice = (tap_dice + crk_dice) / 2

        lr_now = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:02d}/{config.NUM_EPOCHS}  "
            f"loss={avg_train_loss:.4f}  "
            f"tap_mIoU={tap_miou:.4f}  tap_Dice={tap_dice:.4f}  "
            f"crk_mIoU={crk_miou:.4f}  crk_Dice={crk_dice:.4f}  "
            f"min_mIoU={min_miou:.4f}  mean_Dice={mean_dice:.4f}  "
            f"lr={lr_now:.2e}"
        )

        # ── log epoch ─────────────────────────────────────────────────
        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "lr": lr_now,
            "taping": {"mIoU": round(float(tap_miou), 4), "Dice": round(float(tap_dice), 4)},
            "cracks": {"mIoU": round(float(crk_miou), 4), "Dice": round(float(crk_dice), 4)},
            "min_mIoU": round(float(min_miou), 4),
            "mean_Dice": round(float(mean_dice), 4),
        }
        history.append(epoch_record)

        # ── checkpoint (based on min-class mIoU — ensures no class is neglected)
        if min_miou > best_min_miou:
            best_min_miou = min_miou
            best_snapshot = epoch_record.copy()
            torch.save(model.state_dict(), config.CHECKPOINT_DIR / "best.pt")
            print(f"  ↑ new best  (min_mIoU={min_miou:.4f})")

    # ── Done ──────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), config.CHECKPOINT_DIR / "final.pt")
    elapsed = time.time() - t_start
    print(f"\nTraining complete in {elapsed/60:.1f} min")
    print(f"Best min mIoU  : {best_min_miou:.4f}")
    print(f"Checkpoints    : {config.CHECKPOINT_DIR}")

    # ── Save training log ─────────────────────────────────────────────────
    training_log = {
        "config": {
            "model": config.MODEL_NAME,
            "seed": config.SEED,
            "batch_size": config.BATCH_SIZE,
            "num_epochs": config.NUM_EPOCHS,
            "lr": config.LR,
            "weight_decay": config.WEIGHT_DECAY,
            "bce_weight": config.BCE_WEIGHT,
            "dice_weight": config.DICE_WEIGHT,
            "threshold": config.THRESHOLD,
        },
        "data": {
            "train_samples": len(train_ds),
            "valid_samples": len(val_ds),
        },
        "model_info": {
            "trainable_params": trainable,
            "total_params": total,
            "device": str(config.DEVICE),
        },
        "best_checkpoint": best_snapshot,
        "training_time_min": round(elapsed / 60, 1),
        "history": history,
    }
    log_path = Path("training_log.json")
    log_path.write_text(json.dumps(training_log, indent=2))
    print(f"Training log   : {log_path}")


if __name__ == "__main__":
    train()
