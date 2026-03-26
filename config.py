"""Central configuration — paths, hyperparameters, prompt pools."""

from pathlib import Path
import torch

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT  = Path("data")
TAPING_DIR = DATA_ROOT / "taping"
CRACKS_DIR = DATA_ROOT / "cracks"

OUTPUT_DIR     = Path("outputs2")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PRED_DIR       = OUTPUT_DIR / "predictions"
VIS_DIR        = OUTPUT_DIR / "visualizations"

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME = "CIDAS/clipseg-rd64-refined"
INPUT_SIZE = 352                          # CLIPSeg native resolution

# ── Prompt pools (randomly sampled during training for robustness) ────────────
TAPING_PROMPTS = [
    "segment taping area",
    "segment joint tape",
    "segment drywall seam",
    "segment drywall joint compound"    
]
CRACK_PROMPTS = [
    "segment crack",
    "segment wall crack",
    "segment surface fracture",
    "segment drywall crack"
]

# ── Training ──────────────────────────────────────────────────────────────────
SEED         = 42
BATCH_SIZE   = 8
NUM_EPOCHS   = 30
LR           = 4e-4
WEIGHT_DECAY = 1e-4
BCE_WEIGHT   = 1.0
DICE_WEIGHT  = 1.0
THRESHOLD    = 0.5
NUM_WORKERS  = 0                          # 0 is safest on Windows

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")