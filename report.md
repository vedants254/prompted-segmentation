# Prompted Segmentation 

## Model selection and approach

I went with CLIPSeg (CIDAS/clipseg-rd64-refined), a pretrained text-conditioned segmentation model built on CLIP. It already understands the relationship between text and images and has a small decoder on top that produces segmentation masks from text prompts. The model was originally trained on PhraseCut (340K phrase-mask pairs) so it already has a solid foundation for text-guided segmentation. The big advantage here is that I didn't need to build a text-conditioning mechanism from scratch. The CLIP backbone already knows what "crack" and "taping" look like in a general sense, so I just needed to fine-tune the decoder to produce accurate masks for our specific drywall domain.

This is a decoder-only fine-tuning approach. I froze the entire CLIP backbone (both vision and text encoders) and only fine-tuned the existing decoder layers (~1.1M parameters out of 150M total). No new layers were added. The decoder consists of 3 transformer layers with FiLM conditioning and a transposed convolution head. Keeping the backbone frozen preserves CLIP's pretrained text-image alignment while the decoder learns our domain-specific mask outputs. This kept things lightweight and avoided overfitting on our relatively small dataset.

## Data

Two Roboflow datasets:

| | Taping (Drywall-Join-Detect) | Cracks (Cracks-3ii36) |
|---|---|---|
| Format | Bounding boxes → filled rectangles | Polygons → filled masks |
| Train | 820 images | 5,164 images |
| Valid | 202 images | 201 images |

There's a 6:1 imbalance between cracks and taping, so I used a WeightedRandomSampler to make sure both classes show up equally during training.

One important design choice: every image produces **two** training samples: one with the correct prompt and real mask (positive) and one with the wrong prompt and an empty mask (negative). So a crack image is paired with both "segment crack" → crack mask AND "segment taping area" → empty mask. Without this, the model just learns to segment everything regardless of the prompt.

I also randomly varied the prompt wording during training (e.g., "segment crack", "segment wall crack", "segment drywall crack") to make the model more robust to phrasing.

## Training Setup

- **Loss:** BCE + Dice (equal weight)
- **Optimizer:** AdamW, lr=4e-4 with cosine annealing over 30 epochs
- **Gradient clipping:** max_norm=1.0 to stabilize training
- **Augmentation:** Horizontal flip
- **Checkpoint selection:** I saved the model from whichever epoch had the best *min-class* mIoU. That way both classes had to be doing well, not just one carrying the average
- **Seed:** 42, fully deterministic

## Results

Best checkpoint came from epoch 7. Validation metrics at model resolution (352x352):

| Prompt | mIoU | Dice |
|--------|------|------|
| segment taping area | 0.868 | 0.919 |
| segment crack | 0.776 | 0.848 |

Full-resolution evaluation on the validation set:

| Prompt | mIoU | Dice |
|--------|------|------|
| segment taping area | 0.737 | 0.839 |
| segment crack | 0.551 | 0.696 |
| **Overall** | **0.644** | **0.767** |

The drop from model resolution to full resolution is the main issue. More on that below.

## Visual Examples

4 side-by-side comparisons (Original | Ground Truth | Prediction) are included in the visualizations folder, 2 for taping and 2 for cracks.

## What Went Wrong 

**The resolution bottleneck.** CLIPSeg works at 352x352 internally. The final mask needs to match the original image size, so I had to bilinearly upsample the output. This blurs everything, especially thin cracks that are only a few pixels wide. At 352x352 the model was doing a decent job on cracks (0.78 mIoU), but after upsampling it dropped to 0.55. Taping held up better since those are larger rectangular areas.

I ran a probability analysis to understand this better and found that about 28% of crack pixels that the model correctly identified at 352x352 got smoothed below the 0.5 threshold after upsampling. The model knows where the cracks are but the upsampling just destroys that information.

**Cracks plateaued early.** Crack mIoU hit ~0.77 around epoch 7 and didn't budge for the remaining 23 epochs, even though training loss kept going down. Taping kept improving to 0.90+. This makes sense since the ViT patches are 16x16, so a thin crack barely registers in the feature grid.

**Taping labels are coarse.** The taping ground truth comes from bounding boxes, not pixel-level masks. So the model can only be as precise as the box annotations allow.

## Other Things I Tried

**Focal Tversky Loss:** I replaced Dice with Focal Tversky (alpha=0.3, beta=0.7, gamma=1.33) hoping it would penalize missed crack pixels more aggressively. It made basically no difference and cracks still plateaued at the same level. The bottleneck isn't the loss function, it's the resolution.

**Threshold tuning:** I swept thresholds from 0.1 to 0.75 thinking a lower threshold might recover the blurred crack pixels. The pixel-level analysis looked promising, but in practice the false positives outweighed the gains. Threshold 0.5 turned out to be the right call.

## Runtime

| | |
|---|---|
| Training time | ~83 minutes (30 epochs, GPU) |
| Inference | ~39 ms/image |
| Model size | 150.7M params total, 1.1M trainable |

## What I'd Try Next

The main thing holding this back is the resolution gap. A few ideas I'd explore with more time:

- **Sliding window inference:** Run the model on overlapping 352x352 patches of the full image and stitch the results. No upsampling needed and every pixel gets processed at native resolution. This should recover most of the lost crack performance.
- **Guided upsampling:** Use the original high-res image to guide the upsample so edges stay sharp instead of getting blurred.
- **Test-time augmentation:** Average predictions from the original and flipped image for a small consistency boost.
- **Gradual unfreezing:** If the decoder keeps plateauing, carefully unfreeze the last couple of CLIP vision layers with a very low learning rate.

## Reproducibility

Seed 42, deterministic CUDA, all config in `config.py`, training logs in `training_log.json`, prediction masks follow the `{image_id}__segment_crack.png` format.
