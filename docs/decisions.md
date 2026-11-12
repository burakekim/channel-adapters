# Decisions

Dated design decisions that are not derivable from the code. Newest last.

## 2026-11-12: initial design

- **Datasets.** Primary: Similar-but-Different (HF `calebrob6/similar-but-different`, 12 Sentinel-2 L2A bands, 32x32, 10 classes, CC-BY-4.0), because it was built so that RGB alone is insufficient; the adapter therefore matters. Secondary: EuroSAT (torchgeo split, 13 bands, 64x64, 10 classes, MIT), because readers know it, with the caveat that RGB-only already reaches about 98.6%.
- **Models.** Conv: `resnet50.a1_in1k` (first_conv `conv1`). ViT: `vit_small_patch16_224.augreg_in21k` (first_conv `patch_embed.proj`). Both ImageNet checkpoints adapted by timm through `adapt_input_conv`.
- **Input resolution.** Bilinear upsample to 128 px for both datasets and both models. ViT is built with `img_size=128` (64 tokens, position embedding resampled by timm). Chosen to keep a full-budget run near two to three minutes on an L40S.
- **Normalization.** Per-band mean and standard deviation computed on the training split of each dataset, applied to all bands (including the RGB-only condition).
- **Augmentation.** Horizontal and vertical flips and 90-degree rotations only.
- **Adapter initializations.** `pseudo-rgb` starts as an exact RGB selector so that at initialization it equals `rgb-only`. `split-stem` initializes the extra-band conv to zero, so at initialization it also equals `rgb-only`. `mean-rgb-init` gives every band the mean of the R, G, B kernels, scaled by 3/N like timm. `random-init` uses the module's default initialization. `warmup` uses the timm-repeat initialization, trains only the first layer and the classifier head for 3 epochs with everything else frozen, then trains everything for the remaining epochs under the same total epoch budget.
- **Optimizer.** AdamW, weight decay 0.05, cosine schedule with one warmup epoch, mixed precision, batch size 256. 20 epochs at full budget; at 10% and 1% budgets the number of epochs is scaled so the number of optimizer steps is comparable (capped).
- **Learning-rate protocol.** Four learning rates {1e-4, 3e-4, 1e-3, 3e-3} swept on Similar-but-Different at full budget with seed 0 (E1). The best validation learning rate per model and adapter is used for every other full fine-tune run, including EuroSAT. Sensitivity is reported from the full sweep.
- **Linear probe.** Frozen backbone features (global pool for ResNet, CLS token for ViT), multinomial logistic regression, inverse-regularization strength swept on validation, refit on train+val, scored on test. Mirrors the Similar but Different post.
- **Budgets and seeds.** Budgets 100%, 10%, 1% of the training split, stratified by class; seeds 0, 1, 2. EuroSAT at full budget only.
- **Convergence speed.** Defined as the first epoch at which validation accuracy reaches 95% of its final value.
- **Compute.** Slurm array on `vision-beery-l40s`; the local L40S is for smoke tests and timing. Every script is single-GPU.
- **Preview.** Public GitHub repo with GitHub Pages. The post carries `draft: true`. Remote creation waits for explicit confirmation.
