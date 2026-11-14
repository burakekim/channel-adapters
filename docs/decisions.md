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

## 2026-11-12: after the first full pass

- **Learning-rate grid extended for the ResNet (E1b).** In E1 the best ResNet learning rate was the top of the grid (3e-3) for every adapter, so the grid did not bracket the optimum. Added 1e-2 and 3e-2 for the ResNet only. The picker reads E1 and E1b together; any adapter whose pick changes has its E0/E2/E4 ResNet rows rerun at the new rate (run ids include the rate, so old files stay and the figure script filters to the chosen rate).
- **Figure 8 (CKA versus fine-tuning gap) dropped from the post.** Linear CKA at initialisation between the adapted first layer and the RGB first layer is about 0.7 for timm-repeat, mean-RGB and random init alike, and exactly 1 for rgb-only, pseudo-rgb and split-stem, which end up 0 to 7.5 points apart after fine-tuning. The diagnostic does not track difficulty; the numbers stay in `results/E5/` and the post says so in one sentence.
- **Numbers quoted in prose.** `make_figures.py` also writes the derived gaps (adapter minus timm-repeat, adapter minus rgb-only, spread of the non-bottleneck adapters) into `numbers.json`, so `check_numbers.py` can verify them.
- **Headline, as the data supports it.** With full fine-tuning the five full-view adapters tie within 0.9 (ResNet-50) and 1.8 (ViT-S/16) points on Similar-but-Different; `split-stem` is +0.7 over the default on both and +3.4 on the ViT at 10% labels; `pseudo-rgb` loses 4.2 / 3.6; warmup loses 0.2 / 1.1; band order spread is 0.5 / 0.6. Frozen, the tiled stem beats RGB-only on Similar-but-Different and loses on EuroSAT. The post's takeaways are limited to these statements.

## 2026-11-14: second full pass (nine options, five seeds)

- **Two options added.** `wavelength-init` gives each band the R, G or B kernel whose Sentinel-2A centre wavelength (ESA SentiWiki, S2 mission, Table 3) is nearest its own, scaled by 3/N like timm; it isolates whether position-based assignment matters. `stem-lr10` keeps timm's initialisation and trains the first layer at 10x the base learning rate; it is the common alternative to a warmup stage.
- **Five seeds everywhere.** E0, E2 and E4 use seeds 0-4; probes use 5 labelled subsets. Comparisons with timm-repeat are paired by seed (shared labelled subset) and reported with 95% t-intervals.
- **Warmups as fractions of training.** The learning-rate ramp-up is 5% of steps and the `warmup` option's first-layer-only stage is 15% of steps. Previously both were fixed numbers of epochs (1 and 3), which at 1% labels meant 1 and 3 steps out of 200. At the full budget the two definitions give identical step counts (97 and 291 of 1940 on Similar-but-Different, 64 and 192 of 1280 on EuroSAT), so only runs at 10% and 1% labels were rerun. Results at those budgets without `lr_warmup_steps` in their JSON predate the change and are excluded by `make_results.py`.
- **Learning rates.** Selected per model and option on Similar-but-Different validation (seed 0, full labels) from E1 plus E1b. Every selected rate is interior to the swept range. The same rate is used at all budgets and on EuroSAT; whether low budgets prefer other rates is not tested.
- **Dropped.** The CKA diagnostic (E5) did not separate options and is not reported. The convergence-speed metric is still logged but not reported.
