# channel-adapters

How should you feed a 12- or 13-band satellite image to a network that was pretrained on RGB? This repository holds the adapters, experiments, results and the draft of a blog post on that question, by Burak Ekim (MIT CSAIL) and Caleb Robinson (Microsoft AI for Good Research Lab), intended for [geospatialml.com](https://geospatialml.com/).

**Status: draft, experiments not yet run.** The post contains `TODO(results: ...)` placeholders wherever a number will go. Nothing in the post is a result until a file in `results/` backs it.

## What is in here

| Path | Contents |
|---|---|
| `docs/slack_thread.md` | The Slack thread the project started from, verbatim |
| `docs/scope.md` | Who proposed what, what is in scope now, what is deferred to a paper |
| `docs/style_notes.md` | How geospatialml.com posts are written, and the checklist the draft follows |
| `docs/decisions.md` | Dated design decisions (datasets, models, resolution, learning-rate protocol) |
| `src/channel_adapters/` | The adapters (`adapters.py`), model helpers, data loading, training loop, diagnostics |
| `tests/` | Unit tests: our re-implementation of timm's tiling equals timm's, adapters start where they should |
| `experiments/` | Experiment grid (`configs/experiments.yaml`), runner (`run.py`), figure script (`make_figures.py`) |
| `results/` | One JSON per run, committed, so every figure regenerates without a GPU |
| `posts/channel-adapters/` | The Quarto post (`index.qmd`) and its figures |

## The adapters

All of them live in `src/channel_adapters/adapters.py` and take an RGB-pretrained timm model:

| Name | What it does |
|---|---|
| `timm-repeat` | timm's default: tile the R,G,B kernels across the N bands and scale by 3/N ([`adapt_input_conv`](https://github.com/huggingface/pytorch-image-models/blob/12c4939be518a6d02d5d3b47b0b2bed49ad32312/timm/models/_manipulate.py#L293)) |
| `random-init` | a freshly initialised first layer |
| `mean-rgb-init` | every band gets the mean of the R, G, B kernels, scaled by 3/N |
| `rgb-only` | keep the model, feed it only the R, G, B bands |
| `pseudo-rgb` | a 1x1 projection from N bands to 3 in front of the untouched first layer, initialised to select R, G, B |
| `split-stem` | R, G, B through the original first layer, the other bands through a zero-initialised parallel copy, summed |
| `warmup` | `timm-repeat` initialisation; train only the first layer and the head for a few epochs, then everything |

The same code handles a ResNet stem and a ViT patch embedding, because timm exposes both as a `Conv2d` named in `pretrained_cfg['first_conv']`.

## Reproduce

```bash
micromamba create -f environment.yml && micromamba activate chadapt && pip install -e .
pytest                                              # ~1 min, CPU
export CHADAPT_DATA=/path/with/space                 # datasets are downloaded and cached here on first use

python experiments/run.py --exp E1 --all            # learning-rate sweep (or --slurm for a job array)
python experiments/run.py --pick-lr                 # writes results/lr_choice.json
for e in E0 E2 E3 E4 E4p; do python experiments/run.py --exp $e --all; done
python experiments/make_figures.py                  # regenerates posts/channel-adapters/*.png and tables from results/
```

One experiment, one command. `run.py --exp E2 --list` prints the runs in a grid and which already have results. Experiments: E0 band-order permutation, E1 learning-rate sweep, E2 main fine-tuning matrix (Similar-but-Different, three budgets, three seeds), E3 linear probes, E4/E4p EuroSAT, E5 first-layer diagnostics. Details in `docs/decisions.md`.

## Preview the post

```bash
quarto preview          # local
quarto render           # writes _site/
```

CI renders the site on every pull request (downloadable artifact) and deploys `main` to GitHub Pages.

## Datasets

- Similar-but-Different: Robinson and Corley, [geospatialml.com post](https://geospatialml.com/posts/similar-but-different/), data on Hugging Face [`calebrob6/similar-but-different`](https://huggingface.co/datasets/calebrob6/similar-but-different), CC-BY-4.0.
- EuroSAT: Helber et al., [arXiv:1709.00029](https://arxiv.org/abs/1709.00029), via [torchgeo](https://torchgeo.readthedocs.io/) and its HF mirror, MIT.

## License

Code: MIT. Post text and figures: CC-BY-4.0.
