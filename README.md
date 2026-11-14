# channel-adapters

Working draft of a blog post on adapting RGB-pretrained models to multispectral imagery.

**Working title:** More Bands Than Kernels: Does It Matter How You Adapt an RGB Model to Multispectral Imagery?

## Status

| Section | State |
|---|---|
| Title, description, introduction | drafted |
| How timm does it | drafted, with an animation of the repeat, cut and scale |
| The nine options | drafted, with a diagram |
| Setup (datasets, models, training, statistics) | drafted |
| Results (band order, fine-tuning, learning rate, frozen backbone, EuroSAT) | drafted from all runs |
| Limitations and takeaways | drafted |

The draft lives in `posts/channel-adapters/index.qmd`. Every number in it is generated from `results/` and checked by `experiments/check_numbers.py`.

## Reproduce

```bash
micromamba create -f environment.yml && micromamba activate chadapt && pip install -e .
pytest
export CHADAPT_DATA=/path/with/space          # datasets are downloaded and cached here

python experiments/run.py --exp E1 --all      # learning-rate sweep (E1b extends it for ResNet-50)
python experiments/run.py --exp E1b --all
python experiments/run.py --pick-lr           # writes results/lr_choice.json
for e in E0 E2 E3 E4 E4p; do python experiments/run.py --exp $e --all; done
# on a slurm cluster, replace --all with --slurm (see run.py --help)

python experiments/make_results.py            # result figures, tables and numbers.json, from results/ only
python experiments/make_setup_figures.py      # dataset and options illustrations
python experiments/make_tiling_animation.py   # the timm tiling animation
python experiments/check_numbers.py           # every quoted number must be in numbers.json
```

| Experiment | What it is |
|---|---|
| E0 | timm's default under five band orders |
| E1, E1b | learning-rate sweep for every model and option |
| E2 | fine-tuning on Similar-but-Different, 3 label budgets, 5 seeds |
| E3 | frozen-backbone linear probes on Similar-but-Different |
| E4, E4p | fine-tuning and frozen probes on EuroSAT |

Protocol decisions, including what changed between passes and why, are in `docs/decisions.md`.

## Preview

The rendered draft is published to this repository's GitHub Pages site on every push to `main`. To preview locally, install [Quarto](https://quarto.org/) 1.10 and run `quarto preview`.
