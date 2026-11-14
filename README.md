# channel-adapters

Working draft of a blog post on adapting RGB-pretrained models to multispectral imagery.

**Working title:** More Bands Than Kernels: Does It Matter How You Adapt an RGB Model to Multispectral Imagery?

## Status

| Section | State |
|---|---|
| Title and description | drafted |
| Introduction | drafted |
| How timm does it | drafted, with an animation of the tiling |
| Everything after that | not started |

The draft lives in `posts/channel-adapters/index.qmd`.

The animation is generated from real weights and images by `experiments/make_animation.py`.

## Preview

The rendered draft is published to this repository's GitHub Pages site on every push to `main`.

To preview locally, install [Quarto](https://quarto.org/) 1.10 and run:

```bash
quarto preview
```
