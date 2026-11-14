# Contributing

Two authors, one repository, and coding agents on both sides. The workflow below is meant to keep that low-friction and auditable.

## Workflow

1. `main` is protected. Nobody pushes to it directly, humans or agents.
2. Work on a branch named `<owner>/<topic>` (for example `alice/lr-sweep`). An agent working on someone's behalf uses that person's prefix.
3. Open a pull request. CI runs the tests, renders the post, and attaches the rendered site as a downloadable artifact on the PR. Merging to `main` deploys the rendered draft to GitHub Pages.
4. Squash merge. The squash message says what changed and, for experiments, which `results/` files were added.
5. Review comments go on the PR; design decisions that outlive the PR go into `docs/decisions.md` with a date.

## Ground rules for the post

- Every number, figure and claim in `posts/channel-adapters/index.qmd` comes from a file in `results/`. Figures are produced only by `experiments/make_figures.py`, never edited by hand. `experiments/check_numbers.py` cross-checks the numbers in the post against `results/` and fails CI on a mismatch.
- Until a result exists, the text says `**TODO(results: E2)**`, naming the experiment that will supply it.
- The voice is "we". Nothing is attributed to either author beyond what they actually said or ran.
- `docs/style_notes.md` is the style reference; `docs/scope.md` is the scope reference. Change the document first if you want to change the rule.

## Working with coding agents

- Agents work on branches and open PRs like anyone else. The PR description states which commands were run and which `results/` files were produced, so the reviewer can tell a measured number from a typed one.
- An agent may add, rerun or regenerate results, but may not delete or edit an existing `results/*.json` by hand. Rerun the experiment instead; the run id is deterministic, so the file is replaced.
- Point agents at `README.md`, `docs/decisions.md` and this file. They contain everything an agent needs to reproduce a figure.
- Keep agent-written prose out of the post unless a human has read it against the results it cites.

## Setup

```bash
micromamba create -f environment.yml
micromamba activate chadapt
pip install -e .
pytest
```

Quarto 1.10.19 is needed to preview the post (`quarto preview`). Data is cached under `$CHADAPT_DATA` (default `~/.cache/channel-adapters`) on first use.
