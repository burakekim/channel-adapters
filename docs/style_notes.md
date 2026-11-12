# Style notes: matching geospatialml.com

Observed on 2026-11-12 from the rendered site (the source repo `isaaccorley/geospatialml` is private, so nothing here comes from `_quarto.yml` or a post's `.qmd`). Posts read in full: [Similar but Different](https://geospatialml.com/posts/similar-but-different/index.html), [Solving EuroSAT with as Few Parameters as Possible](https://geospatialml.com/posts/eurosat-min-params/index.html), [ThroughputBench](https://geospatialml.com/posts/throughput-bench/index.html), [Seeing the Roads Through the Trees](https://geospatialml.com/posts/long-range-dependencies/index.html).

## Site

- Quarto 1.10.19, Bootstrap `cosmo` theme plus a custom `styles.css` (Space Grotesk body, JetBrains Mono code, warm background `#f5efe6`, accent `#2563eb`), light and dark themes with a toggle, right-margin table of contents, MathJax, GitHub Pages.
- Tagline: "Weekly dispatches on geospatial machine learning — remote sensing, earth observation, and everything in between." Authors Isaac Corley and Caleb Robinson.
- Listing shows only date, title, and description. Categories appear as pills on the post page only. No thumbnails, so no `image:` field is needed.
- Posts live at `posts/<slug>/index.qmd`; figures and any `figures.js` sit in the same folder.

## Title and teaser

- Title: a short hook, optionally a colon and a plain-language subtitle. Examples: "Similar but Different: A Benchmark for Measuring Whether Models Actually Use Multispectral Bands"; "Seeing the Roads Through the Trees: Do Segmentation Models Actually Use Long-Range Context?"
- Description: one or two sentences, often with a number or a twist. "Spoiler: they mostly don't." / "...to reach 96.04% accuracy using 306 learned parameters."
- Front matter fields visible in the rendered pages: `title`, `description`, `author` (a list of plain names, no affiliations rendered), `date`, `categories`. A BibTeX citation appendix is generated, so `citation: true` is in effect (keyed `robinson2026`, `langid: en`).

## Structure

1. One or two paragraphs of motivation in first-person plural, ending in the question the post answers.
2. A blockquote starting with **Spoiler:** that gives the headline result in numbers.
3. h2 sections: setup or dataset, method, results (sometimes with h3 subsections), conclusion or "What's next", then a **Links** h2 with a bullet list (Dataset, Code, Related).
4. Footnotes carry method details (sweep ranges, refit protocol) instead of cluttering the text.
5. Length: 1,500 to 4,700 words. Target for this post: 2,500 to 3,500.

## Tone

- "We" throughout. Short declarative sentences. Numbers in nearly every results sentence.
- Dry humor in parentheses, exclamation marks rarely. "This post is about that gap!" / "We aren't sure why (but reproduced this several times)!"
- Plain about limits: "This recipe probably only works because EuroSAT is easy." / "These are point estimates without confidence intervals." / "Treat accuracy on this dataset as a measurement of band reliance rather than operational land-cover accuracy."
- Results paragraphs often open with a bold lead: "**Plain ViTs see a much bigger lift on V100 than on H100.**" then "The practical takeaway: ...".

## Code

- The four research posts contain no code blocks. Code lives in a linked public GitHub repo (`calebrob6/...`, `isaaccorley/...`); inline code spans name functions, flags, and files.
- For this post we need to show what timm does, so we allow at most two short code blocks (under ten lines each), both lifted from or equivalent to the timm source and linked to a permalink. Everything else points to the repo.

## Figures and tables

- Static PNG or WebP, matplotlib style, 65 to 90 percent width, with one-sentence captions that interpret rather than describe ("Both models start at ~74-76% recall for adjacent pixels and decay monotonically."). Newer posts ship light and dark variants (`class="theme-light"` / `class="theme-dark"`) and long `alt` text. Figure numbering is used in the longer posts ("Figure 3: ...") and omitted in the shorter ones.
- The Similar but Different results figure is a slope chart: one line per model from RGB to all bands. A natural template for our main figure.
- Tables are plain markdown, no caption, best value in bold, a short note line directly below ("mean ± standard deviation in percentage points over five subsample seeds").
- Interactive figures exist (hand-written JS or D3 in `figures.js`) but are not required. Optional for us; not in the first draft.
- Callouts are used sparingly: a `callout-warning` "Read this first" for a global caveat, a collapsed `callout-tip` primer, a `callout-note` for scope.

## Results and caveats

- State the number, the comparison, and the likely reason, in that order. Tie explanations to concrete mechanisms (pretraining data, architecture), and label guesses as guesses ("a plausible reading is that...").
- Caveats are inline or in the conclusion, not a dedicated Limitations section.
- Conclusions give a recommendation and a generalization, then one line of call to action ("Let us know what happens!").

## Checklist for `posts/channel-adapters/index.qmd`

- [ ] Front matter: title, description, author list (Burak Ekim, Caleb Robinson), date, categories, `draft: true`, `citation: true`.
- [ ] Motivation paragraph, then **Spoiler:** blockquote with numbers from `results/`.
- [ ] h2 sections ending with a recipe, future work, and **Links**.
- [ ] At most two code blocks, both tied to timm permalinks.
- [ ] Every figure has an interpretive caption; every table has a note line; best values bold.
- [ ] Caveats inline. No claims beyond the runs in `results/`.
- [ ] "We" voice; nothing attributed to Caleb beyond the thread.

## Caleb's personal blog (calebrob.com), observed 2026-11-12

Added at Burak's request. Source: [calebrob.com/year-archive](https://calebrob.com/year-archive/) and the public Jekyll source at [github.com/calebrob6/calebrob6.github.io](https://github.com/calebrob6/calebrob6.github.io) (AcademicPages, a Minimal Mistakes fork).

**How his site relates to geospatialml.com.** Every 2026 entry in his archive is a link stub: a Markdown file with `link:` front matter that renders as a link post and sends readers to geospatialml.com, with the body "This post is published on geospatialml.com, the blog I write with Isaac Corley." So the post itself lives on geospatialml.com and his site carries a stub. A ready-made stub for this post is in `posts/channel-adapters/calebrob-stub.md`.

**Two registers.** His pre-2024 personal posts are short (300 to 1,100 words), first-person singular, notebook-derived, with 5 to 30 line Python blocks, inline timings ("399 valid words in 37 milliseconds"), "take 1 / take 1.5 / take 2" progressions, and no conclusion section: they end on the result or a link. His 2026 geospatialml posts are the register we match: "we" throughout, 1,200 to 4,500 words, measured but direct.

**Devices to adopt from the 2026 posts.**
- Hook with a concrete number or question in the first line ("6.1 PB", "~10 GPU-hours and ~2,000 GPU-hours", "Spoiler: they mostly don't"), then a paragraph on why it matters for remote sensing in practice.
- One-line pull quotes as blockquotes ("There is no reason to store float32 embeddings.").
- Results prose names the winner and then the complication in the next sentence ("SegFormer leads on overall metrics... But the U-Net wins on DWR").
- **Takeaways** as a numbered list; each item is a bold imperative sentence followed by one line of evidence ("**Always use int8.** It is statistically indistinguishable from float32 across every model and dataset we tested (p > 0.12).").
- **Limitations** as an explicit list introduced by "We have not tested:", each bullet with a one-clause reason, followed by a recommendation to validate on the reader's own task.
- "Read this first" warning callout when a scope caveat applies to every number.
- Statistical claims carry p-values or bootstrap intervals and the phrase "statistically indistinguishable" where it applies.
- Dated **Update** callouts crediting the person who flagged an error.
- Code only where a reader would copy it; analysis posts show none and link a repo. Demos hosted at calebrob.com/<name>/ with code at github.com/calebrob6/<name>.
- Endings: takeaways, links, or a one-line generalization; never a marketing call to action. A "Try it yourself" section with notebook or Colab notes when there is something to run.
- Collaborators credited inline by first name ("Isaac's recent survey").

**What this changes in our draft.** The recipe section becomes numbered takeaways in that form; limitations become an explicit "We have not tested" list before the takeaways; a short "Try it yourself" section points at the repository's one-command reproduction.
