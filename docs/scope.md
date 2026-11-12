# Scope of the first pass

Extracted from [`slack_thread.md`](slack_thread.md). Quotes are verbatim from the thread; everything else is a reading of it. Where the thread is ambiguous, the decision is recorded in [`decisions.md`](decisions.md).

## (a) Ideas Caleb proposed

1. **The post itself.** "I was reminded earlier today that I wanted to take a stab at showing people how ViT and Conv channel adapters for multispectral data work" (7:48).
2. **The problem statement.** "people often want to take pre-trained models, then use them on inputs with a different number of channels. timm allows people to do this by repeating the R,G,B kernels to the number of desired input channels and does a scaling step." (7:57) "You can also adjust the patch embedding layer in ViTs to do a similar thing" (7:59).
3. **The hypothesis.** "My intuition is that both of these will result in a network that is much harder to fine-tune" because the rest of the network "were expecting a certain distribution from the first layer, and now that 'contract' is completely broken" (8:01).
4. **Adapter A: pseudo-RGB projection.** "inserting a new conv / projection layer before the original first layer that squashes your input back to 3 dimension pseudo-RGB" (8:03).
5. **Adapter B: first-layer warmup.** "defining cute warmup routine for adapting just the first layer to the new multispectral input, before the rest of the network" (8:03).
6. **Adapter C: split stem.** "put the RGB bands through the original weights, then have a seperate conv that operates on the remaining bands" (8:03).
7. **Conv versus ViT comparison.** "showing the similarity/differences between the conv adapt case and the ViT adapt case" (8:04).
8. **Downstream transfer effect.** "what affect does this have when you are actually doing transfer learning" (8:05), agreeing with Burak's 8:01 message.
9. **Gradient surprise.** "you can actually show the 'surprise' of the network by measuring the difference in the gradients you get at the adapted layer vs. with RGB inputs" (8:08), with the intuition at 8:10 to 8:11 that layers around the edited one "will want to jump in a crazyyyy direction".
10. **Framing of the post.** "I don't want to get tooo much into the weight dynamics of transfer learning here, but I think doing a nice set of experiments that give a better recipe for adapting pre-trained models would be helpful for folks" (8:12).

## (b) Ideas Burak proposed

1. **Characterize the damage during fine-tuning.** "It would be fun to characterize that also around fine tuning, how much suboptimal timm repeat/rescale damages etc" (8:01).
2. **Two-stage structure.** "first some fun playing around and then analyzing this down-stream" (8:06).
3. **First-layer statistics and CKA as a predictor.** "computing first-layer output stats (maybe CKA) on RGB vs RGB+ inputs and see if that mismatch predicts how hard fine-tuning gets" (8:07).
4. **How far the surprise propagates.** "(and how far the surprise propagates)" (8:09).
5. **How much of the transfer learning survives.** "as the weights go crazy, how much of the transfer learning survives would be a nice question to pose" (8:14).
6. **Blog versus paper split.** "anything beyond offering a better recipe is probably better in a paper not a blogpost" (8:15).
7. **Collaboration mechanics.** "I can invite you to a repo and we can iterate directly on the html?" (8:16).
8. Shared, loosely related reading: the Spectral-Spatial Interpretation paper (arXiv 2610.21114), which Caleb judged "Maybe" relevant. Not used in this pass.

## (c) In scope for this first pass: the recipe

1. Explain, with diagrams and minimal code, how channel adaptation works today: timm's repeat-and-rescale for conv stems and the identical treatment of the ViT patch embedding. The exact behavior is taken from the timm source (`adapt_input_conv` in `timm/models/_manipulate.py`, called from `load_pretrained` in `timm/models/_builder.py`), not from memory.
2. Show the arbitrariness of the default: which pretrained kernel a band inherits depends only on its position in the band list. Demonstrated with a band-to-kernel table and a band-order permutation experiment.
3. Implement and compare the adapters Caleb listed, plus reference points:
   - `timm-repeat`: timm's default (baseline)
   - `pseudo-rgb`: a new 1x1 projection from N bands to 3 before the untouched original first layer (Caleb, 8:03)
   - `warmup`: adapt only the first layer to the multispectral input, then unfreeze the rest (Caleb, 8:03)
   - `split-stem`: RGB bands through the original weights plus a separate conv over the remaining bands (Caleb, 8:03)
   - reference points: `random-init` of the first layer, `mean-rgb-init` (every band gets the mean RGB kernel), and `rgb-only` input.
4. Run everything for one conv model and one ViT and describe similarities and differences (Caleb, 8:04).
5. Measure the downstream effect in transfer learning: linear probe and full fine-tune accuracy, convergence speed, learning-rate sensitivity, across label budgets and seeds (Caleb 8:05, Burak 8:01).
6. End with a concrete recommended recipe, stated only as strongly as the results support (Caleb, 8:12).

Light diagnostic, optional if cheap: first-layer output statistics and linear CKA on RGB versus all-band inputs, and whether that mismatch tracks fine-tuning difficulty (Burak, 8:07). Kept because it costs only forward passes.

## Out of scope: future work (paper material, not implemented here)

Agreed in the thread to belong in a paper rather than the post (Caleb 8:12, Burak 8:15):

- Gradient "surprise" at the adapted layer compared with RGB inputs (Caleb, 8:08).
- How far the surprise propagates through the network (Burak, 8:09).
- Weight dynamics of transfer learning around the edited layer (Caleb, 8:10 to 8:11).
- How much of the pretraining survives once the weights move (Burak, 8:14).
- The Spectral-Spatial Interpretation attribution method (shared reading, not a task).
