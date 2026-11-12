# Slack thread (source of truth)

Copied verbatim from the Slack thread between Burak Ekim and Caleb Robinson, as pasted by Burak on 2026-11-12. Nothing below this line has been edited, paraphrased, or reordered.

---

burak  [7:07 AM]
then put this on my TODO list - to cook and execute an idea with you
image.png 24 repliesCaleb Robinson  [7:48 AM]
I’d be down to do a blog post. I was reminded earlier today that I wanted to take a stab at showing people how ViT and Conv channel adapters for multispectral data work
burak  [7:54 AM]
slightly relevant? https://arxiv.org/abs/2610.21114
image.png Spectral-Spatial InterpretationPost-hoc image explanations usually show where a model looks using spatial heatmaps or indicate which channels matter using global channel scores. These separate views are limiting for multispectral data because channel relevance is often spatially localized and physically meaningful. We introduce Spectral-Spatial Interpretation (SSI), a full-grid attribution method over position-channel players (g,c). For each spatial position, SSI fixes the observed surrounding context, defines a conditional game over the channels at that position, and applies the Shapley value to this local game. The result is a full spectral-spatial attribution matrix rather than a single heatmap or a global channel rank…arXiv.orgCaleb Robinson  [7:57 AM]
Maybe— what I mean is that people often want to take pre-trained models, then use them on inputs with a different number of channels. timm allows people to do this by repeating the R,G,B kernels to the number of desired input channels and does a scaling step.
[7:59 AM]You can also adjust the patch embedding layer in ViTs to do a similar thing
[8:01 AM]My intuition is that both of these will result in a network that is much harder to fine-tune
[8:01 AM]as when the weights of the rest of the network were trained, they were expecting a certain distribution from the first layer, and now that “contract” is completely broken
burak  [8:01 AM]
Gotcha! It would be fun to characterize that also around fine tuning, how much suboptimal timm repeat/rescale damages etc
Caleb Robinson  [8:03 AM]
You can imagine doing a bunch of things:

inserting a new conv / projection layer before the original first layer that squashes your input back to 3 dimension pseudo-RGB
defining cute warmup routine for adapting just the first layer to the new multispectral input, before the rest of the network
[8:03 AM]put the RGB bands through the original weights, then have a seperate conv that operates on the remaining bands
[8:04 AM](and showing the similarity/differences between the conv adapt case and the ViT adapt case)
[8:05 AM]and then yes, as you suggest, what affect does this have when you are actually doing transfer learning
burak  [8:06 AM]
i like the framing! first some fun playing around and then analyzing this down-stream
[8:07 AM]like computing first-layer output stats (maybe CKA) on RGB vs RGB+ inputs and see if that mismatch predicts how hard fine-tuning gets
Caleb Robinson  [8:08 AM]
yep!
[8:08 AM]I think you can actually show the “surprise” of the network by measuring the difference in the gradients you get at the adapted layer vs. with RGB inputs
burak  [8:09 AM]
haha that would be fun
[8:09 AM](and how far the surprise propagates)
Caleb Robinson  [8:10 AM]
For example —- Imagine the original pre-trained network on its last epoch. The learning rate is tiny, and its moving all the weights in really small directions around whatever local optimum it found.
[8:11 AM]Now, all of the sudden you have a different dataset, with different distribution inputs — the gradients are usually pretty big because the objective and thus new local optimum is somewhere completely different
[8:11 AM]but specifically for the layers around the one you edited, my guess is that these will want to jump in a crazyyyy direction
[8:12 AM]I don’t want to get tooo much into the weight dynamics of transfer learning here, but I think doing a nice set of experiments that give a better recipe for adapting pre-trained models would be helpful for folks
burak  [8:14 AM]
lol, and as the weights go crazy, how much of the transfer learning survives would be a nice question to pose
[8:15 AM]but I think yeah, anything beyond offering a better recipe is probably better in a paper not a blogpost
[8:16 AM]how do you collaborate in this agentic era? I can invite you to a repo and we can iterate directly on the html?
