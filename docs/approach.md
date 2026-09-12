# Approach

What the problem actually is, what the literature says to expect from it, and the
sequence I would run. Written **before** the server survey, so the one genuine branch
point — whether the marker signal is an image or a number — is left open and both
branches are costed. Everything else below is the same either way.

Nothing here is a commitment. It is the argument the first experiments should test.

---

## 1. This problem has a name, and a decade of prior art

Predicting a fluorescence channel from an unlabeled transmitted-light image is
**virtual staining** / **in silico labeling**. The field is well developed:

- **[In Silico Labeling](https://www.sciencedirect.com/science/article/pii/S0092867418303647)**
  (Christiansen et al., *Cell* 2018) — the founding result. A deep network predicts
  DAPI/Hoechst nuclear labels, live/dead status, and neuron-vs-other from transmitted
  light alone.
- **[Label-free prediction of 3D fluorescence](https://www.nature.com/articles/s41592-018-0111-2)**
  (Ounkomol et al., *Nature Methods* 2018) — a plain U-Net regressing brightfield to
  fluorescence for DNA, nuclear envelope, nucleoli, membrane and **mitochondria**. The
  mitochondrial result is the closest published analogue to Tomm20 here.
- **[DeepIFC](https://onlinelibrary.wiley.com/doi/10.1002/cyto.a.24770)**
  (Timonen et al., *Cytometry A* 2023) — **the most relevant paper to this project**,
  because it is the same modality: single-cell crops from an imaging flow cytometer, not
  a microscope field.

The novelty here is not the method. It is the biology: four sperm-compartment antigens,
on an instrument and a cell type nobody has done this on. That is a better position to
write from than a methods claim, and it means we should borrow architecture aggressively
rather than invent.

## 2. What DeepIFC says we should expect

DeepIFC trained an Inception U-Net on 527,107 PBMC events, mapping two brightfield
channels plus darkfield to one fluorescent channel at a time. Its per-marker results are
the single most useful calibration available to us:

| Marker | Pearson r | Why |
| --- | --- | --- |
| CD45, CD14, CD3 | 0.87 – 0.90 | track cell type, which has a visible morphology |
| 7-AAD (nuclear) | 0.79 | nucleus is visible |
| CD19, CD56, CD8 | 0.41 – 0.61 | the authors state the differences "were not found to be visible to the human eye in brightfield" |

**The rule that falls out: a marker is predictable exactly to the extent that what it
binds has a brightfield correlate.** Not "is it a strong stain", not "is it abundant" —
is the structure it labels *visible*.

Applying that rule to our four:

| Marker | Structure | Prediction | Reasoning |
| --- | --- | --- | --- |
| **DAPI** | nucleus / head | **easiest.** Expect high r | The head is the dominant object in the frame. Every ISL-family paper finds nuclear stains easiest. Do this one first as the pipeline test. |
| **Tomm20** | midpiece | **moderate–good** | The midpiece is a real, resolvable thickening behind the head, and Ounkomol got mitochondria from brightfield in a microscope. |
| **LDHC** | principal piece (tail) | **moderate** | `sperm_pbmc` established the flagellum is clearly resolved at this magnification, including on curled cells. But LDHC is a *soluble* enzyme in the principal piece — its abundance is not purely a function of tail length. |
| **ACRV1** | acrosome | **hardest, and the interesting one** | The acrosome is a subtle intensity step within the head, and ACRV1 signal reflects acrosomal *integrity* — intact vs reacted — which is a biochemical state, not obviously a morphological one. |

ACRV1 is where I would expect the model to fail, and that failure is worth having. There
is precedent for the same question in livestock: a
[boar sperm study](https://pmc.ncbi.nlm.nih.gov/articles/PMC12368927/) binned PNA
acrosome fluorescence into intensity classes and trained a label-free classifier against
them — i.e. quantised the regression into an ordinal problem, which is a sensible
fallback if continuous ACRV1 regression will not converge.

**Set the expectation now:** four markers, and a defensible result probably looks like
two that work well, one that works partially, and one that does not. That is a finding,
not a failure — "brightfield carries midpiece and nuclear information but not acrosomal
integrity" is a publishable sentence. Planning for four successes is how projects end up
reporting four mediocre ones.

## 3. The branch point: is the target an image or a number?

The survey decides this (see [task_brief.md](task_brief.md), open question 1). The two
branches are different problems and it is worth being explicit about both.

### Branch A — paired marker *images* (image → image)

The standard virtual-staining setup. Recommended shape:

- **A single U-Net, one input channel (brightfield), four output channels** — not four
  independent networks. The four markers are anatomically ordered along one cell; the
  encoder that finds the head, midpiece and tail is the same encoder for all four.
  Shared encoder, four light decoder heads gives you "a model per marker" at the output
  while sharing what should be shared. Joint-vs-separate is cheap to A/B, so test it
  rather than arguing about it — but start joint.
- **Small.** 248 × 248 single-channel crops with one centred object is a much easier
  geometry than a microscope field. A ~5–10M-parameter U-Net is the right starting size;
  reach for a ViT backbone only if it plateaus.
- **Loss: L1, on variance-stabilised targets** (`log1p` or `sqrt` of intensity —
  fluorescence is heavy-tailed and plain MSE will be dominated by the brightest few
  percent of events). Add a per-image correlation term if predictions come out flat.
- **No adversarial loss to begin with.** A GAN makes predictions look right, which is
  precisely the wrong failure mode for a measurement we intend to treat as a readout.
  Blurry-but-calibrated beats sharp-but-invented here.

### Branch B — per-event marker *intensities* from the cytometer (image → scalar)

If the marker signal is an FCS/CSV column rather than a picture, this is image-to-scalar
regression: a small CNN, or the `sperm_pbmc` morphology-feature module plus gradient
boosting, predicting four numbers per event.

This branch is *easier to fit and much harder to trust*, because the spatial check
disappears — you can no longer ask "did the predicted ACRV1 land on the acrosome?", which
is the strongest evidence that a model learned biology rather than a correlate. If we end
up here, substitute:

- **occlusion / attribution maps** — mask the head and see whether predicted DAPI drops;
  mask the tail and see whether LDHC drops. That is the same question asked indirectly.
- the negative-population controls in §5, which do not depend on the target being spatial.

Branch B also makes the feature-based baseline the *primary* model rather than a control,
at least initially — with a few hundred to a few thousand events it will beat a CNN, and
every one of its decisions is attributable to a measurable quantity.

## 4. The failure mode most likely to produce a fake result

`sperm_pbmc` Finding 1, restated for a regression target: **a model given raw crops can
read the acquisition off the background noise floor.** Corner patches containing no cell
at all separated `1S` from `1P` at AUC 0.992, because the wells had background σ of 5.5
and 10.7.

Now make the target a fluorescence intensity, which also scales with detector gain and
laser power per acquisition. If replicate 2 ran at a different gain than replicate 3, then
background brightness predicts target brightness — directly, with no biology in between.
A model can score well on a random split and have learned nothing.

Three things follow, and I would treat all three as non-negotiable:

1. **Background-normalise every input**, in SNR units estimated per image from corner
   patches, exactly as `sperm_pbmc/analysis/cytpix_features.py` does. Never feed raw
   crops.
2. **Carry the background-only control forward as a permanent negative control.** Fit
   the same regression on corner patches alone. Its score is the floor: any real result
   has to clear it by a wide margin, and the gap is the headline number, not the raw r.
3. **Hold out by replicate, not by event.** Train on `2*`, test on `3*`, and report that
   number as the result. Event-level random splits within one acquisition will look
   better and mean less. The
   [virtual staining generalisation study](https://arxiv.org/abs/2407.06979) (772,416
   paired image sets) found exactly this: within-condition performance overstates
   transfer, and how the training data was chosen dominates whether the model moves.

## 5. Evaluation: what to measure, and what not to trust

The virtual-staining literature is consistent that pixel metrics mislead. SSIM, PSNR and
MSE can all look good on an image that is biologically wrong, and
[work on downstream utility](https://pmc.ncbi.nlm.nih.gov/articles/PMC12324553/) shows
image-quality scores tracking poorly with whether the output is usable for the task it
was made for. So pixel metrics are a sanity check, never the result.

What I would actually report, in order of weight:

1. **Per-event correlation of predicted vs measured integrated intensity, per marker,
   held out by replicate.** Directly comparable to DeepIFC's table. This is the headline.
2. **Compartment localisation (Branch A only).** Threshold the true marker frame, threshold
   the predicted one, report Dice — *and* check the predicted mass lands on the right
   structure. A predicted Tomm20 that correlates well but paints the whole cell is not
   predicting Tomm20.
3. **Biological negative controls, which this dataset gives away for free:**
   - **PBMC wells (`2P`, `3P`).** No acrosome, no midpiece, no flagellum. Predicted
     ACRV1 / LDHC / Tomm20 on PBMCs should be near zero. DAPI should *not* be — PBMCs
     have nuclei, and a model predicting zero DAPI on a PBMC has learned "sperm" rather
     than "DNA". This single test separates the two hypotheses cleanly.
   - **Round vs straight sperm.** A curled sperm still has a tail; predicted LDHC should
     follow the flagellum, not the silhouette.
4. **Permutation control.** Re-pair each brightfield image with another event's marker
   target and retrain. Anything above chance means leakage, not biology.
5. **Pixel metrics (SSIM/PSNR/MAE)** last, reported for comparability, interpreted lightly.

## 6. Where genuine self-supervised pretraining fits

Distinct from the free-label framing (see [task_brief.md](task_brief.md)) and worth
separating in the methods section.

We have roughly 180,000 unlabeled brightfield events across replicate 1 alone, plus the
brightfield frames of the marker replicates. That is enough to pretrain a representation
with masked autoencoding or DINO-style distillation, and there is direct precedent for it
in this domain:
[masked autoencoders for microscopy](https://arxiv.org/abs/2404.10242) (Recursion, CVPR
2024) showed ViT MAEs beating weakly-supervised classifiers on cellular tasks and
introduced a channel-agnostic variant that accepts a different number and order of
channels at inference — relevant if replicates 2 and 3 turn out not to carry identical
channel sets. [Cell-DINO](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1013828)
(2025) does the DINOv2 equivalent for fluorescent cell images.

**But I would not start here.** Pretraining buys label efficiency, and in this task the
labels are free — every event in the marker replicates is a training pair, so there may
be ~90,000 pairs per replicate. Pretraining is the right move when paired data is scarce;
we should find out whether it is scarce before spending a GPU-week on it. Two conditions
would change that:

- the survey shows far fewer paired events than expected, or the pairing is unreliable; or
- the supervised baseline plateaus and we want a representation that saw all 180,000
  replicate-1 events, not just the paired ones.

There is also a *cheap* version worth doing regardless: pretrain the U-Net encoder as a
plain brightfield autoencoder or masked autoencoder on replicate 1, then fine-tune for
prediction. That is an afternoon, not a research programme, and it is a clean ablation.

## 7. The sequence I would run

Each step has a decision attached, so a bad result stops the next step rather than being
absorbed into it.

| # | Step | Decision it produces |
| --- | --- | --- |
| 1 | **Survey the marker zips** (`inspect_zip.py --structure --peek-metadata`) | Branch A or B; channel identities; bit depth |
| 2 | **Sample ~300 events/class, pull to the laptop, look at them** | Are the channels registered? Is the signal where the biology says it should be? If ACRV1 does not sit on the acrosome in the raw data, stop — that is an imaging problem, not a modeling one |
| 3 | **Feature baseline + background-only control** on the sample | The floor, and how much is trivially available. Cheap, no GPU, runs on the laptop |
| 4 | **DAPI only, single U-Net, replicate-held-out** | Does the pipeline work at all? DAPI is the easiest channel; if it fails, nothing downstream is worth running |
| 5 | **All four markers, shared encoder** — on the server | The per-marker table that is the actual result |
| 6 | **Negative controls (§5) + permutation control** | Whether to believe step 5 |
| 7 | *If* step 5 plateaus: SSL pretraining on replicate 1 | Whether representation was the bottleneck |

Steps 1–4 are laptop work against the sample. Step 5 onward is the server, matching the
`sperm_pbmc` division of labour.

## 8. Open questions I cannot answer from here

- **Are the marker frames registered to the brightfield frame?** If they are not
  pixel-aligned, Branch A collapses toward Branch B whatever the file layout says.
- **How much dynamic range survives the export?** Fluorescence spanning four decades
  quantised to uint8 is a different regression problem than uint16, and may force a
  log or ordinal target.
- **Do `2*` and `3*` differ by more than replicate** — a different gain, a different
  antibody lot, a different day? If they are a genuine batch pair, the held-out-replicate
  split is the right evaluation. If they differ in some *other* systematic way, that
  split measures something else and we need a third grouping.
- **Was the staining panel the same in both replicates?** Four markers in one panel
  implies four-colour compensation; spectral spillover between channels would put
  correlated signal in channels that should be independent, which a per-marker model
  would happily learn and we would happily misread.

## Sources

- [In Silico Labeling: Predicting Fluorescent Labels in Unlabeled Images](https://www.sciencedirect.com/science/article/pii/S0092867418303647) — Christiansen et al., *Cell* 2018
- [Label-free prediction of three-dimensional fluorescence images from transmitted-light microscopy](https://www.nature.com/articles/s41592-018-0111-2) — Ounkomol et al., *Nature Methods* 2018
- [DeepIFC: virtual fluorescent labeling of blood cells in imaging flow cytometry data with deep learning](https://onlinelibrary.wiley.com/doi/10.1002/cyto.a.24770) — Timonen et al., *Cytometry A* 2023 ([preprint](https://www.biorxiv.org/content/10.1101/2022.08.10.503433v1.full))
- [Label-free prediction of cell painting from brightfield images](https://www.nature.com/articles/s41598-022-12914-x) — Cross-Zamirski et al., *Scientific Reports* 2022
- [Can virtual staining for high-throughput screening generalize?](https://arxiv.org/abs/2407.06979)
- [On the Utility of Virtual Staining for Downstream Applications](https://pmc.ncbi.nlm.nih.gov/articles/PMC12324553/)
- [Masked Autoencoders for Microscopy are Scalable Learners of Cellular Biology](https://arxiv.org/abs/2404.10242) — Kraus et al., CVPR 2024
- [Cell-DINO: Self-supervised image-based embeddings for cell fluorescent microscopy](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1013828) — *PLOS Comput Biol* 2025
- [Deep learning classification method for boar sperm morphology analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC12368927/) — label-free acrosome health from binned PNA fluorescence
- [Deep Learning Models for Multi-Part Morphological Segmentation of Live Unstained Human Sperm](https://pmc.ncbi.nlm.nih.gov/articles/PMC12115634/) — head / acrosome / nucleus / neck / tail segmentation
- [LDHC: The Ultimate Testis-Specific Gene](https://onlinelibrary.wiley.com/doi/full/10.2164/jandrol.109.008367) — Goldberg et al., *J Androl* 2010
- [Acrosomal marker SP-10 (ACRV1)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7541689/) — intra-acrosomal localisation
