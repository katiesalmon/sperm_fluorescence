# Three paths forward

*2026-09-17. Second research round, run after [GOAL.md](../GOAL.md) fixed the objective.
The first round (in [approach.md](approach.md)) was written before we knew what the data
was, and treated this as image-to-image virtual staining. It is not: it is
**image → four scalars per event**, on single-cell crops, with a strong classical baseline
already measured. That changes which literature is relevant and where the headroom is.*

---

## The finding that should drive the decision

The closest published analogue is
[Towards Label-Free Single-Cell Phenotyping Using Multi-Task Learning](https://arxiv.org/html/2605.14717)
(2026): a CNN+ViT hybrid predicting **CD45 among others** as a continuous value from
label-free single-cell images, with a shared encoder and per-marker regression heads.
Their per-marker results, next to our morphology floor from
[results.md](results.md):

| | Their deep model | Our gradient boosting on 10 shape features |
| --- | ---: | ---: |
| CD45 | **0.799** | **0.767** |
| CD16 (stable lineage) | 0.819 | — |
| lineage markers (CD3/CD19/CD56) | 0.768 | — |
| activation-state markers | 0.339 | — |
| DAPI / DNA content | — | 0.776 |
| ACRV1 (acrosomal state) | — | 0.209 |

**Our classical baseline is within 0.03 of a published deep model on the same marker.**
That is the single most important number in this document. It means:

- **DAPI and CD45 are close to done.** A network might add a few points. It will not
  transform them, because what those markers report — DNA content, leukocyte identity —
  is *cell type*, and cell type is legible in shape. Ten features already capture it.
- **The headroom is entirely in ACRV1 and LDHC**, the two markers that report something
  other than which cell this is.

The same paper states the rule directly: *"stable lineage markers are predictable from
morphology, while transient activation states require molecular assays."* DAPI and CD45
are identity. ACRV1 is state. That is not a modelling distinction, it is a biological one,
and no architecture repeals it.

### And there is a specific, physical reason ACRV1 may be unreachable

[Label-Free Detection of Acrosome Reaction of Human Sperm](https://pmc.ncbi.nlm.nih.gov/articles/PMC13217352/)
(2026) recovers acrosome-reaction status label-free at **AUC 0.907** — but with
**diffraction phase microscopy**, and states plainly that *"the nucleus and acrosome's
internal properties are not visually accessible using conventional label-free imaging."*
Related work recovers it from **birefringence**, localised to the post-acrosomal region in
reacted sperm.

The CytPix is conventional brightfield. It measures absorption, not phase or polarisation.
So the most likely explanation for ACRV1 = 0.209 is **an optical limit, not a modelling
one** — and if that is right, no amount of architecture fixes it, while a phase or
polarisation channel would.

This is worth stating up front because it changes what a negative result means. "We could
not predict ACRV1 from brightfield, and here is the optics literature saying why, and here
is what contrast mechanism would be needed" is a far better outcome than "our CNN got
0.24".

---

## Path 1 — Direct supervised network on pixels

Train a shared-encoder network on masked, background-normalised 16-bit crops with four
regression heads, held out by replicate. The literature-validated recipe, applied to our
data.

**Concretely:** small CNN or CNN+ViT hybrid; input the 248 x 248 uint16 crop masked by the
instrument's layer-1 object mask and normalised in SNR units from corner patches; targets
compensated and `asinh`-transformed; loss Smooth L1 plus a Pearson-alignment term (as the
multi-task paper uses); heteroscedastic head optional, using
[beta-NLL](https://arxiv.org/abs/2203.09168) rather than plain Gaussian NLL, which is
known to destabilise the mean.

### Pros

- **It is the obvious next step and the reviewers' expected one.** Not doing it needs a
  better reason than "we think features are enough".
- **Directly tests the floor.** We have a number for every marker; the network either
  clears it or does not, and either answer is informative.
- **The recipe is de-risked.** Architecture, loss, target transform and per-marker
  expectations all come from a 2026 paper doing the same task on the same kind of data.
- **Uses the 16-bit depth**, which the classical baseline does not — the instrument's
  morphology features were computed from its own 8-bit-ish pipeline, so there is genuine
  unexploited signal in the extra two bits.
- **Uses the masks for free**, removing the background confound by construction rather
  than hoping normalisation handled it.
- **Shared encoder is well-motivated here**: four markers, one cell, anatomically ordered.
- **282,874 paired events** is a comfortable dataset size for a small network.

### Cons

- **The headroom on the two markers it will do best at is ~0.03.** The most likely outcome
  is DAPI 0.78 → 0.82 and CD45 0.77 → 0.82, which is a real but unexciting result.
- **It probably does not fix ACRV1**, for the optical reason above. Spending the main
  effort here risks a lot of compute to confirm the floor.
- **Two replicates is a thin generalisation claim.** One held-out acquisition is a single
  sample of "another batch"; it demonstrates transfer, it does not measure its variance.
- **Subtle cues are what convolutional pooling destroys** — the multi-task paper says so
  explicitly. Getting ACRV1 would need architecture that preserves fine spatial detail,
  which is more work than a standard backbone.
- **Adds a dependency and a GPU** to a repo that currently runs on a laptop and a bare
  server Python.

### Self-supervised pretraining, as an option inside this path

MAE/DINO on all 282,874 crops, then fine-tune. The evidence says **deprioritise it**:
pretraining buys label efficiency, and our labels are free and plentiful. The medical
imaging ablations are consistent —
[SSL's benefit concentrates at low label counts](https://www.nature.com/articles/s41598-023-46433-0).
Worth revisiting only if Path 1 plateaus and we want a representation that also saw
replicate 1's 90,000 unstained events.

---

## Path 2 — Compartment-aware measurement, not a bigger model

The bet that **the bottleneck is what we measure, not how we fit it.** The instrument's
features are whole-object summaries: area, eccentricity, circularity, one texture set over
the whole cell. But our markers are *compartment-specific* — ACRV1 on the anterior head,
LDHC on the principal piece, DAPI on the head. A whole-object feature cannot express
"the anterior third of the head is denser than the posterior third".

**Concretely:** use the layer-1 mask to split each event into head / midpiece / tail by
skeletonising and walking the object; compute intensity, texture and phase-proxy features
*per compartment* on the 16-bit data; feed those to gradient boosting. Reframe ACRV1 as
ordinal bins (intact / intermediate / reacted), following the
[boar sperm PNA study](https://pmc.ncbi.nlm.nih.gov/articles/PMC12368927/) that did
exactly this with acrosome fluorescence.

### Pros

- **Aimed squarely at the headroom.** It targets ACRV1 and LDHC, the only markers with
  room, rather than optimising the two that are nearly done.
- **Tests the biological hypothesis directly.** "Does the tail region predict LDHC?" is a
  question a compartment feature answers and a black box does not.
- **Cheapest path by a wide margin.** No GPU, runs on the laptop, days not weeks.
- **Every result is attributable** to a measurable quantity, which is what makes it
  defensible in a write-up and diagnosable when it fails.
- **Robust to the batch problem** almost for free: compartment *ratios* (anterior/posterior
  head density) are internally normalised, so acquisition gain largely cancels.
- **Ordinal ACRV1 sidesteps the hardest part of the regression** — we may be able to
  separate intact from reacted without predicting an intensity.
- **The masks make it tractable.** Segmentation is already done and verified against
  `NumPixels`; this is measurement on top of it, not a segmentation project.

### Cons

- **Lower ceiling on DAPI and CD45.** It will not beat a network on the identity markers,
  and probably will not beat the existing floor there by much.
- **Compartment splitting is real work and can fail** on curled sperm, which are precisely
  the ambiguous cases — `sperm_pbmc` found the flagellum looped back on itself, which
  breaks naive head-to-tail walking.
- **It may simply confirm the optical limit.** If the acrosome is not visible in
  absorption contrast, no feature computed from absorption will find it.
- **Feature engineering is unfashionable** and a reviewer may ask why there is no network,
  even though the floor result is a good answer.
- **Ordinal binning needs a threshold**, and the compensated ACRV1 distribution is not
  obviously bimodal — the bins may be arbitrary.

---

## Path 3 — Change the target: rank within acquisition, and gates instead of intensities

The bet that **absolute intensity is the wrong quantity to predict.** Replicate 2 and 3
disagree by 3.6x on LDHC in the PBMC wells and the sign of its well-separation reverses.
Predicting an absolute compensated intensity means predicting a number whose meaning
shifts between acquisitions. Predicting *where an event sits within its own acquisition*
does not.

**Concretely:** replace the target with the within-acquisition quantile of each marker, or
with the positive/negative call an operator would gate. Evaluate with per-acquisition
Spearman, which is what a rank target optimises. Optionally add batch-aware normalisation
in the network, following
[plate-aware batch normalisation](https://ouci.dntb.gov.ua/en/works/4O2pqWq7/).

### Pros

- **It is what the assay is actually used for.** Nobody acts on "ACRV1 = 8,341". They act
  on "this sperm is acrosome-intact". A gate is the decision-shaped output.
- **Immune to the gain and staining differences** that make our two replicates
  incomparable — and we have measured those differences, they are not hypothetical.
- **Makes the two-replicate limitation survivable.** Rank transfer is a weaker claim than
  intensity transfer, but it is a claim we can actually support with n=2 acquisitions.
- **CD45 becomes a clean binary with free labels**, which is directly useful to
  `sperm_pbmc` — and unlike the DAPI-based signal, it is a true leukocyte call.
- **Sidesteps the compensation mess partly**: rank is invariant to any monotone transform,
  so it survives a spillover correction that is right in direction and wrong in magnitude
  — which describes our matrix, with its suspect 1.0001 coefficient.
- **Pairs naturally with ordinal ACRV1** from Path 2.

### Cons

- **It discards information.** If absolute intensity *is* predictable, ranking throws away
  the part that would have been most impressive.
- **Rank targets are not comparable across acquisitions either** — an event at the 90th
  percentile of a weakly-stained run is not equivalent to the 90th of a strong one. It
  moves the batch problem rather than solving it.
- **Gating needs a threshold we do not have.** Operator gates were not exported with the
  data; we would be inventing them, and the compensated distributions would have to
  justify where.
- **Formal domain-generalisation methods are a weak bet here.** IRM and its variants
  [are unreliable and frequently fail to beat well-tuned ERM](https://arxiv.org/html/2401.17541v3),
  and two environments is the theoretical *minimum*, not a comfortable number. Plate-aware
  batch norm is the modest, robust version; IRM/DANN is not worth the complexity at n=2.
- **Weaker headline result.** "Spearman 0.8 on within-run rank" reads as less than
  "r = 0.8 on intensity" even when it is the more honest quantity.

---

## Recommendation

**Run Path 2 first, then Path 1, and adopt Path 3's evaluation throughout.**

They are not mutually exclusive and the ordering is driven by cost and by where the
headroom is:

1. **Path 2 is days of laptop work** and aims at the only markers with room. If
   compartment features move ACRV1 off 0.209, that is the result of the project. If they
   do not, we have strong evidence the limit is optical — which is a finding, and one the
   phase-microscopy literature corroborates.
2. **Path 1 then answers the question that will be asked anyway**, with Path 2's result as
   the floor it has to clear, and with the compartment features as a diagnostic for what
   the network is using.
3. **Path 3 is not really a separate project** — it is a decision about what to report.
   Adopting per-acquisition rank as a *co-primary* metric costs nothing, and it is the
   metric that survives our two-replicate limitation. Adopt it now; keep absolute
   intensity alongside.

**What would change this:** if the lab can add a phase or polarisation channel to a future
run, ACRV1 moves from "probably unreachable" to "the most interesting target in the panel",
and Path 1 on that data becomes the whole project.

## Sources

- [Towards Label-Free Single-Cell Phenotyping Using Multi-Task Learning](https://arxiv.org/html/2605.14717) — 2026; the closest analogue: CD45 r = 0.799, activation states 0.339
- [DeepIFC](https://onlinelibrary.wiley.com/doi/10.1002/cyto.a.24770) — imaging flow cytometry, CD45 r = 0.90, markers without morphological correlates 0.41–0.61
- [Label-Free Detection of Acrosome Reaction of Human Sperm Based on Diffraction Phase Microscopy](https://pmc.ncbi.nlm.nih.gov/articles/PMC13217352/) — AUC 0.907 from *phase*, and the statement that acrosomal internals are not accessible to conventional label-free imaging
- [Combined Raman and polarization sensitive holographic imaging of human sperm](https://www.nature.com/articles/s41598-019-41400-0) — birefringence localises to the post-acrosomal region after the reaction
- [Deep learning-enabled morphology analysis of bovine sperm for label-free imaging flow cytometry](https://www.frontiersin.org/journals/veterinary-science/articles/10.3389/fvets.2026.1634224/full) — 2026, same modality and species-adjacent
- [Deep learning classification for boar sperm morphology](https://pmc.ncbi.nlm.nih.gov/articles/PMC12368927/) — acrosome health as binned PNA fluorescence classes
- [On the Pitfalls of Heteroscedastic Uncertainty Estimation](https://arxiv.org/abs/2203.09168) — beta-NLL; why plain Gaussian NLL destabilises the mean
- [Towards Understanding Variants of Invariant Risk Minimization](https://arxiv.org/html/2401.17541v3) — IRM variants frequently fail to beat tuned ERM
- [Incorporating knowledge of plates in batch normalization](https://ouci.dntb.gov.ua/en/works/4O2pqWq7/) — the modest, robust batch-aware option
- [Self-supervised pre-training with contrastive and masked autoencoder methods](https://www.nature.com/articles/s41598-023-46433-0) — SSL's benefit concentrates at low label counts
