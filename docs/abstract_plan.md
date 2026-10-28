# Abstract plan: three abstracts on one backbone

*2026-10-28. The ImageStream abstract (stain prediction accuracy) is submitted and the A8 one
(prediction accuracy, sperm vs PBMC separation) is drafted. These three build on them. Each
claims one thing, uses one population engine result as its figure, and leaves its own
validation open for the next.*

## The backbone every abstract shares

label-free images → stems U-Net → virtual stains → aligned stack → {feature engine, encoder latent}
→ populations (gates, Gaussian mixture, galleries) → labels from measurements the model never saw.

Say it once, in the same words, in all three methods sections. The abstracts differ only in
the instrument, the population asked about, and the truth used to check it.

## 1. Whole-ejaculate morphologic classification (ImageStream)

**Claim.** Virtual fluorescence turns an unstained ImageStream acquisition into a
compartment-level morphological census of every object: unsupervised populations in the
virtual-stain space match the populations found on the real stains, and within sperm the
dominant structure is tail configuration.

**What exists for it (results.md).** 4,000 held-out objects; 1,025-column feature table with
Area/Width/Aspect Ratio reproducing IDEAS exactly; k = 2 stained vs unstained at ARI 0.82
with the measured side (latent) and stability 0.99; within the 1,956 stained objects, k = 3
= extended tail / folded tail / large bright bodies, ARI 0.54 (features) with stability 0.79;
all-feature clustering ARI 0.77 at k = 3; the stain-only galleries and the comparison figure.

**The figure.** The two-stage stain-only gallery: k = 2 on everything, then k = 3 within the
stained objects, measured stains used only to name the clusters.

**What it should NOT claim.** WHO % normal or any defect prevalence: the masks are stain halos
(heads measure 7 µm, not 4) and no technician labels exist. Say "compartment-level
populations", not "morphology grades".

**What to add before submission (a week, no new data).** (a) Native-resolution stacks and a
tighter mask level, so head dimensions land near WHO's; (b) the autoencoder trained on the
training half, not the held-out half; (c) the feature-against-feature Bland–Altman table as
a supplementary. (d) One sentence of domain-shift caveat: every image is of a fixed, stained
cell.

**Tees up #3** by establishing that the compartments are measurable and that sperm
subpopulations exist in the space; #3 then asks which subpopulation is "ideal".

## 2. Sperm vs not sperm in light-loss images (A8 / S8)

**Claim.** From the A8's label-free light-loss, FSC and SSC images alone, virtual
fluorescence separates sperm from leukocytes and debris with no supervision: the latent
space of an encoder trained only on virtual stains splits 600 held-out objects into the
sperm and PBMC wells exactly, and the mixed well's spiked leukocytes land with the PBMCs.

**What exists.** The 40-epoch A8 model (LDHC within sperm 0.75, CD45 within PBMC 0.61);
all-feature k = 2 and latent k = 2 both exact on the wells (ARI 0.99 with the measured
side); predicted CD45 alone AUC 0.95; the inputs-only autoencoder fails (111 sperm
misplaced), so the virtual stain step is shown necessary; the k = 2 comparison figure.

**The figure.** The k = 2 stain-only gallery on all 600 objects, features beside latent,
with the well contingency table.

**What to add.** (a) The same on the server's 60,000-object replicate: this is the one
abstract where the laptop sample is too small to publish; (b) a supervised head on the
latent with the sperm-well-vs-PBMC AUC on the held-out replicate, so the claim is a
classifier, not just a clustering; (c) a second patient if one exists; (d) the sort-rate
latency estimate, one sentence, because the audience is ICSI.

**Honest limits.** Wells are population labels, not cell labels (both impure); no round
cells or debris were in the panel, only PBMCs; one donor.

**Tees up #3** by delivering the sperm population that #3 selects from, and by
showing the encoder works on CellView images.

## 3. Ideal sperm selection from CellView images (A8 / S8, Kruger strict)

**Claim (the goal).** Within the sperm population, virtual fluorescence ranks sperm by how
close they sit to the Kruger-ideal form, and the ranking agrees with a technician's strict
scoring of the same cells.

**What exists.** Within-sperm clustering on the A8 (straight / looped / faint tails, stability
0.60–0.80) and on the ImageStream (extended / folded, 0.79); the proposed Kruger encoder and
regional-head architecture (figures/arch_kruger.png, kruger_encoder.png); the feature
engine's WHO-named measurements with Area/Width calibrated against IDEAS.

**What does not exist, and is the whole abstract.** Technician tags. Nothing in Kruger
can be claimed from stains alone: the measured stains give geometry, but "ideal" is a
judgement, and the only truth is a technician scoring the same cells in the WHO order.
A few hundred A8 light-loss images (and the same for ImageStream brightfield) tagged into
the four regional flags is the single data-collection task that unlocks this abstract.

**Also needed.** The A8's resolution is the hard limit: 0.333 µm/px after resampling, a head
6–12 px wide, no DAPI. Head shape on the A8 is marginal, so #3 on the A8 is most honestly
framed as tail- and midpiece-based ideal-form selection, with head criteria from the
ImageStream in a companion. The native-resolution stacks and contrastive encoder are
prerequisites; so is the server-scale run.

**The figure.** Regional-head precision against technician tags, and a gallery ordered by
p(ideal) with the technician's call under each cell.

## Order and dependencies

```
now ──────► #2 (A8 sperm vs not): server run + supervised head; two weeks, no new labels
      └───► #1 (ISX morphology): native stacks, tighter masks, encoder on the training half; one week
                  └───► #3 (Kruger): needs technician tags (collect now, in parallel), then the
                                     contrastive encoder and regional heads; months, not weeks
```

Start the technician tagging immediately: it is the long pole for #3 and costs nothing
for #1 and #2. Run the server-scale A8 `predict` + `features` + `populate` next, because #2
cannot be submitted on 600 objects. #1 is closest to ready.

## One voice across the three

- The same methods paragraph, the same backbone figure (figures/dcae_architecture.png) in
  all three, with the instrument-specific box swapped.
- The same validation discipline, stated each time: populations are found without labels;
  measured stains or wells are used only afterwards; stability across acquisition halves and
  bootstraps is reported; galleries are chosen by rule.
- The same caveats each time, not hidden: one donor; fixed, stained cells only; wells are
  population labels; morphology is not yet WHO-graded.
