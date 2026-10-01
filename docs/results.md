# Results

Measured findings, newest first. Everything here is reproducible from the scripts named
alongside it. The plan these test is in [approach.md](approach.md); what the data turned
out to be is in [task_brief.md](task_brief.md).

---

## First look at the A8 images — the stain pictures, and what they say

*2026-10-01 · `analysis/a8_io.py` on `data/a8_bundle_seed0.zip` (8 samples x 200 events, join
verified on the server; see [task_brief.md](task_brief.md))*

### The data, numerically

| | |
| --- | --- |
| Stack | `(6, h, 104)` float32; h in 57–77 (mode 73) — flow axis varies with the event's transit |
| LightLoss | background 0.129, cells **darker** (extinction); range ~0–0.17 |
| FSC, SSC, fluorescence | **already background-subtracted** by the instrument: background ≈ 0, ~46% of pixels negative (noise), cell signal positive |
| Scale | tiny float units: AF488 p99 ≈ 0.10, PE 0.04, PerCP 0.03; SSC up to 5.5 |

Two consequences for any loss: no log transform (half the pixels are negative), and no
further background subtraction.

**`tifffile.imread` returns page 1 only.** BD writes a `{"shape": [h, w]}` description on
every page, which tifffile reads as six one-page series. Iterate `TiffFile(...).pages`.
`a8_io.read_stack` does; nothing else in the repo should touch these files directly.

### Replicate 1 is unstained on the A8 too

Median background-subtracted fluorescence page sums: `1S` and `1P` read 0–2 on every
channel against 28–38 (AF488) for replicates 2 and 3. Same experimental design as the
CytPix run. **Usable paired events: the six stained samples, 120,000.** Replicate 1's
60,000 (plus 1SP when extracted) are the negative control: predicted fluorescence on a
replicate-1 cell should be ~0.

### The fluorescence pages are raw filter channels — and the spillover is estimable

Not from an unconstrained fit: the unmixed scalars are collinear by cell type (sperm carry
AF488 *and* PerCP, PBMCs carry PE), so regressing page sums on all three returns R² of
0.8–0.9 and coefficients that are physically impossible (PerCP-eF710 does not put 90% of
its light through a 534 nm filter). Physics supplies the constraint: emission spills
**red-ward only**, so the mixing is lower-triangular with three coefficients, each
estimable from a well where the upstream fluorophore dominates:

| | coefficient | r | from |
| --- | ---: | ---: | --- |
| AF488 → PE | **0.202** | 0.965 | S wells (sperm carry no CD45) |
| AF488 → PerCP | 0.023 | 0.205 | S wells |
| PE → PerCP | 0.166 | 0.618 | P wells, AF488's share removed first |

Condition number 1.30 — inverting it per pixel is benign. `a8_io.spillover` and
`a8_io.unmix` implement this. Estimated under a physical constraint, not from single-stain
controls, which the experiment does not have; say so when it is used.

### What the pictures show

Contact sheet: `analysis/out/a8_pages.png`.

- **ACRV1 is a compact spot on the sperm head.** The acrosomal cap, imaged. This is the
  spatial ground truth the CytPix could never provide, on the marker that was least
  predictable from scalars.
- **LDHC/AKAP4 lights the whole flagellum**, plus the head. Principal piece, as the
  biology says.
- **The spillover is visible and the triangular unmix removes it.** On sperm the raw PE
  page shows the same tail as AF488; after unmixing the tail is gone and the page is
  noise. On PBMCs the unmixed PE page stays bright — genuine CD45.
- **The instrument's own unmixed CD45 scalar goes strongly negative on sperm** (−2e5 to
  −6e5), the same over-subtraction signature the Attune matrix showed. For scalar targets
  that means `asinh`; for images, the image-level triangular unmix looks cleaner than the
  instrument's spectral one.
- **A horizontal streak runs through the cell's row** in several fluorescence pages — a
  CellView scan-line artefact. Worth masking or modelling; it will otherwise be learned.

### The design decision this forces

**Use all three label-free pages as input, not LightLoss alone.** LightLoss shows a sperm
head crisply but the tail only faintly, and barely resolves a PBMC at all — a sperm head
is condensed chromatin and extinguishes strongly; a lymphocyte does not. FSC and SSC show
the tail clearly and the PBMC brightly, and they are every bit as label-free. The
image-to-image model's input is `(LightLoss, FSC, SSC)` → `(AF488, PE, PerCP)` unmixed.
Whether LightLoss *alone* suffices is then an ablation, and it is the ablation that bears
on transfer to the CytPix, which has nothing but extinction.

---

## What is actually inside an .acs, and what it changes

*2026-09-13 · `analysis/explore_masks.py`, `scripts/inspect_acs.py`, on a local copy of `3SP.acs`*

### The images are 16-bit, and the `Images\` export is a downconversion

| | `Images\*.zip` | inside the `.acs` |
| --- | --- | --- |
| Format | 248 x 248, **uint8**, RGBA container, LZW | 248 x 248, **uint16**, single channel, uncompressed |
| Per file | 74-105 KB (varies with noise) | 123,250 B (constant) |

Decoded pixel values run ~145-1011 with 686 distinct levels pooled, so the sensor is
~10-bit and the `.acs` preserves it. The `Images\` export throws away about two bits.
**Source images from the `.acs`.** A standard-library decoder for both widths was written and verified byte-for-byte
against PIL on the real archives; it has since been removed as unused (git history).

### The instrument ships its segmentation

Each archive holds a `*.masks.zip` — one JSON per event, keyed by the same event id:

```
[{"version":"1.0"}, {"masks":[{"pixelIndexes":{"indexes":[...]}}]}, {"masks":[...]}]
```

Flat pixel indices into the 248 x 248 frame, in two layers, and both are now identified
against the FCS columns on 594 events:

| Layer | Is | Evidence |
| --- | --- | --- |
| 1 | **Objects** | mask count == `ObjectCount` on 594/594; pixel count == `NumPixels` |
| 2 | **Particles** | mask count == `ParticleCount` on 594/594; lies entirely inside layer 1 |

Layer 2 is the instrument's particle detection — a small region (~7% of the object),
slightly darker than the rest but *not* the darkest part, so not a nucleus proxy. Both
layers are brightfield segmentation products. **There are no per-stain masks in the
archive**: all 30,000 mask files have exactly these two layers, every entry carries only
`pixelIndexes`, and neither XML in the archive names a channel, stain or layer. Rendered,
the object mask tracks the flagellum correctly on tailed cells, so the segmentation step
`sperm_pbmc` built by hand is already done here.

### Compensation: the export is raw, and one coefficient is wrong

The `$SPILLOVER` matrix is 15 x 15 and well-conditioned (condition number 2.8). Whether
the exported `-A` values were pre- or post-compensation could not be read off the matrix,
so it was tested against biology: **PBMCs have no acrosome, so ACRV1 on a CD45-high round
cell must be ~0.**

| | ACRV1 median on CD45-high events | corr(ACRV1, CD45) | events with a negative channel |
| --- | ---: | ---: | ---: |
| as exported | 10,608 | 0.940 | 4.5% |
| **inv(S) applied** | **91** | 0.578 | 88.5% |
| S applied | 22,589 | 0.988 | 0.1% |

**The export is raw and `inv(S)` is the correction.** That settles it: the ACRV1 signal on
PBMCs was spillover, and compensation removes it almost exactly.

But one coefficient does not survive scrutiny. `BL1-A -> YL1-A = 1.0001` says AF488 lands
in the PE detector at 100% of its own-detector intensity. Compensating with it drives the
median CD45 in `3SP` to **-15,408**: most events there are sperm, which are LDHC-high and
genuinely CD45-negative, so a ~100% subtraction of BL1 from YL1 overshoots. Compensated
data legitimately spreads below zero, but not by that much.

**Two consequences for modeling:**

1. **Compensate, then use an `asinh` (biexponential) transform, not `log1p`.** 88.5% of
   compensated events carry a negative in some channel, which `log1p(clip(x, 0))` would
   flatten to zero. That is the standard flow-cytometry transform for exactly this reason.
2. **The morphology baseline below was run on uncompensated targets.** Its ACRV1 number
   (0.209) is partly measuring PE spillover, and has to be rerun compensated before it
   means anything. DAPI and CD45 are far less affected — VL1 has almost no spillover in or
   out — but the rerun is the number to quote.

**Open with the lab:** is `BL1-A -> YL1-A = 1.0001` real, or a compensation setup error?
It is the single largest coefficient in the matrix and it governs the CD45 channel.

---

## Morphology baseline — how much of each marker is predictable without a network

*2026-09-12 · `python analysis/feature_baseline.py data/targets --train 2 --test 3`*

This is step 3 of [approach.md](approach.md), and it ran without a single image: the
CytPix computes shape, intensity and texture per event, so "does what the cell looks like
predict what it stains for" is answerable from the target tables alone.

Spearman correlation of predicted vs measured intensity. **within** is a random split
inside replicate 2 — what a naive evaluation would report. **across** is trained on
replicate 2 and tested on replicate 3 — the honest number. **null** is permuted-target
models, three runs, reported as a range.

### All wells pooled (`S`, `P`, `SP`) — 102,874 train / 90,000 test

| marker | features | within | across | null range | |
| --- | --- | ---: | ---: | --- | --- |
| DAPI | shape only | 0.791 | **0.776** | −0.095 .. 0.101 | solid |
| DAPI | shape + intensity | 0.835 | **0.798** | −0.094 .. −0.024 | solid |
| CD45 | shape only | 0.803 | **0.767** | −0.197 .. 0.085 | solid |
| CD45 | shape + intensity | 0.831 | **0.797** | −0.113 .. −0.029 | solid |
| LDHC | shape only | 0.660 | **0.384** | −0.030 .. 0.038 | solid, but see below |
| LDHC | shape + intensity | 0.716 | **0.425** | −0.023 .. −0.002 | solid, but see below |
| ACRV1 | shape only | 0.359 | **0.209** | −0.070 .. 0.118 | weak |
| ACRV1 | shape + intensity | 0.408 | **0.284** | −0.074 .. 0.161 | weak |

### The `SP` mixture only — the control that rules out acquisition leakage

Every event here comes from a single acquisition per replicate, so nothing can be
explained by "which well was this". 42,874 train / 30,000 test.

| marker | across (shape only) | across (+ intensity) |
| --- | ---: | ---: |
| LDHC | **0.668** | 0.727 |
| DAPI | **0.616** | 0.628 |
| CD45 | **0.544** | 0.565 |
| ACRV1 | **0.306** | 0.348 |

Morphology still predicts every channel inside one acquisition. The pooled numbers are
not an artefact of the model learning which well an event came from.

### What this says

**1. The ordering is the one the literature predicts.** DeepIFC's rule — a marker is
predictable exactly to the extent that what it binds has a visible correlate — gives
DAPI ≈ CD45 ≫ LDHC ≫ ACRV1, and that is the observed order. DAPI and CD45 both come down
to "is this a round cell or a sperm", which morphology answers easily. ACRV1 is acrosomal
*integrity*, a biochemical state, and it is the one that barely moves. That was the
prediction in [approach.md](approach.md) §2 and it held.

**2. The replicate split costs a lot, and that is the point.** LDHC reports **0.660** on a
random split and **0.384** across replicates — a 42% relative drop. Anyone evaluating this
on a naive split would have believed the higher number. The drop is concentrated in the
`P` wells, which is where replicate 2's LDHC channel misbehaves (see
[task_brief.md](task_brief.md)): restricted to the mixture, LDHC holds up at 0.704 → 0.668.

**3. Brightfield intensity is not carrying the result.** Adding `TotalIntensity`,
`AverageIntensity` and `StandardDeviationIntensity` moves every marker by only +0.02 to
+0.06. Given `sperm_pbmc` Finding 1 — that background noise alone separated acquisitions
at AUC 0.992 — the worry was that intensity features would dominate. They do not. The
signal is in shape.

**4. Run the null more than once.** A single permutation put the LDHC null at 0.207, which
looked like serious leakage. Three permutations put the range at −0.03 .. 0.04. A
permuted-label model does not predict a constant — it emits an arbitrary function of the
features, and when the features are this informative, an arbitrary function of them can
correlate with the truth by luck. One permutation is not a control.

### What it means for the next step

These are **floors**. A few dozen instrument-computed features, no learned representation,
no pixels. A CNN on the images has to beat 0.78 on DAPI and 0.77 on CD45 to be worth
having — and the interesting target is ACRV1, where there is the most headroom and the
least reason to expect success.

**Caveat carried forward:** these targets are **uncompensated**. ACRV1 correlates with
CD45 at ρ 0.97–0.98 in the PBMC wells, so part of what "ACRV1 0.209" measures is spillover
from PE. The number could move in either direction after compensation, and the compensated
rerun is the one to quote.
