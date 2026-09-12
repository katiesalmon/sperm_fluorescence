# Results

Measured findings, newest first. Everything here is reproducible from the scripts named
alongside it. The plan these test is in [approach.md](approach.md); what the data turned
out to be is in [task_brief.md](task_brief.md).

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
