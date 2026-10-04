# Task brief

> **Scope, 2026-10-01.** The dataset is the BD FACSDiscover A8 run — eight samples, six of
> them stained. The Attune CytPix sections below are retained as history: they established
> the experimental design and the batch-effect lessons, but the samples were run for the
> A8 and the CytPix is out of scope. `C1-1SP` has no sample; its FCS is an empty tube.


## Goal

Given the raw (brightfield) CytPix event image, predict the fluorescence signal in each
antigen marker channel:

| Detector | Label as recorded ($PnS) | Antigen | Where it sits | Brightfield correlate |
| --- | --- | --- | --- | --- |
| `VL1-A` | `DAPI-A` | DNA | nucleus — the head | the head: largest, highest-contrast structure |
| `BL2-A` | `ACRV-1-PerCP-ef710-A` | ACRV1 / SP-10 | acrosome — cap over the anterior head | a subtle intensity step within the head |
| `BL1-A` | `LDHC_AKAP4-AF488-A` | **LDHC *and* AKAP4, pooled on one fluor** | principal piece of the flagellum | the tail |
| `YL1-A` | `CD45-PE-A` | **CD45** — pan-leukocyte | leukocyte surface; absent from sperm | round cell vs sperm |

Read off the FCS in all nine archives (2026-09-12). **Every one of the nine carries the
identical panel at identical detector voltages** — the same four labels, the same 59
parameters, BL1 300 / BL2 425 / YL1 375 / VL1 250 throughout. It differs from the brief
this project started with in three ways:

- **Tomm20 is not in the experiment.** Not in one replicate — in none of the nine.
  `RL1-A`, where a far-red mitochondrial stain would sit, carries no operator label in any
  archive. Either it was not run on 2026-07-09 or it belongs to a different session.
  **This needs an answer from whoever planned the panel; it is not recoverable from the
  data.**
- **`BL1-A` is a two-antigen cocktail**, LDHC and AKAP4 together on AF488. Both are
  principal-piece proteins, so as a *compartment* readout it is coherent — but it cannot
  be resolved into "LDHC signal" and "AKAP4 signal", and the write-up has to call it a
  flagellar-marker channel rather than an LDHC channel.
- **CD45 is in the panel and was not in the brief.** It is the pan-leukocyte marker, i.e.
  a direct fluorescent label for exactly the round-cell-vs-sperm call that `sperm_pbmc`
  spent its curation effort on. See below.

One model per marker, as four separate heads or four separate models — that decision
belongs in [approach.md](approach.md), not here.

The four markers are worth noticing as a set: they tile the sperm from front to back —
acrosome, nucleus, midpiece, principal piece. Each one localises to a compartment with a
distinct brightfield correlate. That is the reason to expect this to work at all, and it
is also the yardstick: a marker whose predicted signal does not land on its own
compartment is not being predicted, however good its correlation looks.

## Why "self-supervised"

The label comes from the paired fluorescence acquisition rather than from a person. No
one draws a box or types a class; the target is measured at the same instant as the input.

Two distinct things get called self-supervised, and the write-up will be clearer if we
keep them apart from the start:

- **Free-label cross-modal supervision** — the actual task here. Architecturally it is
  ordinary supervised regression; what is free is the *labels*, not the supervision.
  Published work in this area calls it *virtual staining* or *in silico labeling*.
- **Self-supervised representation learning proper** — masked autoencoding, DINO,
  contrastive pretraining — where the objective is constructed from unlabeled images
  alone. This has a real role here as a *pretraining stage*, and we have ~180,000
  unlabeled brightfield events across replicate 1 to do it with.

Both are on the table. They are not the same technique and should not share a name in
the methods section.

## Design of the experiment

**Training source.** Paired events from the marker replicates (`2*`, `3*`), where the
brightfield frame and the marker signal describe the same event at the same moment.

**The held-out axis that matters is the replicate, not the event.** Random event-level
splits will overstate performance, because events from one acquisition share a gain
setting, a focus, and a noise floor. Replicate 2 → replicate 3 (and back) is the split
that answers "does this generalise", and it is available for free.

**The `S` / `P` / `SP` wells are a second, independent check.** PBMCs have no acrosome,
no midpiece and no flagellum. A model that predicts ACRV1, LDHC or Tomm20 signal onto a
PBMC is reading something other than the biology — and the `2P` / `3P` wells supply
thousands of those negatives without anyone labeling anything.

## What we already know, from replicate 1

Established by the survey in the `sperm_pbmc` repo, and worth carrying over because it
sets expectations for what the marker replicates will look like:

| | |
| --- | --- |
| Layout | Flat — no subfolders. One TIF per event, named `<event-id>.tif`. |
| Events per zip | 30,000 in each of `1S`, `1P`, `1SP` |
| Image | 248 × 248, single page, uint8 |
| Channels | Stored RGBA, but R == G == B and alpha constant — 8-bit grayscale in a 4-channel container |
| Compression | LZW (needs `imagecodecs` alongside `tifffile`) |
| Metadata sidecar | None in replicate 1 |

**Two findings from replicate 1 that constrain this project directly:**

1. **Acquisitions are separable by background alone.** A classifier given only 40 × 40
   corner patches — containing no cell whatsoever — told `1S` from `1P` at AUC 0.992,
   because the two wells have different noise floors (background σ 5.5 vs 10.7). Any
   model here that is allowed to see raw, un-normalised background can learn the
   acquisition instead of the biology. For a regression target that varies by
   acquisition gain, that is not a hypothetical.
2. **The single-population wells are impure.** Semen normally contains round cells, so
   a sperm well is a sperm-and-round-cell well. Well identity is a prior, never a label.

## Data inventory

Nine zips, ~24 GB total, exported 2026-07-09. The six marker zips are what this repo is
about; replicate 1 is brightfield only and stays in `sperm_pbmc`.

| File | Size | Replicate | Contents | In scope |
| --- | --- | --- | --- | --- |
| `..._1P.zip` | 2.75 GB | 1 | PBMCs alone | no — brightfield only |
| `..._1S.zip` | 2.35 GB | 1 | sperm alone | no — brightfield only |
| `..._1SP.zip` | 2.60 GB | 1 | sperm + PBMC mixture | no — brightfield only |
| `..._2P.zip` | 2.77 GB | 2 | PBMCs alone + fluorescence | **yes** |
| `..._2S.zip` | 2.34 GB | 2 | sperm alone + fluorescence | **yes** |
| `..._2SP.zip` | 3.55 GB | 2 | mixture + fluorescence | **yes** |
| `..._3P.zip` | 2.77 GB | 3 | PBMCs alone + fluorescence | **yes** |
| `..._3S.zip` | 2.34 GB | 3 | sperm alone + fluorescence | **yes** |
| `..._3SP.zip` | 2.42 GB | 3 | mixture + fluorescence | **yes** |

**The sizes are the first open question.** `2S` is 2.34 GB against `1S`'s 2.35 GB —
essentially identical. If replicate 2 carried four extra image channels per event, it
should be several times larger. Either the marker channels are not stored as images in
these zips, or they compress far better than brightfield, or the event count differs.
The survey settles it; do not assume.

## What the survey found (2026-09-12)

`inspect_zip.py --structure --peek-metadata` (since removed; see git history) over all six marker zips, from
`Z:\Blair_Main\2026\260709_Blair_Sperm_Cytpix\Images\`:

| | `2S` | `2P` | `2SP` | `3S` | `3P` | `3SP` |
| --- | --- | --- | --- | --- | --- | --- |
| Events | 30,000 | 30,000 | **42,874** | 30,000 | 30,000 | 30,000 |
| Files per event | 1 | 1 | 1 | 1 | 1 | 1 |
| Mean file size | 78.0 KB | 92.2 KB | 82.7 KB | 78.0 KB | 92.2 KB | 80.7 KB |
| Sidecars | 0 | 0 | 0 | 0 | 0 | 0 |

Every zip: **flat, one `<event-id>.tif` per event, 248 × 248, 1 page, spp=4, uint8, LZW,
no `ImageDescription`, no sidecar of any kind.**

**That is structurally identical to replicate 1.** Both candidate branches in
[approach.md](approach.md) are ruled out as stated: the markers are not separate files,
not extra pages, and there is no FCS or CSV in these zips to join against.

Three things follow from the numbers themselves:

1. **The RGBA container is the only place left in these files for marker signal.** If
   replicate 1's `R == G == B, alpha constant` no longer holds in `2*` / `3*`, the
   fluorescence is in those channels. `check_channels.py` (since removed) decoded pixels and settled it
   (since removed; see git history).
2. **The file sizes argue it is not.** A 248 × 248 LZW TIF holding grayscale replicated
   across RGBA with constant alpha measures ~83 KB on synthetic cell-like content; three
   independent channels plus alpha measures ~175 KB, four independent ~197 KB. The real
   files are 78–92 KB. They carry roughly *one* channel's worth of entropy. This is
   indirect but quantitative, and it points at these zips being brightfield only.
3. **`S` files (78 KB) are consistently smaller than `P` files (92 KB), in both
   replicates.** That is `sperm_pbmc` Finding 1 showing up in file sizes: the PBMC wells
   have a higher background noise floor (σ 10.7 vs 5.5), noise does not compress, so the
   files are bigger. File size is tracking acquisition noise, not cell content — a
   reminder of how strongly the acquisition confound is present in this data.

### Settled: these zips are brightfield only

`check_channels.py` (since removed) over `2S`, `2P`, `3S`, `3P`, 24 events each:

```
R  mean 149.16   G  mean 149.16   B  mean 149.16   A  mean 255.00  sd 0.00
worst pixel difference: R-G=0  R-B=0  G-B=0
=> GRAYSCALE IN AN RGBA CONTAINER -- R == G == B on every pixel, alpha constant.
```

Identical verdict in all four. **R == G == B on every pixel of every event decoded, and
alpha is a constant 255.** The marker replicates carry no fluorescence in their image
files; they are 8-bit brightfield in a four-channel container, exactly like replicate 1.
The file-size arithmetic above was right.

So the marker signal, if it was recorded at all, is in a separate export.

Incidentally the per-image standard deviations confirm the acquisition confound from a
third direction: `2P` sd 15.20 and `3P` sd 14.76 against `2S` sd 8.73 and `3S` sd 9.00.
The PBMC wells are noisier in both replicates, which is what drove the file sizes, and is
`sperm_pbmc` Finding 1 again. Per-image mean also swings widely within a single zip
(`3P` ranges 113.9–168.9), so per-image normalisation is not optional.

### The event ids say where to look

The images are named with bare integers — `10.tif`, `100002.tif` — and they are **sparse**:
`2S` holds 30,000 images but ids run past 100,002, and there are gaps throughout.

The archive's `ImageCaptureSettings` XML says why: `ImageGate GateName="DAPI+"` and
`ImageTotals Total="30000"`. The camera imaged only events inside a **DAPI+ gate**, and
stopped at 30,000. Each image is named by its event index in the full record. Which means:

- the images are a **subset** of a larger per-event record, roughly 30% of it;
- **the image filename is the join key** to that record;
- an FCS for this run should report `$TOT` of at least ~100,000, not 30,000.

That last point is a test, not just a description: it is how we will know we have found
the *right* FCS rather than a different export. That number is reported by `inspect_acs.py`.

This is Branch B of [approach.md](approach.md), and it appears to be the data model the
instrument was always going to produce.

### The cytometry data: nine .acs archives in the run folder

| Archive | Size | Matching image zip |
| --- | --- | --- |
| `..._1P.acs` | 3.82 GB | 2.75 GB |
| `..._1S.acs` | 3.80 GB | 2.35 GB |
| `..._1SP.acs` | 3.88 GB | 2.60 GB |
| `..._2P.acs` | 3.83 GB | 2.77 GB |
| `..._2S.acs` | 3.80 GB | 2.34 GB |
| `..._2SP.acs` | **5.46 GB** | 3.55 GB |
| `..._3P.acs` | 3.81 GB | 2.77 GB |
| `..._3S.acs` | 3.80 GB | 2.34 GB |
| `..._3SP.acs` | 3.80 GB | 2.42 GB |

An **ACS** (Archival Cytometry Standard) file is a zip container holding FCS data files
alongside a table of contents, and often the images they were acquired with. Two things
line up:

- Every archive is **larger than its image zip**, by roughly the same margin — consistent
  with "the images, plus the cytometry data, plus overhead".
- `2SP` is the outlier in both: 5.46 GB against 3.8 GB, matching its 42,874 events
  against everyone else's 30,000. The two exports agree about which acquisition is
  bigger, which is a good sign they describe the same runs.

**Do not copy these to a laptop.** Nine archives at ~34 GB, where the part we need — the
FCS — is a few MB. `scripts/inspect_acs.py` reads the TEXT segment in place (decompressing
only the head of the member, so listing is fast even on a multi-gigabyte archive) and
extracts the FCS members alone.

Note on transfer: dragging these through a Remote Desktop redirected folder produced
**sparse placeholder files** — correct logical size, zero bytes on disk, every byte zero.
`inspect_acs.py` detects that case and says so rather than reporting a corrupt archive.

### What the FCS holds (`3S.acs`, read 2026-09-12)

```
FCS3.1   $TOT=101510 events   $PAR=59 parameters
$CYT     0A48664 Attune CytPix Flow Cytometer (Lasers: BRV6Y)
$DATE    09-Jul-2026   $BTIM 10:56:27
images in this archive: 30000 of 101510 events (29.6%)
$SPILLOVER: 15x15 over BL1-A BL2-A YL1-A YL2-A YL3-A RL1-A RL2-A RL3-A
            VL1-A VL2-A VL3-A VL4-A VL5-A VL6-A ImageFlag
```

**The subset prediction was right.** 30,000 images against 101,510 recorded events —
29.6%. `ImageFlag` (P23) is the per-event boolean saying which ones were imaged, and
`Event` (P1) is the id. `make_targets.py --verify-images` tests the join directly by
comparing image filenames against event ids.

Four things in here change the plan, in descending order of importance.

**1. The instrument already computed the morphology features.** Parameters 24–59 are not
detectors; they are per-event measurements the CytPix derived from the image:

| Group | Parameters |
| --- | --- |
| Intensity | Max, Min, Total, Average, StandardDeviation, CV, Skewness, Kurtosis, Entropy, and normalised variants |
| Shape | NumPixels, AreaSquareMicrons, PerimeterMicrons, Major/MinorDiameterMicrons, MinorMajorRatioPercent, EccentricityPercent, CircularityPercent, PseudoDiameterMicrons, GyrationRadiusWeighted |
| Texture | Maximum / Contrast / Entropy / AngularSecondMoment CoOccurrence (Haralick) |
| Quality | ParticleCount, ObjectCount, ClumpIndexMax, IsOnBorder, IsProcessable, IsProcessed, ConfidenceScore |

That is the feature baseline from step 3 of [approach.md](approach.md), for free, with no
image processing at all — and it is computed identically across every acquisition, so it
cannot drift the way a hand-rolled segmentation can. It does **not** remove the need for
the background-only control: these features are computed from the same images whose noise
floor differs by well, so they can encode acquisition just as readily.

**2. There are segmentation masks.** `F8C04D37-....masks.zip`, 71 MB, one archive member
alongside the images. If those are per-event object masks, the segmentation step is done,
and the masked-input requirement from `sperm_pbmc` Finding 1 becomes trivial to satisfy.
Worth opening before writing any segmentation code.

**3. The values are uncompensated, and the matrix is 15 × 15.** Four stains across four
lasers will spill into each other — PerCP-eF710 and PE overlap substantially. A per-marker
model trained on raw `-A` values would learn the spillover and we would read it as
biology. Compensation has to be applied before the targets are used, using the matrix in
`$SPILLOVER`. (Note the matrix includes `ImageFlag` as a row/column, which is an artefact
of how it was written, not a real detector.)

**4. Detector voltages are recorded per parameter.** `3S`: BL1 300, BL2 425, YL1 375,
VL1 250. These are the acquisition gain, stated numerically. **Compare them across
replicates before treating `2*` → `3*` as a clean held-out split** — if the voltages
differ, the targets are not on a common scale and the split measures gain as well as
generalisation.

### All nine acquisitions

| Run order ($BTIM) | Acquisition | Events ($TOT) | Images | % imaged |
| --- | --- | ---: | ---: | ---: |
| 10:52:55 | `2S` | 90,864 | 30,000 | 33.0% |
| 10:56:27 | `3S` | 101,510 | 30,000 | 29.6% |
| 11:01:00 | `1P` | 203,357 | 30,000 | 14.8% |
| 11:07:49 | `2P` | 225,413 | 30,000 | 13.3% |
| 11:18:03 | `3P` | 181,608 | 30,000 | 16.5% |
| 11:28:25 | `1SP` | 408,650 | 30,000 | 7.3% |
| 11:36:53 | `2SP` | 223,294 | **42,874** | 19.2% |
| 11:48:45 | `3SP` | 106,667 | 30,000 | 28.1% |
| 11:52:12 | `1S` | 101,800 | 30,000 | 29.5% |
| | **total** | **1,643,163** | **282,874** | 17.2% |

**282,874 paired image-and-measurement events.** There is no label scarcity in this
project at all, which settles §6 of [approach.md](approach.md): self-supervised
pretraining is not needed for label efficiency, and would have to justify itself some
other way.

Three things this table says that the single-file read did not:

**1. Replicate 1 is not brightfield-only** — but it is not a stained replicate either.
See the correction below: the panel *metadata* is identical across all nine because it is
the operator's instrument configuration, which carries over to every tube. It says nothing
about what was actually in the tube. The measured values say `1*` got DAPI and nothing
else.

**2. The voltages are identical across all nine.** That removes the gain confound at the
detector level, which §4 of [approach.md](approach.md) flagged as a reason the
train-on-2 / test-on-3 split might measure something other than generalisation. It does
**not** make the acquisitions equivalent — the background noise floors still differ by
well (measured: `P` sd ~15 against `S` sd ~9), the samples were stained and washed at
different times, and the event rates differ four-fold across the session. Held-out
replicate remains the right split; it is just no longer confounded with gain.

**3. Run order is not replicate order.** The session goes `2S`, `3S`, `1P`, `2P`, `3P`,
`1SP`, `2SP`, `3SP`, `1S` — an hour end to end, with `1S` last and `2S` first. Replicate
number is therefore not a proxy for time, which is good: a replicate split is not secretly
a time split. But `1S` and `2S` are an hour apart on the same instrument, so within-day
drift is the batch effect to watch, not gain.

**The imaged subset is selected, not sampled.** The capture settings gate on `DAPI+` and
cap at 30,000, so the imaged fraction (7.3% in `1SP` to 33.0% in `2S`) is the gate's pass
rate times the cap, not a camera rate. Two consequences: DAPI-negative events — debris,
anucleate fragments — never appear in the training set, so the DAPI target is truncated
from below; and `make_targets.py --compare-imaged` will show imaged events differing from
unimaged ones *by design*. That is expected, not a bias to correct. `2SP`'s 42,874 means
its cap was set differently; its own `capture_settings.xml` will say.

### Correction, from the values themselves: replicate 1 is the unstained control

Reproduce with `python analysis/explore_targets.py data/targets`.

Median intensity, by acquisition, uncompensated:

| well | n | DAPI | ACRV1 | LDHC/AKAP4 | CD45 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1S` | 30,000 | 2,192 | **371** | **570** | **86** |
| `2S` | 30,000 | 2,200 | 8,341 | 18,218 | 131 |
| `3S` | 30,000 | 2,208 | 8,463 | 19,756 | 140 |
| `1P` | 30,000 | 13,023 | **705** | **1,760** | **1,594** |
| `2P` | 30,000 | 12,984 | 11,112 | 33,702 | 52,991 |
| `3P` | 30,000 | 12,763 | 7,522 | 9,380 | 38,614 |
| `1SP` | 30,000 | 6,932 | **644** | **1,161** | **754** |
| `2SP` | 42,874 | 2,611 | 11,732 | 21,604 | 256 |
| `3SP` | 30,000 | 1,783 | 10,942 | 14,840 | 151 |

As a fold-change over the replicate-1 acquisition of the same well type:

| well | DAPI | ACRV1 | LDHC | CD45 |
| --- | ---: | ---: | ---: | ---: |
| `2S` | **1.0x** | 22.5x | 32.0x | 1.5x |
| `3S` | **1.0x** | 22.8x | 34.7x | 1.6x |
| `2P` | **1.0x** | 15.8x | 19.1x | 33.2x |
| `3P` | **1.0x** | 10.7x | 5.3x | 24.2x |

**DAPI is identical across replicates; everything else is 5–35x up in replicates 2 and 3.**
Replicate 1 was stained with DAPI only. So:

- **There are two stained replicates, not three.** Usable paired training events:
  **192,874** (`2*` and `3*`). Replicate 1's 90,000 are an unstained control — which is
  worth having, not a loss: it is the autofluorescence floor, per well type, measured on
  90,000 events, and it is what the fold-change table above is computed against.
- **Held-out-replicate has exactly two groups.** Train on 2, test on 3, or the reverse.
  There is no third fold.

**DAPI reports real biology, which is the best evidence the values mean what we think.**
Sperm nuclei are haploid and protamine-condensed; PBMC nuclei are diploid. Measured ratio
`1P`/`1S` = 5.9x, and DNA content alone separates the two wells at **AUC 0.993** — in the
*unstained* replicate, so that is the dye, not the antibody panel.

### The spillover is severe enough to invalidate a channel

Spearman correlation with CD45, within an acquisition:

| well | ACRV1~CD45 | LDHC~CD45 |
| --- | ---: | ---: |
| `1P` (unstained) | 0.727 | 0.022 |
| `2P` | **0.974** | 0.467 |
| `3P` | **0.980** | 0.313 |

In a PBMC well, ACRV1 is almost a monotone function of CD45. PBMCs have no acrosome, so
that channel is not reporting ACRV1 there — it is reporting PE spilling into the
PerCP-eF710 detector, plus whatever non-specific antibody binding the Fc receptors on
monocytes contribute. `1P`'s 0.727 with no antibody at all sets the autofluorescence
baseline: brighter cells are brighter everywhere.

**Consequence: compensate before using any target.** `make_targets.py` now writes the
instrument's `$SPILLOVER` matrix alongside each table as `<name>.spillover.csv`, because
an uncompensated target table cannot be corrected after the fact. And the PBMC negative
control proposed in [approach.md](approach.md) §5 only means anything post-compensation —
on raw values, predicted ACRV1 on a PBMC is *supposed* to be non-zero.

### The two stained replicates disagree, and one channel disagrees in direction

AUC for separating the PBMC well from the sperm well on one channel alone. For a sperm
marker, below 0.5 is the biologically correct direction:

| replicate | DAPI | ACRV1 | LDHC | CD45 |
| --- | ---: | ---: | ---: | ---: |
| 1 (unstained) | 0.993 | 0.854 | 0.889 | 0.990 |
| 2 | 0.994 | 0.689 | **0.732** | 0.997 |
| 3 | 0.994 | 0.428 | **0.179** | 0.997 |

CD45 and DAPI are rock solid in both. But **LDHC/AKAP4 reverses**: in replicate 3 sperm
carry more than PBMCs (correct), in replicate 2 PBMCs carry more (not). `2P`'s LDHC median
is 3.6x `3P`'s. Something went wrong with that channel in replicate 2 — over-staining,
non-specific binding, or a wash step — and it is a reason to treat replicate 3 as the
cleaner of the two until compensation says otherwise.

This is also the clearest possible argument for the held-out-replicate split: two
acquisitions of the same panel, same voltages, an hour apart, disagree by 3.6x on one
channel and reverse its sign. A random event-level split would never surface that.

### This also unblocks the sibling project

`YL1-A` carries CD45-PE — but **only in replicates 2 and 3**, which is the correction
above. `sperm_pbmc` works on replicate 1, and replicate 1 has no CD45 antibody.

`sperm_pbmc` exists to tell a curled sperm from a PBMC in brightfield. It works on
replicate 1, and its task brief states that replicate 1 is brightfield-only and that the
fluorescent replicates are "the intended independent confirmation", deferred as out of
scope. **Both halves of that are wrong.** Replicate 1 was stained with the same panel as
2 and 3; the fluorescence was simply exported to the `.acs` rather than into the image
files, and nobody had opened the `.acs`.

What replicate 1 *does* have is DAPI, on exactly the 30,000 events per well that project
is already working on. **DNA content separates its PBMC well from its sperm well at AUC
0.993** — sperm nuclei are haploid and condensed, PBMC nuclei are diploid, and the medians
differ 5.9-fold. That is a free per-event quantity, joined by filename, with no curation
and no reviewer agreement ceiling.

It is not the same thing as a CD45 label: DAPI measures DNA content, so it separates
*sperm from round cells* but will not by itself distinguish a leukocyte from an immature
germ cell, and the `indeterminate` bin that project created for round events with no
visible tail is exactly where the two overlap. Still, an AUC-0.993 free signal on the
events already being labelled by hand is worth knowing about, and replicates 2 and 3 do
carry real CD45 for a subset of the same question.

That is worth telling that project. It does not change the work here.

### Open questions, revised

1. ~~**Where is the fluorescence?**~~ **Found.** Nine `.acs` archives sit in the run
   folder `Z:\Blair_Main\2026\260709_Blair_Sperm_Cytpix\`, one per acquisition,
   named to match the image zips
   (`260709_Blair_Sperm_Cytpix_260709_Cytpix_3S.acs`). See below.
2. **Why does `2SP` have 42,874 events when every other zip has exactly 30,000?** 30,000
   is a round number and looks like a collection cap; 42,874 does not. Either that
   acquisition was configured differently or the export was assembled differently.
3. **What is the 5,656-byte file in `2SP`?** Every other file in every zip is 73–105 KB.
   A 5.7 KB LZW frame at this size is nearly blank. Worth decoding before it lands in a
   training set — and `--max-member-bytes` will not catch it, since that guard is for
   files that are too *large*.
4. **Do `2*` and `3*` reuse event ids?** Filenames are bare integers (`10.tif`,
   `100002.tif`), so ids certainly collide across zips. Any index built for training has
   to be keyed on (class, event-id), never the id alone.
5. **Are the replicates the same panel?** Still open, and it now matters more: if the
   fluorescence lives in a separate FCS export, the detector-to-antigen mapping comes
   from that file's `$PnN` / `$PnS` keywords, and it may differ between replicates.

## A second instrument: the `_A8` run folder (surveyed 2026-09-27)

`Z:\Blair_Main\2026\260709_Blair_Sperm_A8\260709\` — same date, same nine samples, a
different instrument. What `survey_tree.py` establishes:

| | |
| --- | --- |
| At root | 9 x `.fcs`, 9 x `.cvw`, 9 x `.zip`, named `A1-1S`, `A2-2S`, `A3-3S`, `B1-1P`, `B2-2P`, `B3-3P`, `C1-1SP`, `C2-2SP`, `C3-3SP` — a plate position plus the sample code |
| Image dirs | 8 (`C1-1SP` has no sample — eight samples, not nine), each split into `00000000 / 00010000 / 00020000 / 00030000` buckets of 10,000 event indices |
| Per sample | **exactly 20,000** images — a capture cap — with sparse indices to ~35-40k, so about half the events were imaged |
| Per image | `<sample>_<8-digit event index>.tiff`, **6 pages, 104 wide x 57-85 high (steps of 4), float32, uncompressed**, ~170 KB |
| Page names | `LightLoss (Imaging)`, `FSC`, `SSC (Imaging)`, `Imaging_BP/534/46/LP/505`, `Imaging_BP/598/60/LP/570`, and a sixth the survey's listing cap hid |
| Total | 160,000 imaged events, 6 channels each; 120,000 of them stained |

**The stain images are here.** Each event carries a label-free image (`LightLoss`,
plus `FSC` and `SSC`) *and* fluorescence images from bandpass-filtered detectors, all
co-registered because they come from the same scan. That is the spatial ground truth
the CytPix data does not have — and it is why the earlier "no stain masks" answer was
right about the `.acs` and wrong about the experiment.

**Verified from `A3-3S.fcs` (2026-10-01):** `$CYT FACSDiscover A8`, FCS 3.2, 37,085 events,
271 parameters. The panel is unmixed by the instrument and named by fluorophore:

| FCS parameter | Label | Marker | Imaged? |
| --- | --- | --- | --- |
| `Alexa Fluor 488-A` | `LDHC_AKAP4-A` | LDHC + AKAP4 | **yes** — `ImgB1 (535)`, TIFF page `Imaging_BP/534/46/LP/505` |
| `PE-A` | `CD45-A` | CD45 | **yes** — `ImgB2 (600)`, TIFF page `Imaging_BP/598/60/LP/570` |
| `PerCP-e710*-A` | `ACRV-1-A` | ACRV1 | **yes** — `ImgB3 (790)`, the sixth TIFF page |
| `DAPI-A` | — | DAPI | **no** — violet-excited; CellView images on the blue laser only |

So the six TIFF pages are three label-free channels (`LightLoss (Imaging)`, `FSC`,
`SSC (Imaging)`) and three fluorescence channels, co-registered. **Three of the four
markers have per-pixel ground truth. DAPI has a scalar only.** The values are already
spectrally unmixed (the 78 x 78 `SPILL` is the unmixing matrix over the spectral
detectors; the three `ImgB` imaging detectors sit outside it), so the compensation
problem that dominated the Attune data does not arise here.

The remaining 250-odd parameters, for the record:

- **P20–P97: CellView's per-channel image features** — Size, Max Intensity, Long/Short
  Axis Moment, Center of Mass X/Y, Total Intensity, Radial Moment, Eccentricity,
  Diffusivity, for each of the six imaging channels; plus **Correlation** between
  fluorescence channel pairs and **Delta CoM** (centre-of-mass offset) between every
  channel pair. The Delta CoM columns are a free compartment-localisation readout:
  `Delta CoM (LightLoss (Imaging)/ACRV-1 PerCP-e710*)` is literally "how far the
  ACRV1 signal sits from the cell's centroid".
- **P103–P264: 81 detectors, `-A` and `-H`** — `ImgB1-3`, `UV1-22`, `V1-20`, `B1-16`,
  `YG1-12`, `R1-8`. Raw spectral data.
- **P98–P102, P265–P271: flags and metadata** — `Saturated`, `Time`, `Sorted`, plate
  `Row`/`Column`, saturation channel masks, `PhaseOffset`, `SpectralEventWidth`,
  `MergedTriggerCount`, `WaveformPresent`.

**Join and page order, verified on the server (2026-10-01).** `make_a8_bundle.py` summed
every page of 150 drawn images per sample and correlated against the FCS
`Total Intensity (...)` columns at row offsets −1, 0, +1. On all six samples that ran:

| | offset −1 | **offset 0** | offset +1 |
| --- | ---: | ---: | ---: |
| mean r, range across samples | −0.01 .. 0.12 | **0.73 .. 0.84** | −0.02 .. 0.12 |

**Filename index = FCS row index.** `WaveformPresent` summed to exactly 20,000 on every
sample — it is the imaged-event flag. The sixth page is `Imaging_BP/788/225/LP/675`,
the far-red filter for PerCP-eF710 = ACRV1. Page order is as named: LightLoss, FSC, SSC,
AF488 (LDHC/AKAP4), PE (CD45), PerCP-eF710 (ACRV1).

**The fluorescence *images* are raw filter channels, not unmixed.** This is the important
subtlety the mapping table exposed. In the sperm wells the PE page correlates with the
AF488 total (r 0.88–0.91), not with the CD45 total — because sperm carry no CD45, and what
the PE filter sees there is AF488 spillover. In the PBMC wells it correlates with CD45
(r 0.88–0.95), as it should. The *scalar* parameters (`PE-A`, `DAPI-A`, …) are unmixed by
the instrument; the three `ImgB` *images* are not. Consequence: per-pixel fluorescence
targets carry cross-channel spillover, and image-to-image training needs a 3 × 3
imaging-channel unmixing estimated from the data — the S wells are AF488-dominant and the
P wells PE-dominant, which gives two of the three single-stain references for free.

Also as expected: `LightLoss` is extinction, so its raw page sum is mostly background and
does not match its own `Total Intensity` column. Background-subtract before use.

**What is not there:** event-level pairing to the CytPix. Different instrument, different
flow cell, different aliquot — the two runs share samples, not events. Within the A8 data
the pairing is exact; across instruments it is population-level only.

## A third instrument: the ImageStream run (`260813_Blair_Sperm_ISX`, surveyed 2026-10-04)

`Z:\Blair_Main\2026\081226\260813_Blair_Sperm_ISX\` — a different day (13 Aug), an Amnis
ImageStream, one sample file `ALL_1_400_1` plus single-stain compensation controls and a
no-primary-antibody control. Native formats: `.rif` raw, `.cif` compensated, `.daf`
analysis, `.ist` template.

**Decoded (2026-10-04, `analysis/isx_io.py`):** the codec is the one documented by the
open-source FlowSight reader — nibble-coded signed deltas with 2-D prediction for images,
byte-pair run-length for masks — and vectorises in numpy at **2.3 ms per object, ~47 s
for the whole file**, with no IDEAS, Bio-Formats or C++ involved. Two corrections to the
survey's reading, both found by rendering objects 0, 2 and 6 against the IDEAS gallery:

- **All 12 channels are written**, not the six in use: slot *k* is Ch(*k*+1), 67 px wide
  at 60×. The "unused" channels are not empty — Ch04/05/08/10/12 carry spillover from
  their neighbours, and **Ch06 is side scatter**, a third label-free input alongside the
  two brightfield cameras (Ch01, Ch09).
- This IDEAS version writes no leading throwaway value; the stream is exactly W × H
  residuals.

Object 0 decoded shows the head with its phase halo in Ch01, a ring in Ch02, ring-plus-
anterior-spot in Ch03, a blob in Ch07 and a spot in Ch11 — the same five panels the
gallery shows for object 0. The `.cif` is IDEAS-compensated, but the Ch02 ring is still
visible in Ch03, so residual AF488→PE spillover remains and the raw `.rif` plus the
single-stain controls would let us redo it if needed.

**Container, from `inspect_amnis.py`:** TIFF with one page per object holding all 12
channels tiled side by side (every page width divisible by 12; 28–90 px per channel),
each uint16 image page followed by a same-size uint8 mask page. Compression 30817/30818
is Amnis-private, but open decoders exist (`cifDataset`, R `IFC`, Bio-Formats). The
compensated `.cif` keeps the same layout.

**From the files themselves (`.daf`, `.ist`, and the acquisition XML inside the `.cif`):**

| | |
| --- | --- |
| Instrument | ImageStream, IDEAS 6.2.187, **60× objective** (the `400` in the filename is not magnification), 12-bit data |
| Objects | **20,000**, acquired under a `DAPI+ & Cells` gate with a 20,000 cap — the same selection design as the A8 |
| Channels with data | Ch01 and Ch09 (two brightfield cameras), Ch02, Ch03, Ch07, Ch11. The rest sit at default display range — unused |
| Compensation | single-stain `.rif` controls for AF488, PE, PerCP-eF710, DAPI; the `.cif` is IDEAS-compensated from them |

**Channel map, confirmed by Katelynn 2026-10-04 — the gallery labels are correct and
the template is stale:** Ch02 = LDHC/AKAP4 (AF488), **Ch03 = ACRV1 (PE)**, Ch07 = DAPI,
**Ch11 = TOMM20 (PerCP-eF710)**. Label-free: Ch01 and Ch09 (brightfield, two cameras) and
Ch06 (side scatter). Sample identity and donor are not known; treated as a single
sperm-only sample.

**Decode validated against IDEAS's own numbers.** The `.daf` feature block is column-major
(a 4-byte feature index, then 20,000 float64 per feature; `analysis/isx_daf.py`). Masked,
background-subtracted channel sums from the decoded images against IDEAS's
`Intensity_MC_ChXX` on 400 objects: **r = 0.9998 (Ch02), 1.0000 (Ch03), 1.0000 (Ch07),
1.0000 (Ch11)**, slope 0.99. Mask pixel count against `Area_M01` gives **0.1111 µm² per
pixel → 0.333 µm pixels at 60×**. That is a stronger check than an IDEAS TIFF export
would have been, and it was done without one.

Either way, **this run images all four markers of the original brief, including
Tomm20**, and CD45 is absent. Every gallery event is a sperm.

## Status

- [x] Repo + event-aware sampling tools scaffolded
- [x] Tooling rehearsed against fixtures in all three candidate layouts
- [x] Marker zips surveyed on the server — see above
- [x] RGBA channels checked — grayscale in an RGBA container; **no fluorescence in these zips**
- [x] Located the cytometry export — nine `.acs` archives in the run folder
- [x] Read the FCS in all nine archives — identical panel, identical voltages, 282,874 paired events
- [x] Tomm20 question resolved as far as the data can: **it is not in this experiment**
- [ ] **Ask the lab about Tomm20** — was it run, and if so where?
- [ ] Build the target tables (`make_targets.py --verify-images --compare-imaged`)
- [ ] Open `masks.zip` and see what the instrument's segmentation gives us
- [ ] Decide how to compensate before the targets are used
- [ ] Working sample built and pulled back to the laptop
- [ ] Approach chosen (see [approach.md](approach.md))
