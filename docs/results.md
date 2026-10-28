# Results

Measured findings, newest first. Everything here is reproducible from the scripts named
alongside it. The plan these test is in [approach.md](approach.md); what the data turned
out to be is in [task_brief.md](task_brief.md).

> **Erratum, 2026-10-27.** In every ImageStream result dated before 2026-10-27 the labels
> **ACRV1 and TOMM20 are swapped**: Ch03 (PE) is TOMM20 and Ch11 (PerCP-eF710) is ACRV1, not
> the other way round (evidence in the full-scale section below and in task_brief.md). Read
> "ACRV1" as TOMM20 and "TOMM20" as ACRV1 in the ImageStream pilot, the generalist, the
> stain-mask, the baseline and the provisional-ablation sections. A8 results are unaffected.
> The caches and runs on disk have been relabelled in place (`scripts/relabel_markers.py`).

---

## Full scale, both instruments: the stems models on the server

*2026-10-27 · `scripts/server_stems.ps1` at commit 7cfe9bc on the RTX A6000, results returned as
`server_results_20261026_2209.zip` and unpacked into `runs/srv_a8_stems`, `runs/srv_isx_stems`,
`figures/srv_*`. Both runs: `--arch stems --role-dropout 0.3 --width 64 --channel-scale
--full-frame --aug 1 --select-on val_event_r --tta`, ragged native-size caches from the new
`prepare`. A8: full run 260709, train replicate 2 (wells 2P, 2S, 2SP), test replicate 3,
40 epochs scheduled. ISX: the one tube, first half train, second half test, 30 epochs.
Comparison charts: `figures/srv_vs_laptop_a8.png`, `figures/srv_vs_laptop_isx.png`
(`sperm_fluorescence compare`).*

**One defect first.** The A8 training went NaN at epoch 22 of 40 (`runs/srv_a8_stems/log.csv`:
train and validation L1 are `nan` from epoch 22 on). CUDA training ran under fp16 autocast, and
an activation overflowed; the GradScaler cannot save a forward pass that is already infinite.
Checkpoint selection had already kept epoch 20 (per-marker bests at 15, 20 and 21), and every
number below comes from that checkpoint. Validation L1 had flattened by then (0.0859 → 0.0856
over epochs 15–21, as the October run also plateaued by epoch 13–25), so little was lost. The
loop now uses bfloat16 on GPUs that have it and skips any non-finite batch; the ISX run,
which reached epoch 30 cleanly, is unaffected.

### A8: three markers from light loss + FSC + SSC, 20,000 held-out objects per well

Within-well event r, the number that cannot be earned by recognising the cell:

| | sperm well (S) | mixed well (SP) | PBMC well (P) | pooled event r | pixel r |
|---|---|---|---|---|---|
| LDHC/AKAP4 | **0.75** | 0.75 | 0.50 | 0.74 | 0.75 |
| CD45 | 0.14 | 0.71 | **0.60** | 0.78 | 0.59 |
| ACRV1 | **0.32** | 0.53 | 0.44 | 0.66 | 0.58 |

Against the October full-scale plain U-Net (fixed frames, width 48, validation-L1 selection,
fold 2 → 3): LDHC within sperm 0.73 → **0.75**, CD45 within PBMC 0.57 → **0.60**, ACRV1 within
sperm 0.28 → **0.32**. Against the laptop sample of 600 (20 epochs, same architecture): LDHC
0.70 → 0.75, ACRV1 0.18 → 0.32, and the PBMC-well numbers rose most (LDHC 0.32 → 0.50, ACRV1
0.17 → 0.44), which is what 100× more training objects buys on the weak signals. CD45 within
the sperm well is a null: sperm are CD45-negative, that channel is read noise plus AF488
spillover, and the laptop's 0.77 there was a 200-object accident; 0.14 on 20,000 is the
honest value.

**What each view is worth at full scale** (`ablate.json`, the same checkpoint, views withheld
at evaluation; within-well event r, S / SP for the sperm markers, P for CD45):

| views | LDHC/AKAP4 | ACRV1 | CD45 (P) |
|---|---|---|---|
| light loss only | 0.56 / 0.53 | 0.19 / 0.46 | 0.26 |
| light loss + FSC | 0.71 / 0.72 | 0.28 / 0.51 | 0.39 |
| light loss + FSC + SSC | **0.74 / 0.74** | **0.31 / 0.51** | **0.59** |
| SSC only | 0.70 / 0.70 | 0.18 / 0.44 | 0.51 |

The laptop ablation holds: light loss alone loses the tail (LDHC 0.56 vs 0.74), FSC restores
most of it, SSC alone is nearly the whole model for LDHC and carries CD45. The input views in
`figures/srv_a8_views.png` show why: the SSC page is the only one on which a sperm's tail and
a PBMC's granular interior are sharp.

**Masks** (`masks.json`, 3,000 held-out objects, reference = thresholded measured stain):
LDHC/AKAP4 IoU 0.67, CD45 0.62, ACRV1 0.43, with 97–100 % of stained objects detected. The
A8 has no instrument mask to score against; these match the ISX numbers below for the same
markers, which do.

### ImageStream: four markers from brightfield + brightfield 2 + darkfield, 10,000 held-out objects

**The ACRV1 and TOMM20 labels were swapped in every ImageStream cache before this date.**
What the sheets called TOMM20 sat on the head, 1 µm from the DAPI centroid, with a pixel
correlation of 0.84 to DAPI; what they called ACRV1 was a bright short segment behind the
head. The files settle it. The acquisition XML inside the `.cif` names physical channel 2
(Ch03) "TOMM20 PE" and physical channel 4 (Ch05) "ACRV-1 PerCP-Cy5.5"; Ch11 is the same
642–745 nm band on the second camera and matches Ch05 pixel for pixel (r = 0.91). The
single-stain controls agree: the PE control lights Ch03, the DAPI control lights Ch07 and
Ch08 and nothing in Ch11. And on 372 held-out sperm whose tail leaves the head, the Ch03
stain centres **4.3 µm behind the nucleus along the tail** (99 % of cells), where the
midpiece is, while the Ch11 stain centres **0.8 µm ahead of it**, where the acrosomal cap
is (`figures/isx_ch03_vs_ch11.png`). The IDEAS gallery labels this was taken from were
wrong; the instrument config now reads Ch03 = TOMM20, Ch11 = ACRV1, and
`scripts/relabel_markers.py` has swapped the names in the existing caches, runs and JSON
on disk. Every number below uses the corrected names; the server figures (`figures/srv_isx_*`,
`runs/srv_isx_stems/*.png`) still carry the old labels until they are redrawn there
(`server_stems.ps1 -SkipTrain`). The A8 is unaffected: its PE is CD45 and its PerCP-eF710 is
ACRV1, the same conjugate as the ImageStream's Ch11.

Within-population event r (populations by the DAPI scalar; sperm is the one that matters):

| | sperm | round cells | DAPI-dim fragments | pooled event r | pixel r |
|---|---|---|---|---|---|
| LDHC/AKAP4 | **0.93** | 0.87 | 0.97 | 0.93 | 0.91 |
| ACRV1 | **0.96** | 0.62 | 0.94 | 0.95 | 0.81 |
| DAPI | **0.77** | 0.37 | 0.37 | 0.84 | 0.86 |
| TOMM20 | **0.98** | 0.93 | 0.99 | 0.97 | 0.94 |

At or above the pilot's level (same split, plain U-Net, 0.92–0.98 within sperm, names
swapped there too). DAPI within sperm is 0.77, the known case: a nuclear stain whose
intensity varies little between sperm, so r within the population is capped by measurement
noise. Against the laptop's 8,000-object, 12-epoch run every marker gained, ACRV1 most
(0.23 → 0.96): the laptop run had not converged on it.

**The converged input ablation replaces the provisional one** (12 epochs, 2026-10-26, which
said darkfield adds nothing):

| views | LDHC/AKAP4 | ACRV1 | DAPI | TOMM20 |
|---|---|---|---|---|
| brightfield only | 0.89 | 0.30 | 0.58 | 0.95 |
| brightfield + brightfield 2 | 0.91 | 0.37 | 0.71 | 0.98 |
| all three | **0.93** | **0.96** | **0.77** | **0.98** |
| darkfield only | 0.76 | 0.84 | 0.38 | 0.44 |

Within sperm. Brightfield carries LDHC/AKAP4 and TOMM20: the midpiece is a visible
thickening directly behind the head, so the mitochondrial sheath is nearly a direct read
of brightfield anatomy. **ACRV1 needs darkfield**: 0.30 from brightfield, 0.37 with the
second camera, 0.96 once darkfield is given, 0.84 from darkfield alone. The acrosomal cap
is invisible in transmitted light but changes how the anterior head scatters, and at 60×
the side-scatter image resolves that; the A8's lower-resolution SSC reaches only 0.32 on the
same marker. DAPI gains 0.06 from darkfield, chromatin density. The under-trained model had
not learned to use the darkfield stem yet, which is the caveat on any ablation read before
convergence. For a brightfield-only instrument (CytPix) this says: LDHC/AKAP4 and TOMM20
yes, DAPI partly, ACRV1 no.

**Masks against the masks IDEAS stores** (1,000 held-out objects): LDHC/AKAP4 IoU 0.73
(thresholded truth vs IDEAS: 0.68), ACRV1 0.60 (0.59), DAPI 0.72 (0.71), TOMM20 0.63 (0.70).
The predicted masks agree with the instrument's about as well as the measured stain itself
does, on every marker. (These were computed with the IDEAS mask of the right channel by
construction: the mask lookup went by column, and the columns were right.)

### A8 rerun, 40 epochs under bfloat16 (2026-10-27)

*`server_results_20261027_1941_a8.zip`, commit db78bce, RTX A6000. Same data, split and
settings as above; the fp16 defect is gone (40 epochs logged, no NaN, validation L1 0.0850 at the
best epoch, 37). The epoch-20 fp16 results are kept in `runs/srv_a8_stems_fp16_ep20` and
`figures/srv_fp16_ep20/`. The zip also carried both runs' `best.pt` (`-Checkpoints`) and the
redrawn ISX figures with the corrected labels, including the submission panel
`figures/srv_isx_panel.png`.*

Counts, from `results.json`: 57,000 training + 3,000 validation events (replicate 2: wells 2S, 2P,
2SP, 20,000 imaged events each), 60,000 test events (replicate 3, the same three wells). The
repo records nothing about the donor(s) or their semen parameters; the run folder is
`260709_Blair_Sperm_Cytpix` and the wells are sperm alone, PBMCs alone and a mixture.

Within-well event r (epoch-20 fp16 value in brackets):

| | sperm well (S) | mixed well (SP) | PBMC well (P) | pooled event r | pixel r |
|---|---|---|---|---|---|
| LDHC/AKAP4 | **0.75** (0.75) | 0.76 | 0.52 | 0.75 | 0.76 |
| CD45 | 0.10 | 0.71 | **0.61** (0.60) | 0.78 | 0.60 |
| ACRV1 | **0.37** (0.32) | 0.53 | 0.44 | 0.66 | 0.59 |

Twenty more epochs bought ACRV1 within sperm 0.32 → 0.37 and nothing measurable elsewhere; the
A8 model is converged on its weak signals. Masks (3,000 held-out, thresholded truth): LDHC/AKAP4
IoU 0.68, CD45 0.62, ACRV1 0.42, 97–100 % detected. Ablation within S (CD45 within P): light loss
only 0.59 / 0.35 / 0.23 (LDHC / CD45 / ACRV1), + FSC 0.74 / 0.44 / 0.32, + SSC 0.74 / 0.60 / 0.36,
SSC only 0.69 / 0.54 / 0.22.

**Sperm versus PBMC on the predicted stains**, scored with this checkpoint on the laptop's
600-object held-out sample (200 per well; the full-scale version needs the `accuracy` command
on the server). AUC for PBMC-well against sperm-well events, per-event stain sum:

| | measured stain | predicted stain |
|---|---|---|
| CD45 (PBMC-high) | 0.987 | **0.954** |
| ACRV1 (sperm-high) | 0.906 | **0.969** |
| LDHC/AKAP4 | 0.62 | 0.50 |

Predicted CD45 alone separates the PBMC well from the sperm well at 0.95, and predicted ACRV1
does it better than the measured ACRV1 (the prediction is a smoothed, PBMC-negative version of
a dim stain). Measured LDHC is weakly discriminative at the well level (replicate 3's PBMC well
carries LDHC-channel signal, see the P:S note above) and the prediction inherits none of it.
These are well labels, not cell labels: both wells are impure.

### What to run next

- The A8 for its full 40 epochs under bfloat16, to confirm that epochs 20–40 add nothing
  (expected) and to have a clean log. Same command after `git pull`; `-SkipPrepare` reuses the
  caches.
- Replicate 3 → 2 for the A8 (`--train 3 --test 2`), as in October, so the LDHC replicate
  reversal is checked against this model too.

---

## Baseline: the plain fixed-channel U-Net against the role stems, 10 epochs, both instruments

*2026-10-26 · `--arch stack` (the architecture of every earlier run) vs `--arch stems` with all
views and no dropout, same caches, splits, seed and 10-epoch budget. A8: the laptop bundle
(600 / 600). ISX: 4,000 / 4,000 objects of the one tube (`data/isx_roles`, first/second
half). Both ISX runs are under-trained at this budget (TOMM20 ~0.2 where 15 epochs on
9,500 objects reached 0.93), so the comparison is between equals, not against the best.*

Within-population event r (A8: sperm / mixed well; ISX: sperm by the DAPI scalar):

| | LDHC/AKAP4 | ACRV1 | CD45 | DAPI | TOMM20 |
|---|---|---|---|---|---|
| A8 stack | 0.69 / 0.69 | 0.18 / 0.45 | 0.80 / 0.88 | — | — |
| A8 stems | 0.66 / 0.64 | 0.04 / 0.46 | 0.75 / 0.88 | — | — |
| ISX stack | 0.87 | 0.69 | — | 0.63 | 0.26 |
| ISX stems | 0.84 | 0.64 | — | 0.56 | 0.14 |

**The stems trail the plain stack at 10 epochs**, by 0.03 to 0.05 on most cells and by 0.07
on ISX DAPI and 0.12 on ISX TOMM20. At 20 epochs on the A8 the gap had closed (stems with
all views 0.68 / 0.44 / 0.88 vs the stack's 0.66 / 0.45 / 0.90 within-well numbers from the
generalist scratch run), so this reads as slower convergence rather than a lower ceiling:
the stems add a layer per role and average the result, and the trunk sees a mean of three
feature maps instead of the three raw pages, which takes longer to disentangle. Where the
training budget is short, the plain stack is the stronger baseline; where it is adequate,
the two are equivalent and only the stems can serve a subset of views.

**ImageStream input ablation at 12 epochs (provisional, under-trained).** Brightfield alone
carries LDHC/AKAP4 within sperm (0.85 of a 0.87 best) because camera brightfield shows the
tail; the second camera adds ~0.14 to ACRV1 and DAPI; darkfield adds nothing and alone
reaches 0.68 on LDHC/AKAP4. The ISX's side-scatter image is a blurred blob, the opposite of
the A8 where SSC is the sharpest view. Figures: `figures/ablation_views_isx.png`,
`figures/views_isx_dropout.png`. A 30-epoch rerun was started and stopped at the user's
request; the converged numbers are in the full-scale section above, and they reverse the
darkfield finding (ACRV1 needs it; the names here are swapped, see the erratum).

---

## Input ablation on the A8: what each label-free view is worth, and one modular model for all of them

*2026-10-26 · `python -m sperm_fluorescence train --arch stems`, `ablate`. Laptop, A8 sample
bundle (600 training objects in replicate 2, 600 held out in replicate 3, three markers),
width 32, 20 epochs, `--channel-scale --full-frame --aug 1`, one seed. Four models that
differ only in the label-free pages they see; the fourth is trained with role dropout
(p = 0.3) and then scored under each subset without retraining.*

| model trained on | LDHC/AKAP4 within S | ACRV1 within SP | CD45 within SP | CD45 within S |
|---|---|---|---|---|
| light loss only | 0.37 | 0.33 | 0.83 | 0.80 |
| light loss + FSC | 0.61 | 0.39 | 0.86 | 0.44 |
| light loss + FSC + SSC | 0.68 | 0.44 | 0.88 | 0.61 |
| all three, role dropout, scored with all three | 0.70 | 0.48 | 0.88 | 0.77 |
| … scored with light loss + FSC | 0.65 | 0.42 | 0.81 | 0.37 |
| … scored with light loss only | 0.47 | 0.41 | 0.78 | 0.31 |
| … scored with FSC + SSC (no light loss) | 0.69 | 0.48 | 0.88 | 0.80 |
| … scored with SSC only | 0.67 | 0.37 | 0.87 | 0.70 |

**Light loss alone loses the tail.** LDHC/AKAP4 within sperm falls from 0.68 to 0.37 when
FSC and SSC are removed, and the pooled event r from 0.66 to 0.29. The marker lives on the
flagellum, and the light-loss page barely shows the flagellum (A8 pages figure); the
scatter views do. ACRV1 and CD45, both head or whole-cell signals, lose little.

**FSC carries most of the gain, SSC the rest.** Adding FSC recovers LDHC/AKAP4 to 0.61;
adding SSC takes it to 0.68 and lifts every marker a few hundredths. SSC on its own
(from the dropout model) already gives LDHC/AKAP4 0.67 and CD45 0.87: for these
markers, side scatter is the single most informative view and light loss the least.

**One modular model matches the specialised ones.** The role-dropout model, trained once,
equals or beats the fixed three-view model with all views (0.70 / 0.48 / 0.88 vs 0.68 /
0.44 / 0.88) and matches the fixed two-view model when FSC is withheld (0.65 vs 0.61).
With light loss alone it is better than the model trained only on light loss on LDHC/AKAP4
(0.47 vs 0.37) and ACRV1, and slightly worse on CD45. The averaging stems do not cost
accuracy at this width.

**A fixed model cannot be ablated after the fact.** Withholding SSC from the fixed
three-view model drops its CD45 to −0.13: it never learned to work without that channel
and a zero image is not "absent". This is the case for role dropout: the modular model is
the only one that can honestly answer "what if this instrument had fewer views".

Caveats: 600 training objects and one seed, so differences under ~0.05 are noise; the CD45
within-S column is a handful of CD45-positive events in a sperm well and swings the most.
The server run at full scale is the confirmation. Figure: `figures/ablation_views.png`.

---

## Generalist A8 + ISX on the laptop: one model, two instruments, and masks IDEAS would draw

*2026-10-05 · `analysis/prepare_generalist.py`, `analysis/train_a8.py` (availability
masks, `--include`, `--init`), `analysis/stain_masks.py`, `analysis/make_sheet.py`. Laptop
MPS, width 32, batch 16, `--channel-scale --full-frame --aug 1`. ISX: 10,000 train /
10,000 test (first/second half of the one tube). A8: the laptop sample bundle, 200 objects
per well in replicate 2 (train) and 3 (test), zoomed 1.25× to the ISX pitch, each A8 object
repeated 6× per epoch. Five target channels; a channel an instrument did not stain is
masked out of the loss, validation and scoring for that instrument.*

### Four runs

| run | train on | epochs | ISX event r (LDHC / ACRV1 / DAPI / TOMM20) | A8 within-well (S / P / SP) LDHC | ACRV1 | CD45 |
|---|---|---|---|---|---|---|
| A8 scratch | A8 only | 20 | — | 0.66 / 0.34 / 0.78 | 0.19 / 0.22 / 0.45 | 0.83 / 0.55 / 0.90 |
| ISX only | ISX only | 15 | 0.968 / 0.965 / 0.888 / 0.934 | — | — | — |
| A8 fine-tune | A8, from the ISX-only checkpoint | 20 | — | 0.65 / 0.34 / 0.78 | 0.13 / 0.20 / 0.39 | 0.23 / 0.55 / 0.89 |
| joint | both | 15 | 0.957 / 0.953 / 0.906 / 0.858 | 0.10 / 0.34 / −0.03 | −0.05 / 0.20 / −0.16 | 0.58 / 0.49 / 0.79 |

**Joint training hurts the A8 badly at this scale.** The ISX side comes out within a few
hundredths of the ISX-only run (DAPI even gains, TOMM20 loses 0.08), but on the A8 the
within-well numbers for LDHC/AKAP4 and ACRV1 collapse to zero and CD45 drops 0.1–0.25.
The sheet shows why: on A8 sperm the joint model paints ACRV1 along the tail and CD45 on
the head — ISX anatomy and a round-cell prior carried across instruments. Two one-hot
planes are not enough to keep the instruments apart when one contributes 9,500 clean
objects per epoch and the other 480 noisy ones repeated six times; the shared decoder
settles on the ISX's answer. Pooled event r (0.94 / 0.95 / 0.91 / 0.86 / 0.82) hides all of
this, because it is earned by telling instruments and cell types apart. The P:S ratio
makes the same point: LDHC predicted 2.0 against a true 0.72.

**Fine-tuning from the ISX checkpoint does not help the A8** on this sample either: every
within-well number lands on the scratch run's within noise (the CD45-in-S drop is a
handful of CD45-positive events in a sperm well, not a trend). With 600 A8 training
objects and 20 epochs the pretrained features are simply overwritten; whether the
transfer shows up at full A8 scale (57,000 events per fold) is a server question. The ISX
side reproduces the earlier joint pilot within 0.01 on every marker, so the five-channel
cache and the availability masking cost nothing.

### Stain masks, scored against the masks IDEAS stores

The `.cif` carries IDEAS's per-channel masks (M02, M03, M07, M11), so on the ISX the
predicted masks can be scored against exactly what IDEAS draws. A predicted channel is
thresholded at a fraction of that object's own peak, the fraction chosen per channel on
training objects to maximise IoU against IDEAS. A population-wide absolute threshold was
tried first and loses 0.06–0.08 IoU on every marker: a sperm head's faint predicted halo
sits above any cut that still keeps a dim diffuse round cell.

ISX held-out, 300 objects, ISX-only model:

| marker | IoU vs IDEAS | Dice | present/absent agree | *thresholded truth vs IDEAS* |
|---|---|---|---|---|
| LDHC/AKAP4 | 0.72 | 0.83 | 1.00 | 0.69 |
| DAPI | 0.72 | 0.83 | 1.00 | 0.70 |
| ACRV1 | 0.64 | 0.75 | 0.96 | 0.69 |
| TOMM20 | 0.62 | 0.72 | 0.90 | 0.62 |

The last column is the ceiling that matters: thresholding the *real* fluorescence image
reproduces IDEAS's mask only to IoU 0.62–0.70, because IDEAS's default mask is its own
algorithm, not a threshold. The predicted masks match IDEAS as well as the truth does.
On the sheet the acrosome cap, nuclear head, midpiece and tail masks each land on the
IDEAS outline.

A8 has no instrument mask, so its reference is the thresholded truth (a rule that scores
0.62–0.70 against IDEAS on ISX, above). A8 held-out, 600 objects: LDHC/AKAP4 IoU 0.63–0.64,
CD45 0.61, ACRV1 0.40, for scratch and fine-tuned alike; the joint model drops to 0.50 /
0.50 / 0.27. On the ISX the joint model's masks are the best of the three (LDHC/AKAP4 0.75,
DAPI 0.75, ACRV1 0.66, TOMM20 0.59), consistent with its ISX event r. The predicted masks are cleaner
than the A8 truth masks because the network averages the A8 noise away, which costs IoU
against a noisy reference even where the shape is right.

### Where this leaves the generalist

One model for both instruments is not free. At laptop scale the options are, in order:
(1) **per-instrument models**, which is what the A8 scratch run already is, and the
only configuration where the A8 holds its own; (2) **ISX-pretrain then A8 fine-tune**,
which cost nothing here but gained nothing either, and deserves one run at full A8 scale
on the server before it is dismissed; (3) **joint training**, which needs either a
balanced A8 share (the full 57,000-event A8 set, not 600) or instrument-specific decoder
heads before it is worth another run. The generalist cache and the availability masking
are the right scaffolding for all three and are kept.

Two practical notes from running this on a 16 GB laptop. Ragged batches give the caching
allocator a block per distinct (H, W); with two instruments' worth of shapes the MPS cache
reached 25 GB and the process stalled in swap. The trainer now returns cached device memory
after every epoch and every 20 evaluation batches, and evaluates at batch 64. Training
epochs still peak near 13 GB; the server's 48 GB card will not notice.

### Why the A8 images are what they are

CellView does not take a picture. One laser is split into a row of radiofrequency-tagged
beamlets, a single PMT records the sum, and the image is demodulated out of it: each pixel
integrates microseconds of a few milliwatts, so shot noise dominates, worst on PerCP
(ACRV1); the beamlets form a band, so noise is structured by row (the streaks the cache
builder removes); unmixing AF488/PE/PerCP subtracts noise along with spillover; and the
label-free channel is loss of a single coherent wavelength, not white-light brightfield.
The ISX is a 60× TDI camera at a few hundred events per second. The S8 inherits the A8's
imaging, so for sorting these are the images the model has to live with.

---

## Zero-shot ISX → CytPix: fails outright

*2026-10-05 · `analysis/predict_cytpix.py`, the laptop ISX joint model on 15 CytPix
replicate-1 crops (`sperm_pbmc` sample, 1S/1P/1SP). No ground truth exists for these —
replicate 1 is the unstained control — so this is a qualitative look.*

The model paints saturated stain over the **background** and leaves the cell dark, on
every object and every marker. Tight cropping, a Gaussian blur to match the ISX's smooth
background, and mirroring brightfield into the empty scatter slot change nothing. Pixel
pitch is not the problem: a sperm head spans ~10 px on the CytPix and ~9 on the ISX.

Three causes, in order of how much they are our fault:

1. **This checkpoint learned nothing about background.** The laptop pilot predates
   `--full-frame`; its loss never covered the padded zone, and on a CytPix crop nearly
   the whole frame is background. The GPU `isx_rag_joint` model was trained full-frame
   and is the right checkpoint to retry with.
2. **Domain gap in the input.** The CytPix background is a noise floor (the `sperm_pbmc`
   finding), the ISX's is smooth; the CytPix has one brightfield channel where the model
   expects two plus side scatter; the contrast mechanisms differ. Per-event
   standardisation turns CytPix noise into texture the model has never seen.
3. **It was never going to work zero-shot**, and we said so: experiment C in the
   generalist plan. A model that has never seen an instrument's images does not transfer
   to it. The realistic CytPix route is an ISX-pretrained encoder fine-tuned on CytPix
   data — and since the CytPix has only scalar targets (replicates 2 and 3), that is the
   image-to-scalar head from the original plan, on top of an encoder that already knows
   sperm anatomy.

---

## Joint vs per-marker models: the shared encoder wins or ties on every marker

*2026-10-05 · Laptop: fixed-frame cache, width 32, 20 epochs, scored within nucleated sperm.
GPU (RTX A6000): ragged native-size cache, width 48, 40 epochs, `--full-frame`, whole-sample
event r. Same architecture and budget per arm; a dedicated model has one output channel.*

| event r | joint, laptop | dedicated, laptop | joint, GPU | dedicated, GPU |
| --- | ---: | ---: | ---: | ---: |
| LDHC/AKAP4 | **0.972** | 0.951 | 0.983 | — |
| ACRV1 | **0.981** | 0.974 | 0.981 | — |
| DAPI | 0.922 | *(running)* | 0.916 | — |
| TOMM20 | 0.976 | *(queued)* | **0.978** | 0.977 |

**Dedicated models never beat the joint one.** LDHC loses 0.02 on its own, ACRV1 loses
0.007, TOMM20 ties. The 7.5-minute GPU runs and the 15-minute laptop runs agree.

**The joint model learns the hard channel faster.** TOMM20 — dimmest, last to train — hit
0.87 by epoch 10 in the joint model and 0.53 on its own; LDHC was ahead in the joint model
at every checkpoint. The other markers' supervision teaches the shared trunk the anatomy
(head, midpiece, tail) before a weak channel could on its own. That is the multi-task
argument made by the data, and the case for per-marker networks — strongest where a
channel is hard — is where it fails most clearly.

**Per-marker checkpoints are moot on this instrument.** Every marker's best validation
epoch in the 40-epoch GPU run was 40. Nothing peaked and declined, so the stopping effect
that cost ACRV1 on the A8 (peak at epoch 15–20, decline to 40) does not occur here. Keep
the mechanism for the A8; it changes nothing on the ISX.

**Decision:** one shared-encoder model with per-channel loss scaling, per-marker
checkpoints retained as a free safeguard. Four networks would cost 4x for no gain.

The ISX split is nearly saturated (laptop and GPU within 0.01 of each other despite 2x
epochs and native-size objects), so these differences are small in absolute terms; a
second sample is what would make them sharper, and what would make any of this a
generalisation claim.

---

## ImageStream pilot — all four markers predicted within nucleated sperm at r ≈ 0.92–0.98

*2026-10-04 · `analysis/prepare_isx.py`, `train_a8.py` (width 32, 20 epochs, `--channel-scale`,
laptop MPS, 21 min), `eval_subset.py`. One sample, 20,000 objects; trained on the first
10,000 by acquisition order, tested on the second 10,000. Inputs Ch01 + Ch09 (two
brightfield cameras) + Ch06 (side scatter); targets the IDEAS-compensated Ch02/Ch03/Ch07/Ch11.*

| within nucleated sperm (6,241 test objects) | pixel r | event r | morphology floor (42 IDEAS features) |
| --- | ---: | ---: | ---: |
| LDHC/AKAP4 | 0.947 | **0.972** | 0.932 |
| ACRV1 | 0.950 | **0.981** | 0.912 |
| DAPI | 0.923 | **0.922** | 0.811 |
| TOMM20 | 0.942 | **0.976** | 0.719 |

Whole-sample numbers are within 0.01 of these for the three sperm markers; DAPI drops to
0.40 on the DAPI-low fragments, as it should — they have little DNA to predict.

**Correction (2026-10-05): the "nucleated" subset contains a round-cell population, and
it was propping up DAPI.** 4–6% of objects are large, round, granular cells at 18–27× the
sperm DAPI level, bright in every stain including LDHC and ACRV1 — which argues for
immature germ cells (LDHC is testis-specific; ACRV1 marks the forming acrosome in round
spermatids) over leukocytes, though with no CD45 in the panel the two cannot be separated.
Scored on **sperm only** (DAPI between the fragment trough and 4× the sperm median,
n = 5,660): LDHC 0.974, ACRV1 0.985, TOMM20 0.978 — unchanged — but **DAPI 0.800**, down
from 0.922. Among true haploid sperm there is little DNA variance to explain, and the
0.92 came partly from the round cells' extreme values. DAPI's honest number within sperm
is 0.80. On the round cells alone the model is weakest on TOMM20 (0.66): too few examples.

**This is a different regime from the A8.** There, ACRV1 within sperm reached 0.25 from
57,000 training events; here it is 0.98 from 10,000, and the prediction sheets show the
acrosomal cap drawn on the anterior head from epoch 3 onward. Four things differ, and
the comparison has to keep them apart:

1. **Contrast.** 60× brightfield with a phase-like halo on two cameras plus side scatter,
   against the A8's extinction channel. The acrosome is *visible* in the ISX Ch01 panel;
   it was not in LightLoss. This is the explanation the phase-microscopy literature
   predicts, and the one that matters scientifically.
2. **Target quality.** The ISX stains are bright, 12-bit, compensated and masked; the A8
   pages were tiny float values with heavy noise. A noisy target caps achievable r
   regardless of the input. "Higher quality images" was true on both sides of the pair.
3. **The split.** First half versus second half of one tube, minutes apart — no batch
   shift at all. The A8's held-out replicate carried real staining and gain differences.
   The A8's own within-replicate random split gave LDHC 0.66 against 0.38 across, so
   part of the gap is the evaluation, not the instrument. A same-split A8 run is the
   control this comparison needs.
4. **A shared per-object factor.** The four stains inter-correlate at 0.48 within
   nucleated sperm, so something per-cell — focus, in-frame fraction — scales all of
   them, and it is label-free-visible. It cannot explain 0.98 on each marker alone, but
   the marker-*ratio* numbers (e.g. ACRV1 relative to DAPI) are the version of the result
   that is immune to it, and should be reported alongside.

The 38.5% DAPI-low population is why the morphology floor read 0.96 on the whole sample:
telling fragment from sperm is most of that. Within nucleated sperm it is still high
(0.72–0.93), and the network clears it on every marker, by the most on TOMM20 (+0.26)
and DAPI (+0.11).

---

## Full-scale virtual staining — 57,000 training events per fold, RTX A6000

*2026-10-04 · `analysis/train_a8.py`, width-48 U-Net, 40 epochs, ~1 h per fold. Checkpoint
chosen on validation L1 within the training replicate; the test replicate is never used
for selection. Outputs in `runs/fold_2to3/` and `runs/fold_3to2/` on the server.*

### The headline table

| | fold A: train 2 → test 3 | | | fold B: train 3 → test 2 | | |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| marker | pixel r | event r | **within-type r** | pixel r | event r | **within-type r** |
| LDHC/AKAP4 (within sperm) | 0.715 | 0.729 | **0.727** | 0.713 | 0.720 | **0.708** |
| CD45 (within PBMC) | 0.512 | 0.756 | **0.567** | 0.523 | 0.737 | **0.525** |
| ACRV1 (within sperm) | 0.505 | 0.601 | **0.282** | 0.557 | 0.572 | **0.227** |

*Within-type r is event r computed inside one well type, where recognising the cell earns
nothing. It is the number to quote.*

### What scale changed, and a correction

**ACRV1 within sperm is not zero.** At 600 training events it read 0.02 / −0.00 and I
wrote "zero is zero". At 57,000 it reads **0.28 / 0.23**, replicated across folds on
20,000 test sperm each — small, but far from chance. The pilot was underpowered to see
r ≈ 0.25, and the earlier claim was overconfident. What the signal *is* remains the
question: ACRV1 is intra-acrosomal, so its total scales with acrosome size, and acrosome
size is visible in scatter. An r of 0.25 — six percent of variance — is consistent with
the model reading acrosome *extent*, not acrosomal *state*. The optical-limit conclusion
softens from "nothing" to "a sliver, probably geometric".

**LDHC/AKAP4 is the strong result**: 0.73 / 0.71 within sperm, up from 0.58 / 0.61 in
the pilot. Half the variance in flagellar stain intensity is predictable from label-free
scatter. **CD45 within PBMCs** is 0.57 / 0.53 — essentially unchanged from the pilot, so
that is probably its ceiling: CD45 level varies by leukocyte subtype, which morphology
only partly resolves.

### Controls

| | fold A pred / true P:S | fold B pred / true P:S | want |
| --- | --- | --- | --- |
| ACRV1 | 0.00 / 0.02 | 0.20 / 0.02 | ≪ 1 ✓ |
| CD45 | 17.98 / 7.54 | 8.49 / 9.09 | ≫ 1 ✓ (fold A overshoots: sperm predictions near zero inflate the ratio) |
| LDHC/AKAP4 | 1.07 / **0.53** | 0.44 / **1.07** | ≪ 1 — see below |

LDHC's control reverses between folds *in the truth*: replicate 3 has sperm > PBMC
(0.53, the biology), replicate 2 has PBMC > sperm (1.07, the antibody binding PBMCs
non-specifically). Each model faithfully reproduces its training replicate and so fails
on the other. This is now established on 60,000 events per replicate, not inferred.
**Replicate 3 is the LDHC reference; replicate 2's LDHC channel should not be trained on
without saying so.**

### Training dynamics, and the fix they point at

Validation L1 plateaued by epoch 13–25 and drifted up after; train L1 kept falling — mild
overfit to the training replicate. Test ACRV1 within-sperm r **peaked at 0.30 around
epochs 15–20 and declined to 0.26** by epoch 40, so selection on validation L1 is not
selecting for the metric we care about.

A second thing in the loss: a single shared target scale lets the brightest channel
dominate. AF488 is ~3× PerCP, so ACRV1 received roughly a fifth of the gradient. Two
flags added for the next run — `--channel-scale` (one scale per channel) and
`--select-on acrv1_sperm` (checkpoint on within-sperm ACRV1 r, measured on the validation
slice of the training replicate, never on test). Neither changes the defaults, so these
results stay reproducible.

---

## Pilot image-to-image model — the stain is predictable from label-free pages

*2026-10-01 · `analysis/pilot_floor.py` and `analysis/pilot_unet.py` on
`data/a8_bundle_seed0.zip` — 600 training events, 600 test, held out by replicate.
A feasibility check on a laptop (74 s per fold on MPS), not a result; the full run is
120,000 events on the server GPU.*

Input: the three label-free pages, normalised per event. Target: the three fluorescence
pages after the triangular unmix. Model: a 0.5M-parameter U-Net with no global pooling,
L1 loss on a validity mask. The **floor** is a ridge regression from a 5 × 5 label-free
patch to the centre pixel — the cheapest model that could possibly work.

### Fold A: train replicate 2 → test replicate 3

| marker | floor pixel r | **U-Net pixel r** | floor event r | **U-Net event r** | pred P:S | true P:S | want |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| LDHC/AKAP4 | 0.364 | **0.607** | 0.155 | **0.540** | 1.45 | 0.72 | ≪ 1 |
| CD45 | 0.387 | **0.506** | 0.567 | **0.873** | 9.35 | 8.04 | ≫ 1 |
| ACRV1 | 0.262 | **0.388** | −0.346 | **0.513** | 0.35 | 0.20 | ≪ 1 |

### Fold B: train replicate 3 → test replicate 2

| marker | U-Net pixel r | U-Net event r | pred P:S | true P:S | want |
| --- | ---: | ---: | ---: | ---: | --- |
| LDHC/AKAP4 | 0.623 | 0.613 | **0.61** | 0.96 | ≪ 1 |
| CD45 | 0.532 | 0.880 | 10.69 | 9.96 | ≫ 1 |
| ACRV1 | 0.407 | 0.457 | 0.37 | 0.25 | ≪ 1 |

*P:S = median predicted page sum on PBMC-well events ÷ sperm-well events, in the stained
test replicate. The floor scores ~1.12 on all three — a local patch cannot tell a PBMC
from a sperm.*

### What it says

**ACRV1 is *not* read from label-free pages — the head is.** The cross-type event r of
0.51 / 0.46 looked like the project's headroom. It is not. Computed **within sperm only**,
where recognising the cell earns nothing, ACRV1 event r is **0.017 and −0.001** across
the two folds — while true ACRV1 on sperm spans 11–14× between the 5th and 95th
percentile, so there is a real intact-versus-reacted spread to explain and the model
explains none of it. The spot on the head is a prior ("sperm have acrosomes"), not a
reading. This is the optical limit the phase-microscopy literature predicted, now shown
with pixel-registered ground truth rather than inferred. At 600 training events a true
within-sperm r of ~0.3 would have been visible; zero is zero. The full run will say
whether weak signal appears with 60,000 — `train_a8.py` now reports within-well r, and
for ACRV1 that is the only number that matters.

| within one well type, event r | sperm only | PBMC only | mixture |
| --- | ---: | ---: | ---: |
| LDHC/AKAP4 | **0.58 / 0.61** | 0.35 / 0.51 | 0.62 / 0.66 |
| CD45 | 0.65 / 0.04 | **0.53 / 0.59** | 0.88 / 0.83 |
| ACRV1 | **0.02 / −0.00** | 0.14 / 0.47 | 0.38 / 0.29 |

*(fold A / fold B)* LDHC carries genuine within-sperm signal — the tail is visible in FSC
and SSC and its extent predicts the stain. CD45 carries within-PBMC signal. ACRV1 carries
none within sperm. Across cell types all three look predictable; only two are.

**CD45 is the easy one, as everywhere.** Event r 0.87–0.88 and the right cell-type
behaviour (P:S 9–11 against 8–10 true). The model has learned "round cell".

**LDHC fails the control in fold A and passes it in fold B — and that is the staining,
not the model.** Trained on replicate 2, the model puts *more* LDHC on PBMCs than sperm
(P:S 1.45). Replicate 2's LDHC/AKAP4 antibody binds PBMCs non-specifically — the Attune
scalars showed it, the A8 contact sheet shows AF488 blobs on PBMCs, and the fold-A
model reproduces its training data faithfully. Trained on the cleaner replicate 3 it gets
the direction right (0.61). The "≪ 1" expectation is biology; replicate 2's staining
violates it. **Treat replicate 3 as the reference for LDHC, and report both folds.**

**Every number clears the floor.** Pixel r up by 0.12–0.25, event r up by 0.3–0.85,
and the cell-type controls go from uniformly uninformative to mostly correct.

### Known pipeline defects, to fix before the full run

- A bright band along the padded edge in some predicted LDHC frames (visible on the
  sheet). Padding events to 80 rows leaves a boundary the network learns; crop
  predictions to the valid region, or pad with reflection rather than zeros.
- The CellView horizontal scan-line streak is present in the targets and will be learned
  as signal at scale. Mask it or model it.
- 600 events is far too few to speak to generalisation; the ~0.1 spread between folds is
  mostly sample size.

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
40,000 are **not** a negative control for image-to-image work — a correct virtual-stain
model should predict the stain on an unstained cell, since the label-free pages cannot
know the tube was not stained. The controls are cell-type ones inside the stained
replicates: ACRV1 and LDHC low on PBMCs, CD45 low on sperm. Replicate 1 is extra
label-free input, nothing more.

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
