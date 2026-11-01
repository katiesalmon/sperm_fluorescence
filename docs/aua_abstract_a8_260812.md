# A8, 2026-08-12 plate (LDHC / TOMM20 / ACRV1): methods and results paragraphs for the abstract

*Drafted 2026-11-01 from `runs/srv_a8_stems_260812` (results.json, masks.json, ablate.json), scored
against the raw measured stains on the 20,000 imaged events of well B5. Denoised scoring not yet run.*

## Methods (training)

A multi-stem U-Net was trained to predict the LDHC, TOMM20 and ACRV1 fluorescence images from
the light-loss, forward-scatter and side-scatter images of each event. Training used 19,000
imaged events from one acquisition of the fully stained ejaculate (well A5, antibodies at 1:400),
with a further 1,000 events from the same acquisition held aside for validation to select the
checkpoint; the model was trained for 40 epochs and the epoch with the highest validation event
correlation was kept. Accuracy was then measured on 20,000 imaged events from an independent
acquisition of the same sample (well B5, antibodies at 1:800), which the model never saw.

## Results

On the 20,000 held-out events, the predicted LDHC image reproduced the measured flagellar stain
closely: per-cell intensity correlated at r = 0.82 and the images agreed pixel by pixel at
r = 0.81, with the area the predicted stain covered overlapping the measured area at IoU = 0.71
(Dice 0.83) and the stain detected in 100% of cells. The two head and midpiece markers were
located correctly but their brightness was predicted less well: for TOMM20 the pixel correlation
was 0.51 and the mask overlap IoU = 0.60 (Dice 0.72, 98% detected), and for ACRV1 the pixel
correlation was 0.63 and IoU = 0.63 (Dice 0.75, 98% detected), while per-cell intensity
correlated at only r = 0.33 and 0.31. The held-out well was stained at half the antibody
concentration of the training well, which compresses the range of per-cell intensities that the
event correlation depends on; the pixel and overlap measures, which score where the stain sits
rather than how bright it is, were unaffected. Restricting the model's inputs showed where each
marker's information lies: light loss alone predicted LDHC at r = 0.65, adding forward scatter
raised it to 0.78 and adding side scatter to 0.81, and side scatter alone reached 0.75; for the
midpiece and acrosome the same progression ran from 0.19 to 0.33 (TOMM20) and from 0.21 to 0.31
(ACRV1), so the two scatter images carry most of what the model knows about the small head and
midpiece compartments.

## Numbers behind the paragraph

| marker | event r | pixel r | IoU | Dice | detected |
|---|---:|---:|---:|---:|---:|
| LDHC | 0.82 | 0.81 | 0.71 | 0.83 | 100% |
| TOMM20 | 0.33 | 0.51 | 0.60 | 0.72 | 98% |
| ACRV1 | 0.31 | 0.63 | 0.63 | 0.75 | 98% |

Input ablation, event r: light loss / + FSC / + SSC / SSC alone = LDHC 0.65 / 0.78 / 0.81 / 0.75,
TOMM20 0.19 / 0.29 / 0.33 / 0.26, ACRV1 0.21 / 0.28 / 0.31 / 0.24. IoU on 1,000 held-out cells with
thresholds calibrated on 400 training cells; correlations on all 20,000.

## Before submission

- Rescore against denoised measured stains (`--denoise-measured smooth`, as for the July abstract)
  and decide which set the abstract quotes; say which in the methods.
- The dilution difference between A5 and B5 is a real confound for event r; either state it (as
  above) or hold out a same-dilution acquisition if one is exported.
- Spillover matrix reused from July; say so, or re-estimate from the FMO wells once exported.
