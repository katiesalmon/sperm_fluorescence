# AUA abstract draft: A8 virtual fluorescence (adapted from the submitted ImageStream abstract)

*Draft 2026-10-27, numbers from the 40-epoch bfloat16 run `runs/srv_a8_stems`
(`server_results_20261027_1941_a8.zip`, commit db78bce, best epoch 37). Bracketed items need filling in. The A8 abstract's purpose is identification for sorting, not
morphology, so the objective, conclusions and impact statement are rewritten rather than adapted.*

## Title

Virtual Fluorescence for Image-Enabled Cytometry: Stain-Free Prediction of Leukocyte, Flagellar and Acrosomal Fluorescence from Light Loss and Scatter Images to Identify Sperm in Whole Ejaculate Samples

## Objective

Image-enabled cell sorters can now capture a label-free image of every object in a whole ejaculate sample and sort on it, at high speed and without touching the cell. Fluorescent antibody staining is what reliably tells a spermatozoon from a white blood cell or debris in such a sample, but staining is costly and slow, and it effectively destroys the cell, so a stained sperm cannot go on to Assisted Reproductive Technology (ART). To enable identification and sorting of viable sperm from whole ejaculate for intracytoplasmic sperm injection (ICSI), we investigate a machine learning model that predicts the fluorescent staining outcome of each cell from only the light loss and scatter images that the cytometer's imaging technology (CellView) produces.

## Materials and Methods

A remnant ejaculate sample from a high-volume male infertility practice and peripheral blood mononuclear cells (PBMCs) [source: fill in] were analyzed on a BD FACSDiscover A8 cell analyzer maintained by the BCM Flow Cytometry Core, in three wells: sperm alone, PBMCs alone, and a sperm–PBMC mixture, each acquired twice as independent replicates. The cells were fixed [Stocks fill in: methanol, acetone, PFA] and stained for Lactate Dehydrogenase C (LDHC), an enzyme found in high concentrations in the flagellum; Acrosomal Vesicle Protein 1 (ACRV1), a marker of the acrosome; and CD45, the leukocyte common antigen, carried by white blood cells and absent from sperm. For every event the A8's CellView imaging captured three label-free images, light loss (extinction), forward scatter and side scatter, co-registered with the three fluorescence images, with 20,000 events imaged per well. The three label-free images of each event were encoded as an image tensor and used as the input to a U-Net Convolutional Neural Network with a separate encoder for each image, trained on the 60,000 events of one replicate with the LDHC, ACRV1 and CD45 fluorescence images as the target response.

## Results

The final model was tested on the 60,000 events of the independent replicate, a separate acquisition with its own staining and instrument settings. Accuracy was scored on both the intensity and the location of the predicted fluorescence against the true staining result. For intensity, the Pearson coefficient r correlated the brightness of the predicted fluorescence with the measured result for each marker. Across all events, predicted stain intensity tracked the measured intensity at r = 0.75 for LDHC, 0.78 for CD45 and 0.66 for ACRV1; within a single cell population, where cell identity cannot help the model, r = 0.75 for LDHC among sperm, 0.61 for CD45 among PBMCs, and 0.37 for ACRV1 among sperm. Predicted CD45 alone separated PBMC-well from sperm-well events at AUC = 0.95 (measured CD45: 0.99), and predicted ACRV1 at AUC = 0.97 (measured: 0.91) [on 600 held-out events; rerun at full scale before submission]. For location, an Intersection-over-Union (IoU) score described how closely the area covered by the predicted fluorescence matched the area of the measured stain: IoU = 0.68 (LDHC), 0.62 (CD45) and 0.42 (ACRV1), with 97–100% of stained events detected. Systematically restricting the inputs of the model revealed that light loss alone predicts LDHC and CD45 at r = 0.59 and 0.35, adding forward scatter raises them to 0.74 and 0.44, and adding side scatter to 0.74 and 0.60: the side-scatter image, which resolves the granular interior of a leukocyte, carries most of the CD45 signal.

## Conclusions

The fluorescent results of LDHC and CD45, and to a lesser degree ACRV1, can be predicted from the light loss and scatter images of an image-enabled cytometer alone, well enough to tell sperm from white blood cells without a stain. Because the FACSDiscover S8 sorter shares this imaging technology, the same prediction can serve as the sorting decision, laying the groundwork for isolating untouched, viable sperm from whole ejaculate samples for ICSI at high speed. The acrosome is predicted less well than on a higher-resolution imaging cytometer, so acrosomal status is not yet a sorting criterion. Validation across additional ejaculate samples is currently underway.

## Impact Statement

Predicting leukocyte and sperm immunofluorescence from the label-free images of an image-enabled cell sorter would allow viable, unstained sperm to be identified and sorted from whole ejaculate samples for downstream use in ICSI.

## Numbers to update before submission

- The sperm-vs-PBMC AUCs were scored with the server checkpoint on the laptop's 600-event held-out
  sample (200 per well). Rerun at full scale (60,000 events) on the server before submission.
- The repo does not record how many donors contributed the ejaculate, or their semen parameters;
  the methods sentence says "a remnant ejaculate sample" and needs the Stocks lab to confirm.
- The AF488 channel is named LDHC/AKAP4 in the panel files (FCS label `LDHC_AKAP4-A`); confirm
  whether the abstract should say LDHC alone (as the ImageStream abstract does) or LDHC/AKAP4.
- The A8 has no instrument-drawn masks, so the IoU reference is the thresholded measured stain,
  not an instrument mask as on the ImageStream; the results sentence is worded accordingly.

## Figure

Drawn on the laptop with the server checkpoint (`runs/srv_a8_stems/best.pt`, from the `-Checkpoints`
zip) on the local 600-event held-out sample:

`python -m sperm_fluorescence panel runs/srv_a8_stems --cache data/a8_roles --test 3 --rows "best:5" --stained "LDHC/AKAP4,ACRV1" --size 56 --um-per-px 0.333 --approx-scale --range sample --sample 600 --tta --out figures/aua_a8_panel.png`

`figures/aua_a8_panel.png`: five sperm-well objects chosen by the best-case rule, as in the ImageStream
figure. `figures/aua_a8_panel_with_pbmc.png` (`--rows "best:4,P:1"`): four sperm and one PBMC-well
object, which shows the leukocyte marker lighting up on a white blood cell and staying dark on sperm,
the point of this abstract. `--range sample` scales each marker by the brightest quarter of the sample,
so CD45 is scaled by leukocytes; `--stained` leaves CD45 out of the "well stained" rule because sperm
do not carry it. Legends are written beside each PNG as `.txt`.
