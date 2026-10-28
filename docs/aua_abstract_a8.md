# AUA abstract draft: A8 virtual fluorescence (adapted from the submitted ImageStream abstract)

*Draft 2026-10-27 from `runs/srv_a8_stems` (server run, commit 7cfe9bc, epoch-20 checkpoint of the
fp16 run). The 40-epoch bfloat16 rerun now on the server will replace every number below; the
bracketed items need filling in. The A8 abstract's purpose is identification for sorting, not
morphology, so the objective, conclusions and impact statement are rewritten rather than adapted.*

## Title

Virtual Fluorescence for Image-Enabled Cytometry: Stain-Free Prediction of Leukocyte, Flagellar and Acrosomal Fluorescence from Light Loss and Scatter Images to Identify Sperm in Whole Ejaculate Samples

## Objective

Image-enabled cell sorters can now capture a label-free image of every object in a whole ejaculate sample and sort on it, at high speed and without touching the cell. Fluorescent antibody staining is what reliably tells a spermatozoon from a white blood cell or debris in such a sample, but staining is costly and slow, and it effectively destroys the cell, so a stained sperm cannot go on to Assisted Reproductive Technology (ART). To enable identification and sorting of viable sperm from whole ejaculate for intracytoplasmic sperm injection (ICSI), we investigate a machine learning model that predicts the fluorescent staining outcome of each cell from only the light loss and scatter images that the cytometer's imaging technology (CellView) produces.

## Materials and Methods

A remnant ejaculate sample from a high-volume male infertility practice and peripheral blood mononuclear cells (PBMCs) [source: fill in] were analyzed on a BD FACSDiscover A8 cell analyzer maintained by the BCM Flow Cytometry Core, in three wells: sperm alone, PBMCs alone, and a sperm–PBMC mixture, each acquired twice as independent replicates. The cells were fixed [Stocks fill in: methanol, acetone, PFA] and stained for Lactate Dehydrogenase C (LDHC), an enzyme found in high concentrations in the flagellum; Acrosomal Vesicle Protein 1 (ACRV1), a marker of the acrosome; and CD45, the leukocyte common antigen, carried by white blood cells and absent from sperm. For every event the A8's CellView imaging captured three label-free images, light loss (extinction), forward scatter and side scatter, co-registered with the three fluorescence images, with 20,000 events imaged per well. The three label-free images of each event were encoded as an image tensor and used as the input to a U-Net Convolutional Neural Network with a separate encoder for each image, trained on the 60,000 events of one replicate with the LDHC, ACRV1 and CD45 fluorescence images as the target response.

## Results

The final model was tested on the 60,000 events of the independent replicate, a separate acquisition with its own staining and instrument settings. Accuracy was scored on both the intensity and the location of the predicted fluorescence against the true staining result. For intensity, the Pearson coefficient r correlated the brightness of the predicted fluorescence with the measured result for each marker. Across all events, predicted stain intensity tracked the measured intensity at r = 0.74 for LDHC, 0.78 for CD45 and 0.66 for ACRV1; within a single cell population, where cell identity cannot help the model, r = 0.75 for LDHC among sperm, 0.60 for CD45 among PBMCs, and 0.32 for ACRV1 among sperm. Predicted CD45 was 6.9-fold higher on PBMC-well events than on sperm-well events (measured: 7.6-fold), and predicted ACRV1 was 16-fold higher on sperm-well events than on PBMC-well events (measured: 66-fold), so the predicted stains separate the two populations in the direction the real stains do. [Add: AUC for sperm vs PBMC on predicted CD45 alone; needs the checkpoint back from the server.] For location, an Intersection-over-Union (IoU) score described how closely the area covered by the predicted fluorescence matched the area of the measured stain: IoU = 0.67 (LDHC), 0.62 (CD45) and 0.43 (ACRV1), with 97–100% of stained events detected. Systematically restricting the inputs of the model revealed that light loss alone predicts LDHC and CD45 at r = 0.56 and 0.26, adding forward scatter raises them to 0.71 and 0.39, and adding side scatter to 0.74 and 0.59: the side-scatter image, which resolves the granular interior of a leukocyte, carries most of the CD45 signal.

## Conclusions

The fluorescent results of LDHC and CD45, and to a lesser degree ACRV1, can be predicted from the light loss and scatter images of an image-enabled cytometer alone, well enough to tell sperm from white blood cells without a stain. Because the FACSDiscover S8 sorter shares this imaging technology, the same prediction can serve as the sorting decision, laying the groundwork for isolating untouched, viable sperm from whole ejaculate samples for ICSI at high speed. The acrosome is predicted less well than on a higher-resolution imaging cytometer, so acrosomal status is not yet a sorting criterion. Validation across additional ejaculate samples is currently underway.

## Impact Statement

Predicting leukocyte and sperm immunofluorescence from the label-free images of an image-enabled cell sorter would allow viable, unstained sperm to be identified and sorted from whole ejaculate samples for downstream use in ICSI.

## Numbers to update before submission

- Every result above is from the epoch-20 checkpoint of the first full run (the fp16 run that went
  NaN at epoch 22). Replace them from the 40-epoch bfloat16 rerun when its zip arrives.
- Sperm-vs-PBMC AUC on predicted CD45 (and on predicted LDHC) is the number this abstract's
  purpose really rests on and has not been computed at full scale. It needs `best.pt` from the
  server (`server_stems.ps1 ... -Checkpoints`); then it is a few lines on the laptop.
- The AF488 channel is named LDHC/AKAP4 in the panel files; confirm with the Stocks lab whether
  the abstract should say LDHC alone (as the ImageStream abstract does) or LDHC/AKAP4.
- The A8 has no instrument-drawn masks, so the IoU reference is the thresholded measured stain,
  not an instrument mask as on the ImageStream; the results sentence is worded accordingly.
