# AUA abstract draft: ImageStream virtual fluorescence

*Draft 2026-10-27 from `runs/srv_isx_stems` (server run, commit 7cfe9bc, corrected channel labels).
Bracketed items need filling in. The results section is written for the single sample we have;
if more samples are acquired before submission, the methods and results need their numbers.*

## Title options

1. Virtual Fluorescence for Imaging Flow Cytometry: Label-Free Prediction of Acrosomal, Mitochondrial, Flagellar and Nuclear Staining in Whole Ejaculate
2. Untouched Sperm Morphology: Deep Learning Predicts Four Fluorescent Compartment Stains from Brightfield and Darkfield Imaging Cytometry
3. Fluorescence Without the Stain: Compartment-Level Morphological Analysis of Unstained Whole-Ejaculate Samples by Virtual Fluorescence

## Abstract

**Introduction and Objective:** Imaging flow cytometry (Amnis ImageStream) records high-resolution images of every object in a whole ejaculate sample, and fluorescent antibody staining then delineates the sperm compartments (acrosome, mitochondrial midpiece, flagellum, nucleus) on which morphological analysis depends. Fixation and staining, however, are slow and costly, and they destroy the cells, so the same analysis cannot be applied to an untouched aliquot. At the Stocks Lab at Baylor College of Medicine we asked whether the label-free images alone carry enough information to predict the fluorescence images, so that fluorescence-grade compartment masks can be generated for unstained cells.

**Methods:** A whole ejaculate sample [from n = 1 patient; update if more are run] was stained for LDHC/AKAP4 (AF488, flagellum), TOMM20 (PE, mitochondrial midpiece), ACRV1 (PerCP-eF710, acrosome) and DAPI (nucleus), and 20,000 DAPI-positive objects were imaged on an ImageStream at 60× with two brightfield cameras and a darkfield (side-scatter) channel. A U-Net with a separate encoder for each label-free image was trained on the first 10,000 objects in acquisition order to predict the four fluorescence images from the three label-free images, and tested on the remaining 10,000. Accuracy was scored two ways. First, each predicted image was summed to a per-cell stain intensity and correlated (Pearson r) with the measured intensity among sperm only, so the score reflects how well the model reads each cell's stain, not how well it recognises a sperm. Second, predicted images were thresholded into compartment masks, as morphology software does with a real stain, and compared with the masks the instrument software (IDEAS) drew from the real stain, using intersection-over-union (IoU; 1 = identical).

**Results:** Among sperm, predicted stain intensity tracked the measured intensity closely: r = 0.98 for TOMM20, 0.96 for ACRV1, 0.93 for LDHC/AKAP4 and 0.77 for DAPI (nuclear stain varies little between sperm, which limits any correlation). Predicted masks overlapped the instrument's masks at IoU 0.73 (LDHC/AKAP4), 0.72 (DAPI), 0.63 (TOMM20) and 0.60 (ACRV1), and a mask was produced for 87–100% of stained cells. For comparison, thresholding the real stain the same way gave IoU 0.59–0.71 against the instrument's masks, so the predicted masks match the instrument as closely as the real stain itself does. Removing inputs showed that brightfield alone predicts the flagellum and midpiece (r = 0.89 and 0.95) but not the acrosome (r = 0.30); adding the darkfield image raised the acrosome to r = 0.96.

**Conclusions:** Fluorescence images of four sperm compartments can be predicted from label-free imaging-cytometry images with accuracy comparable to the stain itself, which enables compartment-level morphological analysis of every object in an unstained whole ejaculate. The acrosome requires side scatter, whereas the flagellum and midpiece are readable from brightfield alone, which bounds what brightfield-only instruments can recover. Validation across additional patients and semen phenotypes is underway.

## Figure

`python -m sperm_fluorescence panel runs/srv_isx_stems --cache data/isx_roles --test 3 --rows "best:2,typical:2,dim:1" --um-per-px 0.333 --out figures/aua_isx_panel.png`

One row per held-out object; columns are the three label-free inputs (brightfield 1, brightfield 2,
darkfield), then for each marker the measured stain beside the prediction. Rows: two sperm chosen as the
best cases by rule, two typical sperm, one DAPI-dim fragment. The command writes the legend to
`figures/aua_isx_panel.txt` and the chosen object ids to `figures/aua_isx_panel.json`.
`figures/aua_isx_panel_laptop.png` is the layout draft from the laptop model (8,000 objects, 12 epochs);
the submission version needs the server model (`server_stems.ps1 -SkipTrain -Checkpoints`, then the
command above on the laptop, or take `figures/srv_isx_panel.png` straight from the zip).
