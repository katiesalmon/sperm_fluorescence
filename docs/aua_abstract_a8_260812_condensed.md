# A8 2026-08-12 abstract (the V2 draft), condensed to the AUA limits (2026-11-01)

## Title (255)

Predicting fluorescent staining patterns from high-throughput imaging cytometry of whole ejaculate bridges resolution limits for improved image processing

## Introduction and objectives (400)

Imaging spectral cytometers such as the BD FACSDiscover A8 image every event in whole ejaculate and can sort cells in real time, but their images are too low in resolution to recognize sperm features. Fluorescent staining reveals these features but damages sperm for downstream use. We developed a model that predicts the staining of sperm and non-sperm events from label-free images.

## Methods (600)

Remnant normospermic ejaculate from a male infertility practice was imaged on the A8, then fixed and stained for lactate dehydrogenase C (LDHC; flagellum), TOMM20 (midpiece) and acrosomal vesicle protein 1 (ACRV1; acrosome). The light loss, forward scatter and side scatter images of each event were inputs to a U-Net convolutional neural network predicting the three stains. The model was trained on 19,000 imaged events, with 1,000 held out for validation, and tested on 20,000 unseen events from the same sample. Accuracy was scored on the intensity and location of predicted versus true stain.

## Results (600)

Predicted LDHC reproduced the measured flagellar stain closely: pixel-by-pixel correlation r=0.81, intersection over union (IoU) of predicted and true stain areas 0.71, and predicted fluorescence present in 100% of truly stained events. TOMM20 and ACRV1 tracked the location of true stain well (IoU 0.60 and 0.63; predicted in 98% of stained events) but intensity less closely (r=0.51 and 0.63). Eliminating inputs showed side scatter to be the most informative image: alone it predicted LDHC at r=0.75.

## Conclusions (400)

The U-Net reproduced the flagellar LDHC stain accurately in location and intensity; midpiece and acrosomal stains were located well, their intensity less so. The model also filters the random noise of the true stain, which it cannot reproduce, so correlation understates its accuracy. Bypassing fixation and staining, virtual fluorescence can augment raw cytometry images for morphologic evaluation and sorting.

## Impact statement

High-fidelity prediction of sperm fluorescent staining from low-resolution imaging of whole ejaculate will enable improved image processing and cell sorting.
