# A8 2026-08-12 abstract, condensed to the AUA limits (2026-11-01)

## Title (255)

Virtual fluorescence from label-free imaging cytometry of whole ejaculate resolves sperm head, midpiece and tail structure without staining

## Introduction and objectives (400)

Imaging spectral cytometers such as the BD FACSDiscover A8 image every event in whole ejaculate and can sort cells in real time, but their label-free images are too low in resolution to recognize sperm features, and fluorescent staining damages sperm for downstream use. We trained a neural network to predict fluorescent staining from label-free images so sperm can be characterized unstained.

## Methods (600)

A remnant normospermic ejaculate was imaged on the A8, then fixed and stained for lactate dehydrogenase C (LDHC, flagellum), TOMM20 (midpiece) and acrosomal vesicle protein 1 (ACRV1, acrosome). A U-Net neural network predicted the three fluorescence images from each event's light loss, forward scatter and side scatter images. It was trained on 19,000 imaged events, with 1,000 held out for validation, and tested on 20,000 events from a separate acquisition of the same sample. Accuracy was scored by pixel-wise correlation (r), intersection over union (IoU) of stain areas, and detection rate.

## Results (600)

Predicted LDHC matched the measured flagellar stain at r=0.81 and IoU 0.71, with the stain detected in 100% of stained events. TOMM20 and ACRV1 were placed correctly (IoU 0.60 and 0.63; detected in 98%) with lower intensity agreement (r=0.51 and 0.63). Removing inputs showed that side scatter carries most of the information: alone it predicted LDHC at r=0.75, against 0.81 for all three images, and light loss alone reached 0.65.

## Conclusions (400)

A U-Net trained on under 20,000 events reproduced the flagellar stain with high fidelity and located the midpiece and acrosome in nearly every cell, recovering head, midpiece and tail structure from low-resolution label-free images without fixation or stain. Virtual fluorescence lets the A8 recognize sperm and their morphology in real time, a step toward sorting unstained sperm from whole ejaculate.
