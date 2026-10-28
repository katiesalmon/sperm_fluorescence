# How sperm morphology is analysed today: a survey before we design ours

*2026-10-28. Read before changing `morphology_framework.md`. Primary sources are the WHO 6th
edition manual (text extracted from the published PDF), the papers linked at the end, and vendor
pages where nothing better exists. Where a source could not be read in full it is said so.*

## 1. The clinical reference method: WHO 6th edition (2021)

**Input.** An air-dried smear of ejaculate, fixed and stained (modified Papanicolaou is the
recommendation; Shorr and Diff-Quik are alternatives), examined in brightfield at ×1000 with oil
immersion. At least 200 spermatozoa; the manual states that to tell 3 % from 5 % normal forms in
one man, 1,500 must be assessed. Only intact sperm (head and tail) are scored; tailless heads are
tallied separately and reported if more than 20 per 100 sperm; overlapping and side-on sperm are
skipped; every assessable sperm in a field is scored to avoid selection bias.

**Classification.** Binary at the cell level, "ideal / typical" versus abnormal, where *all
borderline forms are abnormal*. The criteria (Table 2.6) by region:

| region | normal | abnormal when |
|---|---|---|
| head | smooth, regularly contoured, oval; acrosomal region 40–70 % of head area; no large vacuoles, at most two small ones under one fifth of the head; no post-acrosomal vacuoles | acrosome < 40 % or > 70 %; length/width < 1.5 (round) or > 2 (elongated); pyriform, amorphous, asymmetrical, non-oval apex; vacuoles > 1/5 of head or post-acrosomal; double heads |
| midpiece | slender, regular, about the head's length; major axis aligned with the head's | irregular, thin or thick, asymmetric or angled insertion, sharply bent |
| tail (principal piece) | uniform calibre, thinner than the midpiece, ≈ 45 µm (≈ 10 head lengths); may loop back if there is no sharp angulation | sharply angulated bends, smooth hairpin bends, coiled (> 360°), short/broken, irregular width, multiple |
| cytoplasmic residue | droplets under one third of a normal head are normal | residual cytoplasm over one third of a normal head |

Head *shape* is stated to matter more than exact size. Reference dimensions (Section 2.5.16,
77 normal Papanicolaou-stained sperm by a computerised system): length median 4.1 µm (95 % CI
3.7–4.7), width 2.8 (2.5–3.2), length/width 1.5 (1.3–1.8); midpiece length 4.0 µm (3.3–5.2),
**midpiece width 0.6 µm (0.5–0.7)**. Diploid large heads have about 1.6× the normal head area.

**Recording and outputs.** A multi-entry count: one key for ideal, one for abnormal, and one
each for the four regions, head (%H), neck/midpiece (%NM), tail (%T), excess residual cytoplasm
(%C), so one sperm can carry up to four defects. Reported: % ideal forms (whole number), the four
regional percentages, and the **teratozoospermia index, TZI = total defects / abnormal sperm**,
between 1.00 and 4.00, to two decimals. The reference limit is 4 % normal forms. MAI (mean
defects per abnormal sperm, droplets excluded, no cap) and SDI (defects per sperm counted) are
the related indices from the literature; reference data for MAI/TZI in fertile and infertile men
are tabulated in the manual (fertile TZI mean 1.51, infertile 1.81).

**A structured order per cell** (Fig. 2.11): is it a sperm → can it be classified (complete,
visible, not overlapped) → large cytoplasmic residue? → tail abnormal? → midpiece abnormal? →
head abnormal? → else ideal. Non-sperm objects (leukocytes, germ cells) are identified and
scored separately. This decision list is the thing an automated pipeline should reproduce.

**Two cautions in the manual that bear on us.** Cytoplasmic residues are osmotically sensitive
and not preserved by air-drying; they are visible on living cells in suspension by phase
contrast but appear only as small midpiece distensions on smears. And the manual's own view of
automated morphology (Section 4.5.3): image analysis has the potential for greater objectivity
and precision (reproducibility ≥ 92 %), but results are "easily biased due to variation in
staining", sperm heads are hard to tell from debris at low concentration, there is no way to
compensate for preparation artefacts, and "there is still not enough evidence that would allow
the use of computer analysis CASA in wide clinical practice". Table 7.15 lists the sources of
error in manual morphology: staining, smear thickness, scoring non-flat or overlapping sperm,
selecting sperm rather than scoring all, observer drift.

## 2. Computer-aided sperm morphometric assessment (CASMA) and CASA morphology modules

**Inputs.** The same stained smears (Papanicolaou, Diff-Quik, Shorr; SpermBlue and Hoechst for
fluorescence head morphometry), ×60–×100 brightfield or fluorescence, automated stage and
segmentation of the head by thresholding.

**Outputs, the de facto standard parameter set** (CASA-Morph / Sperm Class Analyzer, as
documented in Yániz et al.): head **area** (µm²), **perimeter** (µm), **length** and **width**
(largest and smallest Feret diameters over 0°, 30°, …, 150°), and four derived shape indices,
**ellipticity** L/W, **rugosity** 4πA/P², **elongation** (L − W)/(L + W), **regularity**
πLW/4A. Vendor modules add midpiece width, area and insertion angle, acrosome area and
percentage, vacuole number and area, and the indices TZI, SDI and MAI. Hamilton Thorne's
Dimensions module hard-codes the Tygerberg strict criteria. Formulas for "elongation" and
"ellipticity" differ between systems and papers, so values are not comparable across systems
without checking definitions.

**How well it agrees with manual scoring: poorly, for morphology specifically.** A systematic
review of CASA validity found morphology the parameter with the largest automated-versus-manual
discrepancy, attributed to the heterogeneity of sperm shape; a 195-man study found poor
correlation and automated values far below manual; intraclass correlations of 0.16–0.26 against
manual reads have been reported for two analysers. The WHO manual cites two studies where CASMA
normal-form outcomes did predict fertility end-points (Coetzee; Garrett's "zona-preferred"
head shape %Z with VSL), i.e. the automated measure can be prognostic even when it does not
reproduce the manual number. *Implication: "agreement with manual % normal" is a weak target
on its own, because the manual number itself is noisy; cell-level agreement on measurements and
patient-level prognosis are the better validations.*

**Subpopulation analysis is established practice in morphometry.** The standard pipeline is
PCA on the morphometric parameters (Kaiser criterion, typically three components explaining
≈ 88 % of variance) followed by clustering (two-step or k-means) into subpopulations; in
normozoospermic men this reliably yields three head subpopulations, roughly large-round,
small-round and elongated, each about a third of cells, whose proportions differ between men
and between ejaculate fractions. Elliptic Fourier descriptors of the head contour with PCA are
the other common shape representation (bull, stallion, ram, mouse work; related to fertility in
several species).

## 3. Deep learning on stained microscopy

**Datasets.** HuSHeM (216 head crops, 131 × 131 px, four classes: normal, tapered, pyriform,
amorphous), SCIAN-MorphoSpermGS (≈ 1,100 head crops, five classes incl. small, labelled by
three experts, with partial agreement), MHSMA (1,540 crops of 128 × 128 or 64 × 64 px from 235
infertile men, four binary attributes: acrosome, head, vacuole, tail/neck; tail label 95 %
positive), SMIDS. All are **head crops of stained cells at low resolution with
categorical labels**, which is why most published models are head-shape classifiers reporting
accuracy on four or five classes. Reviews note the shortage of annotated data and large
accuracy differences between datasets.

**Segmentation-and-measurement work is newer and closer to what we want.** An instance-aware
part-segmentation network (2024) segments acrosome, vacuole, nucleus, midpiece and tail on
stained images (320 images, ≈ 5–6 sperm each) and derives head length/width/ellipticity/area by
ellipse fit, acrosome/nucleus/vacuole areas and counts, head–midpiece angle, midpiece
length/width/angle by rectangle fit, and tail length/width/curvature from a centreline, with
tail measurement accuracy 95/96/91 % against manual ImageJ measurement of 50 sperm. A
stain-free follow-up on live sperm at 20× (Sensors 2025) reaches only 10–20 % relative error
and reports blurred boundaries as the limiting factor. Neither maps its measurements onto WHO
criteria explicitly.

**Stain-free selection for ICSI is being validated with the same measurements.** The QART
quantitative-phase study (2023, 73 subjects, 1,451 cells, 326 in the agreement analysis)
applied WHO 2021 head criteria (length 3.7–4.7 µm, width 2.5–3.2, L/W 1.3–1.8, acrosome 40–70 %
of head area) to stain-free phase maps and validated by **Bland–Altman against the same cells
after staining**, with acceptance limits derived from the reference method's own repeatability.
Mean differences were 0.18 µm (length), −0.26 µm (width), 0.17 (ratio), 5.7 points (acrosome
fraction, with a wide ±18-point interval). Only ~19–25 % of embryologist-selected "normal"
cells were WHO-compliant by either method. *This is the validation design to copy: same cells,
stain-free versus stained, Bland–Altman with limits from the reference's repeatability.*

## 4. Imaging flow cytometry (ImageStream) and sperm

This is the closest literature and it is thin for human sperm.

- **Human, stained:** Matamoros-Volante et al. (2018) built a semi-automatic IDEAS workflow to
  segment head and flagellum of human sperm for capacitation-induced tyrosine phosphorylation,
  seven donors; later authors note its mask "may include segments where the flagellum is
  out-of-focus". Full text could not be retrieved here (PMC blocked), so the mask functions and
  the focus gate are unverified; by convention the IDEAS focus gate is Gradient RMS on the
  brightfield channel and single cells are gated on Area versus Aspect Ratio.
- **Mouse, stained (2023):** ResNet50 to reject out-of-focus images (F1 0.99 on 604 labelled
  images), then U-Net on brightfield + membrane-stain channels to segment head and midpiece
  (Dice ≈ 0.81, 294 hand-traced images); 60×, NA 0.9. No morphology classes, no comparison with
  conventional methods.
- **Bovine, label-free brightfield (2025 preprint, 2026 Frontiers):** ImageStream X Mk II at
  **40×**, 1.8 M events → 436,374 single, focused, fully-in-frame cells by IDEAS gating; two
  researchers labelled 200 images per class per breed per condition into eight morphology
  classes (normal, irregular head, elongated head, abnormal midpiece, abnormal tail, proximal
  droplet, distal droplet, coiled tail+midpiece) plus "multiple" and "debris"; MobileNetV4
  classifier, 92.3 % accuracy, macro-F1 0.91; confusion mainly coiled-tail vs abnormal-tail and
  elongated vs irregular head. **Generalisation dropped 6–8 % between fresh and frozen and up to
  ~12 % across breeds**, which the authors call domain shift. Normal-form percentages (34–60 %)
  were lower than microscopy's ~70 %, attributed to IFC seeing rare tail defects and droplets
  that microscopy misses.
- **Boar, label-free (Keller et al., Andrology 2025):** ImageStream at 20×/40×/60×, 10 boars,
  2,000 labelled images per class for five classes (normal, proximal droplet, distal droplet,
  distal midpiece reflex, coiled tail), classified with the **vendor's Amnis AI module** (VGG16
  at 350 × 350): weighted F1 96.7 / 98.6 / 99.3 % at 20/40/60×. **Label-free acrosome status**
  (ground truth PNA lectin stain) was learnable only at 60× (F1 99.8 %) and not at 20× or 40×
  (64–68 %). No comparison with stained slides; "better than human" rests on an assumed 95 %
  technician benchmark.
- **Stallion (Bulkeley 2023):** IDEAS used to classify abnormal head, midpiece, tail, proximal
  and distal droplets with viability and ROS co-stains (paywalled; abstract only).
- **IDEAS features:** per-mask Area, Length, Width, Aspect Ratio, Circularity, Elongatedness,
  Shape Ratio, Major/Minor axis and intensity features; a patent describes a "bending" feature
  as the maximum curvature along a skeleton mask, used to count distal midpiece reflex and
  curved tails, and another describes combining brightfield and fluorescence masks for
  per-compartment morphology, with shape descriptors invariant to scale, position and
  orientation (diameter, circularity, elongation, corners, negative curvatures).

Every IFC morphology paper gates the same way before analysis: in focus, single cell, whole
cell inside the frame, debris and multiples labelled as their own classes.

## 5. Virtual staining of sperm

HoloStain (Tel Aviv) converts holographic phase images of single sperm into virtual
histochemical stains (Papanicolaou-like) with GANs; the stated motive is that staining is
prohibited in clinical IVF. A related patent names nucleus, acrosome, mitochondria, vacuole and
cytoplasm as target organelles. No published work was found that predicts
**antibody fluorescence** of sperm compartments from label-free images, on an imaging
cytometer or otherwise; virtual mitochondrial labelling exists for cultured cells (brightfield →
mitochondria at r ≈ 0.86). Our approach appears to be unreported for sperm, which cuts both
ways: nothing to copy, and no accepted validation convention beyond the QPI one above.

## 6. What this changes in our framework

1. **Report in the WHO structure, not a private one.** Per sperm: ideal / abnormal plus the
   four region flags; per sample: % ideal, %H, %NM, %T, %C, TZI (1.00–4.00), optionally MAI
   and SDI, free heads and free tails tallied separately, non-sperm objects counted separately.
   Follow the Fig. 2.11 decision order. Report sample size and the 200 / 1,500 thresholds.
2. **Measure the standard morphometric set** so our numbers can be set against CASMA and
   WHO reference values: head area, perimeter, Feret length and width, L/W, rugosity,
   elongation, regularity; acrosome area fraction; midpiece length, width, insertion angle;
   tail length, width profile, curvature. Define each formula in the config and in the feature
   table header; the literature's own formulas are inconsistent.
3. **Midpiece width is 0.6 µm, under 2 px at 60×.** "Thin / thick midpiece" cannot be a width
   measurement here; use TOMM20 mask area per unit length and brightfield darkness as proxies,
   and say so. Head width thresholds (2.5–3.2 µm) are 7.5–9.6 px: sub-pixel ellipse fits and
   a calibration against IDEAS Length/Width on the same cells are mandatory.
4. **Keep 60×.** The boar result (acrosome only learnable at 60×) matches our ablation (ACRV1
   needs darkfield at 60×). The bovine group chose 40× to fit whole cells; the trade-off is
   real and should be measured on one sample at both magnifications, but acrosome criteria
   require 60×.
5. **Add a categorical classifier beside the rules.** The field's working IFC solution is a
   CNN over technician-labelled brightfield crops into 5–8 classes (bovine, boar), including
   "multiple" and "debris" classes, with ~200–2,000 labels per class. Our rule engine on
   predicted masks is the interpretable arm; a classifier on the same crops (or on the U-Net
   bottleneck) is the arm the literature has validated, and the two can cross-check each other.
   The vendor's Amnis AI module is an off-the-shelf baseline worth running once.
6. **Validate the way QART did, not by % normal alone.** Same cells, predicted-mask
   measurements versus measured-stain and IDEAS measurements, Bland–Altman with acceptance
   limits from the reference's repeatability. Then patient-level agreement with manual
   morphology, judged against the known inter-method noise (CASMA versus manual is poor), and
   ultimately prognosis. Expect, as QART found, that most "normal-looking" sperm fail strict
   criteria by any measurement.
7. **Expect and test domain shift.** Fresh versus frozen cost the bovine model 6–8 points;
   fixed-and-stained versus unstained will cost us something. One unstained acquisition of the
   same sample is the test; the bovine authors' fix was more, broader training data.
8. **Cytoplasmic residue may be a strength of flow.** WHO notes residues are lost on smears and
   visible on cells in suspension; the IFC veterinary work scores proximal and distal droplets
   routinely. Our brightfield plus TOMM20 midpiece mask can score it; keep the WHO one-third-
   of-head threshold and the physiological-versus-excess distinction.
9. **Subpopulations: use the established recipe first.** PCA on the morphometric set, cluster,
   expect three head subpopulations and compare their proportions across men. Tail-shape
   clustering (curvature profiles, Fourier descriptors of the skeleton) has no precedent and is
   the genuinely new piece; present it as exploratory.
10. **Gate like IDEAS.** Focus (brightfield gradient RMS), single cell, whole cell in frame,
    debris and multiples as explicit classes, side-on and overlapping excluded, and report the
    fraction removed at each gate. The bovine paper kept 24 % of events after gating.

## Sources

- WHO laboratory manual for the examination and processing of human semen, 6th ed. (2021): Sections 2.4.9, 2.5.16, 4.5.3, Table 2.6, Fig. 2.11, Table 3.2, Table 7.15. https://mes-global.com/wp-content/uploads/2023/09/WHO-6th-Edition-Manual-for-Semen-Analysis.pdf
- Björndahl & Kirkman Brown (2022) on the 6th edition, Fertility and Sterility. https://www.fertstert.org/article/S0015-0282(21)02298-6/fulltext
- Yániz et al., morphometric subpopulations, CASA-Morph parameter definitions (PMC5109871, via Europe PMC); Yániz/Soler 2016 Asian J Androl (21 men, three subpopulations). https://zaguan.unizar.es/record/57770
- SCA Morphology module (Microptic). https://www.micropticsl.com/products/sperm-class-analyzer-casa-system/main-modules/sca-morphology/
- Dimensions II (Hamilton Thorne), Tygerberg strict criteria hard-coded. https://planer.com/products/hamilton-thorne/casa-products/dimensions-ii-software-for-the-ivos-ii-and-ceros-ii-systems.html
- Systematic review of CASA validity and reliability (morphology the weakest). https://repository.uwc.ac.za/handle/10566/6578
- Instance-aware part segmentation and tail measurement (arXiv 2408.00112). https://arxiv.org/html/2408.00112v1
- Stain-free sperm morphology measurement at 20× (Sensors 2025). https://pmc.ncbi.nlm.nih.gov/articles/PMC11821022/
- QART quantitative-phase stain-free selection vs WHO 2021 (2023). https://pmc.ncbi.nlm.nih.gov/articles/PMC10604130/
- WHO 2021 head reference values and a 14-parameter head morphometry in fertile men (2024). https://pmc.ncbi.nlm.nih.gov/articles/PMC11156450
- Deep learning datasets and reviews: HuSHeM/SCIAN/MHSMA/SMIDS. https://pmc.ncbi.nlm.nih.gov/articles/PMC12538767/ , https://www.aimspress.com/aimspress-data/math/2023/7/PDF/math-08-07-838.pdf , https://data.mendeley.com/datasets/hjygss6sw2
- Bovine label-free IFC morphology with deep learning (bioRxiv 2025.06.12.659335; Frontiers Vet Sci 2026). https://www.biorxiv.org/content/10.1101/2025.06.12.659335v1.full.pdf
- Keller et al., boar morphology and label-free acrosome health on ImageStream with Amnis AI (Andrology, doi 10.1111/andr.13758), as summarised at https://www.swineweb.com/deep-learning-classification-method-for-boar-sperm-morphology-analysis/
- Mouse sperm IFC: ResNet50 focus filter + U-Net head/midpiece (2023). https://www.scielo.org.mx/scielo.php?script=sci_arttext&pid=S1405-55462023000401133&lng=pt&nrm=iso&tlng=en
- Matamoros-Volante et al. 2018, human sperm IFC segmentation (Mol Hum Reprod 24:64). https://pmc.ncbi.nlm.nih.gov/articles/PMC6454810 (not retrievable here)
- Bulkeley et al. 2023, stallion IFC morphology and ROS. https://onlinelibrary.wiley.com/doi/10.1111/rda.14241 (paywalled)
- IFC morphology patents (brightfield + fluorescence masks; skeleton bending feature). https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11972620 , https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10846509
- HoloStain virtual staining of sperm. https://arxiv.org/pdf/1909.11374
- Unsupervised flagellar beat classification (3D+t). https://pubmed.ncbi.nlm.nih.gov/36085948/
- TZI/MAI/SDI definitions (secondary). https://www.invitra.com/en/teratozoospermia-diagnosis
