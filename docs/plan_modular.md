# Plan: a per-marker virtual-stain framework with modular label-free inputs

*2026-10-26. A proposal, not an implementation. It builds on what the repo has already
measured ([results.md](results.md)) and on the current trainer, and it names the decisions
that are yours before any code moves.*

## 1. What we are building

One trained model per fluorescence marker. Each model takes whatever label-free views the
acquiring instrument offers and emits one fixed output: that marker's fluorescence image
(from which a per-event intensity and an IDEAS-style mask follow). New markers are added
by registering them and training a new model; existing models are untouched.

| | fixed | modular |
|---|---|---|
| **output** | one channel, the marker's fluorescence image at input resolution; per-event intensity and a calibrated mask derived from it | |
| **input** | | any subset of the label-free *roles* the instrument has: `lightloss`/`brightfield`, `fsc`/`brightfield2`, `ssc`/`darkfield`. One model runs with one, two or three views. |
| **instrument** | | A8/S8 (CellView: three views), ImageStream (BF, BF2, SSC), CytPix (one camera brightfield). |

Three facts from the repo shape the design:

1. **Per-marker and joint models tie on the ISX** (joint vs per-marker ablation). So the cost
   of per-marker models is training time, not accuracy, and the benefit is that adding a
   marker never disturbs the others.
2. **Instruments do not mix in one model at our scale** (the generalist run). Per-instrument
   training, with optional warm-start from another instrument, is the safe default. The
   architecture below keeps a single code path for all instruments without forcing their
   data through one set of weights.
3. **The three label-free views carry different information** (A8 pages figure): light loss
   sees the dense head, FSC the outline, SSC the tail and internal structure. A model that
   must accept any subset has to learn each view's contribution separately, which is what
   the per-role stems below do.

## 2. Research: how others handle variable input channels

The question "same network, different channel sets" has a recent literature in microscopy,
all of it on classification/representation rather than image-to-image, but the input
mechanism transfers directly.

- **Per-channel tokenisation with learned channel embeddings.** ChannelViT (Bao et al.,
  2023) and ChAda-ViT (Bourriez et al., CVPR 2024) embed every channel independently with a
  shared projection, add a learned embedding that says *which* channel it is, and let the
  transformer attend across channels. Any number or order of channels works; missing
  channels are simply absent. ChAda-ViT was built for exactly this heterogeneity in
  microscopy. DiChaViT (Pham & Plummer, 2024) adds a diversity term so channels do not
  collapse onto the same features, and reports the best robustness when channels are
  missing at test time. The CHAMMI benchmark (Chen et al., NeurIPS 2023) is the standard
  test for this, with 3-, 4- and 5-channel sets.
- **Channel dropout during training** is the common thread: training on random subsets of
  channels is what makes a single model usable on any subset, and every paper above relies
  on it.
- **For image-to-image virtual staining** the field still trains fixed-channel U-Nets
  (Christiansen 2018; Ounkomol 2018; DeepIFC on the ImageStream with brightfield plus
  darkfield). None of them makes the input modular. The transfer is: keep the U-Net trunk
  that works, replace its first convolution with a channel-adaptive stem.

The convolutional analogue of the ViT trick is simple and is what I propose: **one small
stem per input role, summed into a common feature map, with a learned role embedding and
channel dropout.** It is order-invariant, tolerates missing roles, and adds nothing to the
trunk. A transformer trunk is not warranted: our objects are 50–150 px, the U-Net already
reaches r 0.97 on the ISX, and a ViT would need pretraining data we do not have.

## 3. Proposed architecture

```
                 role stems (one per label-free role, shared across instruments)
 lightloss/BF ──► Stem(1→w) ─┐
 fsc/BF2      ──► Stem(1→w) ─┼─► Σ (+ role embeddings, + instrument embedding) ─► U-Net trunk ─► Head(w→1) ─► stain image
 ssc/darkfield ─► Stem(1→w) ─┘        (absent roles contribute nothing)                    │
                                                                                            └─► event sum, calibrated mask
```

**Input adapter (`RoleStems`).** For each role present in the batch, a 2-conv stem maps
the single channel to `w` features; a learned per-role bias (the "embedding") is added; the
present roles are summed and divided by the count. During training each role is dropped
with probability p (never all), so the trunk learns to work from any subset. At inference
the caller passes a dict `{role: image}` with whatever it has. An instrument embedding is
added the same way, so an A8-trained model can later be warm-started on S8 data with the
instrument token as the only new parameter.

**Trunk.** The existing 4-level U-Net (`build_model`), unchanged: reflect padding, no global
pooling, native-size ragged batches, `--full-frame` loss. Width 32 on the laptop, 64 on the
server.

**Head.** One 1×1 convolution to one channel. Fixed. For markers that only exist as scalars
on an instrument (DAPI on the A8) an optional scalar head (global-sum pooling over the
predicted map, trained with the scalar label) sits beside it; it is the image-to-scalar
route from the original plan and costs nothing when unused.

**Post-processing, outside the network.** Per-event intensity = masked sum (as now). Mask =
threshold at a calibrated fraction of the object's own peak (`stain_masks.py`), stored with
the model so prediction is self-contained.

**Loss and selection.** Masked L1 on the scaled target with per-marker scale (as now);
selection on validation event r within cell type. Optional gradient-difference term and TTA
are kept as flags.

**What this is not.** Not a multi-instrument model: each marker model is trained per
instrument (or warm-started across instruments). Not a foundation model: a shared frozen
trunk with per-marker decoders is the phase-3 option if per-marker training becomes too
slow, and the ISX-pretrain result says it needs full-scale data before it pays.

## 4. Data model

Everything upstream of the trainer becomes "a set of event records":

```
EventRecord
  id            (instrument, sample, event index)
  inputs        {role: (h, w) float32}         roles present vary by instrument
  targets       {marker: (h, w) float32}        only the markers this acquisition stained
  scalars       {marker: float}                 per-event intensities when images are absent
  mask          (h, w) bool                     valid-pixel mask (object vs padding)
  meta          well type, replicate, population label, pixel pitch
```

Readers produce records (`a8`, `isx`, `cytpix`); one cache writer stores them ragged with
named arrays and an availability table; one loader serves `{role: tensor}` batches with
role dropout. This replaces the three prepare scripts, which each bake a different fixed
channel order into `x`.

Instrument definitions live in config, not code:

```toml
# configs/instruments/a8.toml
pixel_um = 0.42
roles = { lightloss = "page:0", fsc = "page:1", ssc = "page:2" }
stains = { "LDHC/AKAP4" = "page:3", "CD45" = "page:4", "ACRV1" = "page:5" }
unmix = "triangular"          # see a8_io.spillover

# configs/markers/acrv1.toml
name = "ACRV1"
colour = [1.0, 0.9, 0.2]
sources = { a8 = "ACRV1", isx = "Ch03" }
population = "sperm"          # the within-population score that selects the checkpoint
```

TOML reads with the standard library from Python 3.11; on the 3.9 laptop it needs the
`tomli` package (pure Python, one file). JSON is the fallback if you would rather add no
dependency.

## 5. Repository restructure

A package with one job per module, a thin CLI, configs, and the server scripts left as they
are (standard library, run under PowerShell, data copied by hand).

```
vstain/                     the package (name is yours to choose)
  io/        a8.py isx.py isx_daf.py cytpix.py fcs.py        readers -> EventRecords
  data/      records.py cache.py batching.py augment.py       ragged cache, role dropout
  models/    stems.py unet.py heads.py registry.py             RoleStems, trunk, heads, marker registry
  train/     loop.py losses.py select.py                       the training loop, L1/grad, selection
  eval/      metrics.py masks.py sheets.py figures.py          event r, IoU vs IDEAS, IDEAS-style sheets, slide/accuracy figures
  predict.py                                                   load a marker model, run on {role: image}
  cli.py                                                       prepare | train | evaluate | predict | figures
configs/
  instruments/  a8.toml isx.toml cytpix.toml
  markers/      ldhc_akap4.toml acrv1.toml dapi.toml tomm20.toml cd45.toml
  experiments/  laptop.toml server.toml
scripts/        unchanged: stdlib tools for the Windows server
docs/           unchanged, plus this plan and an ARCHITECTURE.md
tests/          synthetic-event tests: cache round-trip, role dropout, native-size eval
runs/ figures/ data/   ignored, as now
```

Migration map (what moves where; nothing is rewritten that already works):

| today | becomes |
|---|---|
| `analysis/a8_io.py`, `isx_io.py`, `isx_daf.py` | `vstain/io/` unchanged in substance |
| `analysis/prepare_isx.py`, `prepare_generalist.py`, `train_a8.py prepare` | `vstain/data/cache.py` + `cli prepare --instrument a8 --config ...` |
| `train_a8.py` (`Cached`, `RaggedCached`, `make_batches`, `augment`) | `vstain/data/` |
| `train_a8.py` (`build_model`) | `vstain/models/unet.py`; new `stems.py` replaces the first conv |
| `train_a8.py train` (loop, `evaluate`, selection) | `vstain/train/loop.py`, `vstain/eval/metrics.py` |
| `render.py`, `_sheet`, `make_sheet.py`, `slide_figure.py`, `accuracy_figures.py` | `vstain/eval/sheets.py`, `figures.py` |
| `stain_masks.py` | `vstain/eval/masks.py`; the calibrated threshold is saved into the model bundle |
| `eval_subset.py` | a `--population` flag on `cli evaluate` |
| `predict_cytpix.py` | `vstain/predict.py` with `--roles brightfield=...` |
| `explore_*.py`, `pilot_*.py`, `feature_baseline.py` | `analysis/archive/`, kept for the record |

A trained marker model is one directory: `weights.pt`, `marker.toml` (copy), `instrument`,
`roles_trained`, `scale`, `mask_threshold`, `results.json`. `cli predict` needs nothing else.

## 6. Experiments that validate the design (in order)

1. **Input ablation on the A8** — light loss only / light loss + FSC / all three, per
   marker, same held-out replicate. Decides how much the extra views are worth and whether a
   CytPix-class instrument is viable for which markers. Laptop, ~15 min per run.
2. **Role dropout vs fixed channels** — one model trained with dropout, evaluated on every
   subset, against the three fixed-subset models from (1). Decides whether one model per
   marker can serve every input configuration or whether we train one per configuration.
3. **Same on the ISX** (BF / BF+BF2 / BF+BF2+SSC). Server, since ISX runs are long.
4. **Warm start across instruments at full scale** — ISX-pretrained trunk, A8 fine-tune on
   the full 57,000-event folds. The laptop result was a null at 600 events; this is the
   test that counts. Server.
5. **Scalar head** — DAPI on the A8 from label-free images, scored against DAPI-A. Decides
   whether spectral-only markers are reachable.

## 7. What I need from you

Decisions:

1. **Marker list and priority** for the first registry, and which instrument each must be
   deployable on. (Current: LDHC/AKAP4, ACRV1, DAPI, TOMM20 on ISX; LDHC/AKAP4, ACRV1, CD45
   on A8; DAPI scalar-only on A8.)
2. **Deployment target.** S8 only, or also CytPix-class instruments? This sets whether the
   single-view configuration is a first-class target or a fallback.
3. **Package name** and whether a TOML dependency (`tomli` on Python 3.9) is acceptable, or
   JSON configs instead.
4. **Keep per-instrument training as the default** (my recommendation) or require one
   model per marker across instruments from the start.

Data:

5. **Full A8 run access** for the server experiments (already on `Z:`; nothing to copy).
6. **Any further ISX samples.** One tube exists; the current ISX split is first/second half
   of it, which is a within-sample estimate. A second tube would make the ISX numbers honest.
7. **Single-stain controls on the A8 imaging detectors** (AF488, PE, PerCP-eF710 alone) on
   the next acquisition. The spillover matrix is inferred from the data now; a control makes
   it a measurement.
8. **Whether the A8 DAPI-A scalar is a target you care about** (experiment 5).
9. **For S8 deployment later:** what the sort software can ingest (a per-event scalar, a
   mask, or an image), which fixes the predict-side output format.

Practice:

10. Confirm the working practice stays: development on the laptop sample, full runs on the
    server by PowerShell, server scripts standard-library only, run outputs never committed.

## 8. Phasing

- **Phase 1 (restructure, no behaviour change):** package layout, configs, cache from
  EventRecords, trainer reproduces today's numbers on `gen_cache`. Tests on synthetic events.
- **Phase 2 (modular input):** RoleStems + role dropout; experiments 1–3.
- **Phase 3 (scale and transfer):** server runs at width 64; experiment 4; scalar head.
- **Later:** shared frozen trunk with per-marker decoders if per-marker training becomes
  the bottleneck; the IDEAS/A8 adapter for writing virtual stains back into instrument files.

Sources: [ChAda-ViT (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Bourriez_ChAda-ViT__Channel_Adaptive_Attention_for_Joint_Representation_Learning_of_CVPR_2024_paper.pdf), [DiChaViT](https://arxiv.org/pdf/2405.16419), [C3R, channel-conditioned cell representations](https://arxiv.org/html/2505.18745), [DC-ViT (2026)](https://arxiv.org/html/2603.14281).
