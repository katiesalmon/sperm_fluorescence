# sperm_fluorescence

Predicting **per-event antigen-marker intensity from the brightfield CytPix event image**,
with the cytometer's own measurement supplying the training target. Nobody labels
anything: the target is measured at the same instant as the input.

Sibling repo: **`sperm_pbmc`**, which classifies brightfield events as straight sperm /
curled sperm / PBMC. Two of its findings constrain this work and are carried forward in
[docs/task_brief.md](docs/task_brief.md).

## Where this is up to

**The dataset is the BD FACSDiscover A8 run** — eight samples imaged label-free and in
three fluorescence channels, pixel-registered. In short:

- **120,000 stained paired events** (replicate 1 is the unstained control), each a
  six-page float32 TIFF: LightLoss, FSC, SSC, and AF488 / PE / PerCP-eF710 for
  LDHC+AKAP4, CD45 and ACRV1. DAPI is a scalar only.
- **The join is verified from the data**: filename index = FCS row, offset 0 at r ≈ 0.8.
- **The fluorescence pages are raw filter channels**; a physically-constrained
  triangular spillover (AF488→PE 0.202) unmixes them per pixel.
- **The pictures confirm the biology**: ACRV1 a compact spot on the sperm head, LDHC
  along the whole flagellum.
- The Attune CytPix archives were surveyed first and are now out of scope; their
  findings — replicate 1 unstained, the panel, the batch-effect lessons, a morphology
  floor — stay in the docs as history.

Full detail in [docs/results.md](docs/results.md) and [docs/task_brief.md](docs/task_brief.md).

## Quick start

Survey the archives on the server:

```bash
py -3 scripts\inspect_acs.py <run-folder>
```

Cut a bundle — matched images, targets and CellView features, with the join verified
from the data before anything is written:

```bash
py -3 scripts\make_a8_bundle.py <a8-run-folder> --per-class 200 --out a8_bundle_seed0.zip
```

Check it survived the copy:

```bash
python3 scripts/make_bundle.py --verify bundle_seed0.zip
```

Full detail, including the guards worth knowing about, is in
[docs/sampling.md](docs/sampling.md).

## Layout

| Path | What it is |
| --- | --- |
| `docs/task_brief.md` | The task, the marker biology, and what the data turned out to be |
| `docs/approach.md` | Prior art, per-marker expectations, evaluation design, and the sequence to run |
| `docs/results.md` | **Measured results**, newest first |
| `docs/paths.md` | Three costed paths forward, and which to run first |
| `docs/framework.md` | Design sketch for per-marker stain predictors, and the contract they share |
| `docs/sampling.md` | How to survey the archives and cut a working subset |
| `docs/server-setup.md` | What has to be installed where (for sampling: nothing) |
| `scripts/survey_tree.py` | Summarise an unfamiliar directory tree: files per kind per folder, sizes, names, TIFF headers |
| `scripts/inspect_acs.py` | Survey an `.acs` or a bare `.fcs`: `$TOT`, the parameter table, spillover |
| `scripts/make_a8_bundle.py` | **The A8 sampler.** Matched six-page images + unmixed targets + CellView features; verifies the row-index join from the data before writing |
| `scripts/make_bundle.py` | *CytPix, out of scope.* Images + targets + masks from the `.acs` |
| `scripts/make_targets.py` | Full per-event measurement table, including events with no image |
| `scripts/fcs_probe.py` | FCS TEXT segment: keywords, the `$PnN` → `$PnS` table, the spillover matrix |
| `scripts/fcs_data.py` | FCS DATA segment: the event matrix |
| `scripts/tiff_probe.py` | TIFF headers — pages, dtype, channels — without decoding pixels |
| `scripts/cytpix.py` | Shared helpers: class tokens, archive opening, sizes |
| `analysis/explore_targets.py` | Which tubes were stained, spillover, replicate disagreement |
| `analysis/feature_baseline.py` | Predict each marker from the instrument's morphology columns |
| `analysis/a8_io.py` | Load A8 six-page stacks and targets from a bundle; constrained imaging spillover + per-pixel unmix |
| `analysis/explore_masks.py` | Render events with the instrument's segmentation drawn on |
| `data/` | Archives, bundles, unpacked samples — **gitignored** |

Nothing under `data/` is ever committed.

## Conventions

- **Everything under `scripts/` is standard library Python 3.8+**, so it runs on the
  imaging server with no install. Keep it that way — that constraint is why the survey and
  the sampler can be run by whoever has access to the data rather than whoever has admin
  rights. `analysis/` is where numpy and friends are allowed.
- **Replicate is the held-out axis.** Train on 2, test on 3. Random event-level splits
  share an acquisition's gain, focus and noise floor; on LDHC that difference is 0.66
  against 0.38.
- **Compensate before using any target**, and keep every detector in `$SPILLOVER` —
  compensation spans all of them, so a four-column table is a dead end.
- **Compensated values go negative.** Use an `asinh`/biexponential transform, not `log1p`.
- **Never assume channel order.** It comes from the FCS `$PnN`/`$PnS` table.
- Code that runs over a bundle takes its data root as an argument, so the same code runs
  unchanged over a full archive.
