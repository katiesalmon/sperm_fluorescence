# sperm_fluorescence

Predicting **per-event antigen-marker intensity from the brightfield CytPix event image**,
with the cytometer's own measurement supplying the training target. Nobody labels
anything: the target is measured at the same instant as the input.

Sibling repo: **`sperm_pbmc`**, which classifies brightfield events as straight sperm /
curled sperm / PBMC. Two of its findings constrain this work and are carried forward in
[docs/task_brief.md](docs/task_brief.md).

## Where this is up to

The data is understood and the first result is in. In short:

- **The images and the measurements are both inside the nine `.acs` archives**, not in the
  `Images\*.zip` export. The archives hold 248 x 248 **uint16** images, the full FCS, and
  the instrument's own per-event segmentation masks.
- **The panel is DAPI, ACRV1, LDHC+AKAP4 (pooled on one fluor), and CD45.** There is no
  Tomm20 anywhere in the experiment.
- **Replicate 1 is the unstained control** — DAPI only. Two stained replicates, 192,874
  paired events, and a held-out-replicate split with exactly two folds.
- **The exported values are uncompensated.** Applying `inv($SPILLOVER)` drops ACRV1 on
  CD45-high round cells from 10,608 to 91, which is the biologically correct answer.
- **Morphology alone predicts DAPI at ρ 0.78 and CD45 at 0.77** across replicates; ACRV1
  barely clears its null. That is the floor a network has to beat.

Full detail in [docs/results.md](docs/results.md) and [docs/task_brief.md](docs/task_brief.md).

## Quick start

Survey the archives on the server:

```bash
py -3 scripts\inspect_acs.py <run-folder>
```

Cut a bundle from the A8 run — the primary dataset — with the join verified from the data:

```bash
py -3 scripts\make_a8_bundle.py <a8-run-folder> --per-class 200 --out a8_bundle_seed0.zip
```

Or from the CytPix archives — images, scalar targets, masks and compensation matrices:

```bash
py -3 scripts\make_bundle.py <cytpix-run-folder> --per-class 200 --out bundle_seed0.zip
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
| `scripts/inspect_acs.py` | Survey an `.acs`: members, `$TOT`, the detector-to-antigen table, voltages, spillover |
| `scripts/make_a8_bundle.py` | **The A8 sampler.** Matched six-page images + unmixed targets + CellView features; verifies the row-index join from the data before writing |
| `scripts/make_bundle.py` | The CytPix sampler. Images + targets + masks + compensation matrices from the `.acs` |
| `scripts/make_targets.py` | Full per-event measurement table, including events with no image |
| `scripts/fcs_probe.py` | FCS TEXT segment: keywords, the `$PnN` → `$PnS` table, the spillover matrix |
| `scripts/fcs_data.py` | FCS DATA segment: the event matrix |
| `scripts/tiff_probe.py` | TIFF headers — pages, dtype, channels — without decoding pixels |
| `scripts/cytpix.py` | Shared helpers: class tokens, archive opening, sizes |
| `analysis/explore_targets.py` | Which tubes were stained, spillover, replicate disagreement |
| `analysis/feature_baseline.py` | Predict each marker from the instrument's morphology columns |
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
