# Server setup

## Who needs to install what

| Task | Needs |
| --- | --- |
| **Surveying the zips and building a sample** (`scripts/`) | Python 3.8+ and nothing else |
| **Working with the images** (`analysis/`, modeling) | Python 3.8+ **and** `requirements.txt` |

Everything under `scripts/` is standard library only, on purpose. The survey is the first
thing that has to run on the imaging server, and it should not be blocked behind a pip
install on a machine that may not reach the internet. That includes the TIFF header
reader (`tiff_probe.py`) and the FCS parser (`fcs_probe.py`) — both written against the
file formats directly rather than against `tifffile` / `fcsparser`.

So: to answer "what is in these zips" and "give me 300 events", a bare Python is enough.

## Surveying and sampling — no setup

From the cloned repo folder on the server:

```powershell
py -3 scripts\inspect_acs.py <run-folder>
```

```powershell
py -3 scripts\make_bundle.py <run-folder> --per-class 200 --out bundle_seed0.zip
```

Then copy `bundle_seed0.zip` back to the laptop.

Optionally point the tooling at the data once per machine, the same way `sperm_pbmc` does:

```powershell
Set-Content -Path images_path.txt -Value 'Z:\<share>\<year>\<run-folder>\Images' -Encoding utf8
```

Use `Set-Content -Encoding utf8` or Notepad — **not** `echo path > file`, which in Windows
PowerShell writes UTF-16 and puts a byte-order mark on the front of the path.

## The analysis environment, when you get there

```powershell
py -3 -m venv .venv
```

```powershell
.venv\Scripts\python -m pip install --upgrade pip
```

```powershell
.venv\Scripts\python -m pip install -r requirements.txt
```

Roughly 100–200 MB of wheels — numpy, scipy, scikit-image and scikit-learn are the large
ones. Nothing compiles; these all ship as prebuilt wheels for Windows.

Check it took:

```powershell
.venv\Scripts\python -c "import numpy, scipy, pandas, skimage, sklearn, tifffile, imagecodecs, PIL; print('ok')"
```

`imagecodecs` is not optional if the marker TIFs are LZW like replicate 1's —
`tifffile` cannot decode them without it, and the failure looks like a codec error rather
than a missing package.

## The GPU machine (RTX A6000, 48 GB, Windows, CUDA 13.2 driver)

Training runs on the same machine that reads `Z:`. Install once, into the repo's venv:

```powershell
py -3 -m venv .venv
```

```powershell
.venv\Scripts\python -m pip install --upgrade pip
```

```powershell
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
```

```powershell
.venv\Scripts\python -m pip install numpy tifffile matplotlib scipy scikit-learn
```

The cu124 wheels run on any driver reporting CUDA 12.4 or newer, which 13.2 does. Check:

```powershell
.venv\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Then `analysis/train_a8.py prepare` once (reads every image off the share, writes a
~16 GB cache to a **local** disk — not `Z:` — so training never touches the network),
and `train` per fold. Both are documented in the script's header.

## The stems models at full scale (`scripts\server_stems.ps1`)

One script runs the whole chain for both instruments: cache from the run folder / the
`.cif`, train the modular (stems, role-dropout) model, then `ablate`, `views`, `sheet`,
`masks`, `accuracy` and `slides`. Once, after the torch install above:

```powershell
.venv\Scripts\python -m pip install -r requirements.txt
```

```powershell
.venv\Scripts\python -c "import torch, skimage, tifffile, matplotlib; import tomllib" 2>$null; if ($LASTEXITCODE) { .venv\Scripts\python -m pip install tomli }
```

A short smoke run first (2,000 events per class, ~20 minutes), then the full one:

```powershell
.\scripts\server_stems.ps1 -A8Run "Z:\Blair_Main\2026\260709_Blair_Sperm_A8\260709" -IsxCif "Z:\<isx-folder>\ALL_1_400_1.cif" -IsxDaf "Z:\<isx-folder>\ALL_1_400_1.daf" -Cache C:\sf_cache -Limit 2000 -A8Epochs 5 -IsxEpochs 5
```

```powershell
.\scripts\server_stems.ps1 -A8Run "Z:\Blair_Main\2026\260709_Blair_Sperm_A8\260709" -IsxCif "Z:\<isx-folder>\ALL_1_400_1.cif" -IsxDaf "Z:\<isx-folder>\ALL_1_400_1.daf" -Cache C:\sf_cache
```

`-Cache` must be a local disk: the A8 cache is ~20 GB (every event at native size, 1.25x
resampled) and training reads it continuously. `-SkipPrepare` reuses caches already there.
Defaults: width 64, A8 40 epochs, ISX 30 epochs, batch 32, role dropout 0.3, checkpoint
selected on validation event r, TTA at evaluation. On the A6000 expect roughly 2 hours for
the A8 and 1 hour for the ISX.

At the end the script zips everything worth carrying back into `server_results_<timestamp>.zip`
in the repo folder: the JSON, PNG, CSV and log files of both runs, `figures\srv_*`, the commit
hash and the torch/CUDA versions. A few MB. Copy that one file to the laptop and unpack it
with `python scripts/unpack_results.py <the zip>`, which puts `runs/srv_*` and `figures/srv_*`
where every laptop command expects them (plain `unzip` keeps the Windows backslashes in the
names of zips made before 2026-10-27). Caches stay on the server; checkpoints too unless the
script is run with `-Checkpoints`, which adds each run's `best.pt` (about 30 MB at width 64) so
the laptop can redraw figures from the full model, e.g. the submission panel:

```
python -m sperm_fluorescence panel runs/srv_isx_stems --cache data/isx_roles --test 3 --rows "best:5" --um-per-px 0.333 --out figures/aua_isx_panel.png
```

(the laptop's `data/isx_roles` holds the same held-out half at 4,000 objects, enough to choose rows from).

Training on CUDA runs in bfloat16 mixed precision where the GPU supports it (the A6000 does).
The first full A8 run used fp16 and went NaN at epoch 22 of 40; the epoch-20 checkpoint was
kept and is what `runs/srv_a8_stems` reports. bfloat16 has fp32's exponent range and cannot
overflow that way; a batch whose loss is still non-finite is skipped and reported, never
stepped into the weights.

### Relabelling the ImageStream run after the ACRV1/TOMM20 swap (2026-10-27)

Caches and runs made before 2026-10-27 call Ch03 "ACRV1" and Ch11 "TOMM20"; it is the other
way round. The image columns are right, only the names are wrong, so swap the names in place
and redraw the figures from the existing checkpoints instead of retraining:

```powershell
git pull
.venv\Scripts\python.exe scripts\relabel_markers.py --swap ACRV1 TOMM20 data\server_cache\isx_roles runs\srv_isx_stems figures\srv_isx_accuracy
.\scripts\server_stems.ps1 -IsxCif "Z:\Blair_Main\2026\081226\260813_Blair_Sperm_ISX\ALL_1_400_1.cif" -IsxDaf "Z:\Blair_Main\2026\081226\260813_Blair_Sperm_ISX\ALL_1_400_1.daf" -SkipTrain
```

`-SkipTrain` runs only `views`, `sheet`, `masks`, `accuracy` and `slides` and then zips the
results as usual. The relabel refuses to run twice on the same directory.

## If pip cannot reach the internet

Institutional servers often block it. Two ways round:

**Through a proxy** — if the site has one:

```powershell
.venv\Scripts\python -m pip install --proxy http://proxy.host:port -r requirements.txt
```

**Offline** — download the wheels on a machine that does have access, matching the
server's Python version and architecture (`cp312`, `win_amd64`), copy the folder over,
then:

```powershell
.venv\Scripts\python -m pip install --no-index --find-links wheels -r requirements.txt
```

To fetch them on the connected machine:

```powershell
py -3 -m pip download -r requirements.txt -d wheels --platform win_amd64 --python-version 312 --only-binary=:all:
```

## Speed

Reading ~16 GB of marker zips **over a mapped network drive** is usually what sets the
pace. The survey is cheap either way — it reads the central directory plus a dozen small
members per zip. Sampling reads only the drawn events.

If a full pass over the images is ever needed (scoring every event, as `sperm_pbmc` did),
copy the zips to a local disk first; the results are identical either way, since the
features come from the image contents.

## Notes

- `requirements-dev.txt` adds matplotlib and jupyterlab. No script imports them — they
  are for poking at data in a notebook. Skip them on the server.
- The venv lives in `.venv/` and is gitignored, so it never travels with the repo. Each
  machine makes its own.
- Nothing here needs admin rights as long as Python itself was installed per-user.

## The ImageStream run on the GPU machine

The `.cif` is read directly (`analysis/isx_io.py`); nothing from IDEAS is needed. Build
the cache once from the share, then train. `prepare_isx.py` is pure numpy, ~2 minutes.

```powershell
.venv\Scripts\python analysis\prepare_isx.py "Z:\Blair_Main\2026\081226\260813_Blair_Sperm_ISX\ALL_1_400_1.cif" "Z:\Blair_Main\2026\081226\260813_Blair_Sperm_ISX\ALL_1_400_1.daf" --cache C:\Users\blair\katie\isx_cache
```

The ablation -- one joint model with per-marker checkpoints, then one model per marker --
is five `train` runs on the same cache; see the commands in the project log.

## Virtual stains for a whole acquisition (`predict`)

```
.venv\Scripts\python.exe -m sperm_fluorescence predict runs\srv_isx_stems --cache data\server_cache\isx_roles --classes 3I --isx "Z:\...\ALL_1_400_1.cif" --tta --out data\server_cache\isx_vstain
```

writes `data\server_cache\isx_vstain` (about 60 MB per 1,000 objects: float16 images plus two
bool masks per marker). For an unstained `.cif`, first
`prepare --instrument isx --source <cif> --unstained --cache data\server_cache\isx_unstained`,
then `predict ... --cache data\server_cache\isx_unstained --classes 1I`.

## Full-population A8 populations (`scripts/server_populations.ps1`)

After `server_stems.ps1` has left `runs\srv_a8_stems` and `data\server_cache\a8_roles`:

```
git pull
.\scripts\server_populations.ps1
```

Virtual stains for the whole held-out replicate (60,000 objects → `data\server_cache\a8_vstain`,
about 12 GB, stays on the server), the feature catalogue and aligned stacks (~2 h, single
process), virtual stains and stacks for the training replicate, an autoencoder trained on
the training replicate and applied to the held-out one, clustering at k = 2..5 on the latent
and k = 2..4 on the features with stain-only galleries, the features-versus-latent
comparison, the compartment-agreement figure, and `server_populations_<timestamp>.zip`
(populations, summaries, embeddings, figures). `-TrainOnHeldOut` skips the training-replicate
pass; `-WithFeatures` adds the 60,000-row features.csv to the zip. Unpack on the laptop with
`python scripts/unpack_results.py <zip>`.
