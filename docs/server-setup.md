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
