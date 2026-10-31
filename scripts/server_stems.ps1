# Full-scale stems models on the GPU machine: A8 (CellView family) and ImageStream (camera family).
# Run from the repo folder in PowerShell after docs/server-setup.md's install:
#
#   .\scripts\server_stems.ps1 -A8Run "Z:\...\260812" -A8Config configs\instruments\a8_260812.toml -Suffix _260812 -A8Well M -A8Stained "LDHC,ACRV1"   # the 2026-08-12 plate, TOMM20 panel
#   .\scripts\server_stems.ps1 -A8Run <run> -DenoiseTargets smooth   # retrain on denoised measured stains -> runs\srv_a8_stems_smooth, figures\srv_a8_smooth_*
#   .\scripts\server_stems.ps1 -A8Run "Z:\Blair_Main\2026\260709_Blair_Sperm_A8\260709" `
#                              -IsxCif "Z:\...\260813_Blair_Sperm_ISX\ALL_1_400_1.cif" -IsxDaf "Z:\...\ALL_1_400_1.daf" `
#                              -Cache C:\sf_cache -Width 64        (any LOCAL disk; default <repo>\data\server_cache)
#
# Caches go to a LOCAL disk (-Cache), never Z:. Outputs: runs\srv_a8_stems, runs\srv_isx_stems (results.json,
# ablate.json, masks.json, sheet/masks/views PNGs) and figures\srv_* (accuracy, slides, panel). At the end everything
# worth carrying back is zipped into server_results_<timestamp>.zip in the repo folder (no caches; best.pt only with -Checkpoints).
param(
    [string]$A8Run = "",
    [string]$IsxCif = "",
    [string]$IsxDaf = "",
    [string]$Cache = "",          # local disk; default: <repo>\data\server_cache (data\ is gitignored)
    [int]$Width = 64,
    [int]$A8Epochs = 40,
    [int]$IsxEpochs = 30,
    [int]$Limit = 0,            # events per class, 0 = all; use e.g. 2000 for a first smoke run
    [string]$A8Config = "a8",        # instrument config name or path: a8 (July plate) or configs\instruments\a8_260812.toml (one well, TOMM20 panel)
    [string]$Suffix = "",            # run / cache / figure suffix for a second A8 dataset, e.g. _260812
    [string]$A8Well = "S",           # the well the slides figure draws (the July plate: S; a single mixed well: M)
    [string]$A8Stained = "LDHC/AKAP4,ACRV1",   # markers that must be bright for a panel row to count as a stained sperm (2026-08-12 plate: LDHC,ACRV1)
    [string]$DenoiseTargets = "",   # "smooth" or "floor": measured stains denoised before training; cache a8_roles_<mode>, run srv_a8_stems_<mode>, figures srv_a8_<mode>_*
    [switch]$SkipPrepare,
    [switch]$SkipTrain,         # reuse runs\srv_*: no prepare, train or ablate; redraw views, sheet, masks, accuracy, slides, panel
    [switch]$Checkpoints        # also put each run's best.pt (~30 MB) in the zip, so the laptop can redraw figures from the full model
)
$ErrorActionPreference = "Stop"
$py = ".venv\Scripts\python.exe"
if ($Cache -eq "") { $Cache = Join-Path (Split-Path $PSScriptRoot -Parent) "data\server_cache" }
New-Item -ItemType Directory -Force -Path $Cache | Out-Null
$common = @("--train", "2", "--test", "3", "--batch", "32", "--width", "$Width", "--channel-scale", "--full-frame", "--aug", "1",
            "--eval-every", "5", "--arch", "stems", "--role-dropout", "0.3", "--select-on", "val_event_r", "--tta")

if ($A8Run -ne "") {
    $sfx = $Suffix + $(if ($DenoiseTargets -ne "") { "_$DenoiseTargets" } else { "" })
    $a8cache = "$Cache\a8_roles$sfx"; $a8out = "runs\srv_a8_stems$sfx"; $a8fig = "figures\srv_a8$sfx"
    $dn = if ($DenoiseTargets -ne "") { @("--denoise-targets", $DenoiseTargets) } else { @() }
    if (-not $SkipPrepare -and -not $SkipTrain) {
        & $py -m sperm_fluorescence prepare --instrument $A8Config --reader a8_run --source $A8Run --cache $a8cache --limit $Limit @dn
    }
    if (-not $SkipTrain) {
        & $py -m sperm_fluorescence train --cache $a8cache --epochs $A8Epochs --roles lightloss,fsc,ssc --out $a8out @common
        & $py -m sperm_fluorescence ablate $a8out --cache $a8cache --test 3 --subsets "lightloss; lightloss,fsc; lightloss,fsc,ssc; ssc"
    }
    & $py -m sperm_fluorescence views  $a8out --cache $a8cache --test 3 --subsets "lightloss,fsc,ssc; lightloss; ssc" --per-well 2 --crop 56 --out ${a8fig}_views.png
    & $py -m sperm_fluorescence sheet  $a8out --cache $a8cache --test 3 --per-well 3
    & $py -m sperm_fluorescence masks  $a8out --cache $a8cache --test 3 --limit 1000 --calib 400
    & $py -m sperm_fluorescence accuracy $a8out --cache $a8cache --test 3 --title "BD FACSDiscover A8, held-out replicate (full run)" --sample 5000 --out ${a8fig}_accuracy
    & $py -m sperm_fluorescence slides $a8out --cache $a8cache --test 3 --well $A8Well --crop 56 --title "BD FACSDiscover A8: three markers predicted from light loss + FSC + SSC" --input-name "Light loss" --um-per-px 0.333 --approx-scale --n 4 --out ${a8fig}_slides
    & $py -m sperm_fluorescence panel  $a8out --cache $a8cache --test 3 --rows "best:5" --stained $A8Stained --size 56 --um-per-px 0.333 --approx-scale --range sample --out ${a8fig}_panel.png
}

if ($IsxCif -ne "") {
    $isxcache = "$Cache\isx_roles"
    if (-not $SkipPrepare -and -not $SkipTrain) {
        & $py -m sperm_fluorescence prepare --instrument isx --source $IsxCif --daf $IsxDaf --cache $isxcache --limit $Limit
    }
    if (-not $SkipTrain) {
        & $py -m sperm_fluorescence train --cache $isxcache --epochs $IsxEpochs --roles brightfield,brightfield2,darkfield --out runs\srv_isx_stems @common
        & $py -m sperm_fluorescence ablate runs\srv_isx_stems --cache $isxcache --test 3 --subsets "brightfield; brightfield,brightfield2; brightfield,brightfield2,darkfield; darkfield"
    }
    & $py -m sperm_fluorescence views  runs\srv_isx_stems --cache $isxcache --test 3 --subsets "brightfield,brightfield2,darkfield; brightfield; darkfield" --per-well 4 --out figures\srv_isx_views.png
    & $py -m sperm_fluorescence sheet  runs\srv_isx_stems --cache $isxcache --test 3 --per-well 4
    & $py -m sperm_fluorescence masks  runs\srv_isx_stems --cache $isxcache --test 3 --isx $IsxCif --limit 1000 --calib 400
    & $py -m sperm_fluorescence accuracy runs\srv_isx_stems --cache $isxcache --test 3 --title "ImageStream, held-out half of the tube (full run)" --sample 5000 --out figures\srv_isx_accuracy
    & $py -m sperm_fluorescence slides runs\srv_isx_stems --cache $isxcache --test 3 --population sperm --title "ImageStream (60x): four markers predicted from brightfield" --input-name "Brightfield" --um-per-px 0.333 --n 4 --out figures\srv_isx_slides
    & $py -m sperm_fluorescence panel  runs\srv_isx_stems --cache $isxcache --test 3 --rows "best:5" --um-per-px 0.333 --out figures\srv_isx_panel.png
}
# ---- one zip to carry back: every JSON, PNG, CSV and log from the two runs, plus figures\srv_* (no caches; checkpoints only with -Checkpoints)
$stamp = Get-Date -Format "yyyyMMdd_HHmm"
$zip = "server_results_$stamp.zip"
$staging = Join-Path $env:TEMP "sf_results_$stamp"
New-Item -ItemType Directory -Force -Path $staging | Out-Null
$zipRuns = @("runs\srv_isx_stems")
if ($A8Run -ne "") { $zipRuns = @($a8out) + $zipRuns } else { $zipRuns = @("runs\srv_a8_stems") + $zipRuns }
foreach ($run in $zipRuns) {
    if (Test-Path $run) {
        $dest = Join-Path $staging $run
        New-Item -ItemType Directory -Force -Path $dest | Out-Null
        Get-ChildItem $run -File | Where-Object { $_.Extension -in ".json", ".png", ".csv", ".log" } | Copy-Item -Destination $dest
        if ($Checkpoints -and (Test-Path (Join-Path $run "best.pt"))) { Copy-Item (Join-Path $run "best.pt") -Destination $dest }
    }
}
Get-ChildItem figures -Directory -Filter "srv_*" -ErrorAction SilentlyContinue | ForEach-Object { Copy-Item $_.FullName -Destination (Join-Path $staging "figures\$($_.Name)") -Recurse -Force }
Get-ChildItem figures -File -Filter "srv_*" -ErrorAction SilentlyContinue | ForEach-Object { New-Item -ItemType Directory -Force -Path (Join-Path $staging "figures") | Out-Null; Copy-Item $_.FullName -Destination (Join-Path $staging "figures") -Force }
& $py -c "import torch, sys; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" | Out-File -Encoding utf8 (Join-Path $staging "environment.txt")
git rev-parse HEAD | Out-File -Encoding utf8 (Join-Path $staging "commit.txt")
if (Test-Path $zip) { Remove-Item $zip }
Add-Type -AssemblyName System.IO.Compression.FileSystem
# .NET's ZipFile writes forward-slash member names; PowerShell 5.1's Compress-Archive writes backslashes, which unzip on macOS keeps
[System.IO.Compression.ZipFile]::CreateFromDirectory($staging, (Join-Path (Get-Location) $zip), [System.IO.Compression.CompressionLevel]::Optimal, $false)
Remove-Item $staging -Recurse -Force
Write-Host "ALL DONE -- copy $zip ($([math]::Round((Get-Item $zip).Length / 1MB, 1)) MB) back to the laptop"
