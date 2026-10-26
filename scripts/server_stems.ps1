# Full-scale stems models on the GPU machine: A8 (CellView family) and ImageStream (camera family).
# Run from the repo folder in PowerShell after docs/server-setup.md's install:
#
#   .\scripts\server_stems.ps1 -A8Run "Z:\Blair_Main\2026\260709_Blair_Sperm_A8\260709" `
#                              -IsxCif "Z:\...\260813_Blair_Sperm_ISX\ALL_1_400_1.cif" -IsxDaf "Z:\...\ALL_1_400_1.daf" `
#                              -Cache D:\sf_cache -Width 64
#
# Caches go to a LOCAL disk (-Cache), never Z:. Outputs: runs\srv_a8_stems, runs\srv_isx_stems (results.json,
# ablate.json, masks.json, sheet/masks/views PNGs) and figures\srv_* (accuracy, slides). Copy runs\*\*.json,
# runs\*\*.png and figures\ back to the laptop; leave the caches and checkpoints on the server.
param(
    [string]$A8Run = "",
    [string]$IsxCif = "",
    [string]$IsxDaf = "",
    [string]$Cache = "D:\sf_cache",
    [int]$Width = 64,
    [int]$A8Epochs = 40,
    [int]$IsxEpochs = 30,
    [int]$Limit = 0,            # events per class, 0 = all; use e.g. 2000 for a first smoke run
    [switch]$SkipPrepare
)
$ErrorActionPreference = "Stop"
$py = ".venv\Scripts\python.exe"
$common = @("--train", "2", "--test", "3", "--batch", "32", "--width", "$Width", "--channel-scale", "--full-frame", "--aug", "1",
            "--eval-every", "5", "--arch", "stems", "--role-dropout", "0.3", "--select-on", "val_event_r", "--tta")

if ($A8Run -ne "") {
    $a8cache = Join-Path $Cache "a8_roles"
    if (-not $SkipPrepare) {
        & $py -m sperm_fluorescence prepare --instrument a8 --reader a8_run --source $A8Run --cache $a8cache --limit $Limit
    }
    & $py -m sperm_fluorescence train --cache $a8cache --epochs $A8Epochs --roles lightloss,fsc,ssc --out runs\srv_a8_stems @common
    & $py -m sperm_fluorescence ablate runs\srv_a8_stems --cache $a8cache --test 3 --subsets "lightloss; lightloss,fsc; lightloss,fsc,ssc; ssc"
    & $py -m sperm_fluorescence views  runs\srv_a8_stems --cache $a8cache --test 3 --subsets "lightloss,fsc,ssc; lightloss; ssc" --per-well 2 --crop 56 --out figures\srv_a8_views.png
    & $py -m sperm_fluorescence sheet  runs\srv_a8_stems --cache $a8cache --test 3 --per-well 3
    & $py -m sperm_fluorescence masks  runs\srv_a8_stems --cache $a8cache --test 3 --limit 1000 --calib 400
    & $py -m sperm_fluorescence accuracy runs\srv_a8_stems --cache $a8cache --test 3 --title "BD FACSDiscover A8, held-out replicate (full run)" --sample 5000 --out figures\srv_a8_accuracy
    & $py -m sperm_fluorescence slides runs\srv_a8_stems --cache $a8cache --test 3 --well S --crop 56 --title "BD FACSDiscover A8: three markers predicted from light loss + FSC + SSC" --input-name "Light loss" --um-per-px 0.333 --approx-scale --n 4 --out figures\srv_a8_slides
}

if ($IsxCif -ne "") {
    $isxcache = Join-Path $Cache "isx_roles"
    if (-not $SkipPrepare) {
        & $py -m sperm_fluorescence prepare --instrument isx --source $IsxCif --daf $IsxDaf --cache $isxcache --limit $Limit
    }
    & $py -m sperm_fluorescence train --cache $isxcache --epochs $IsxEpochs --roles brightfield,brightfield2,darkfield --out runs\srv_isx_stems @common
    & $py -m sperm_fluorescence ablate runs\srv_isx_stems --cache $isxcache --test 3 --subsets "brightfield; brightfield,brightfield2; brightfield,brightfield2,darkfield; darkfield"
    & $py -m sperm_fluorescence views  runs\srv_isx_stems --cache $isxcache --test 3 --subsets "brightfield,brightfield2,darkfield; brightfield; darkfield" --per-well 4 --out figures\srv_isx_views.png
    & $py -m sperm_fluorescence sheet  runs\srv_isx_stems --cache $isxcache --test 3 --per-well 4
    & $py -m sperm_fluorescence masks  runs\srv_isx_stems --cache $isxcache --test 3 --isx $IsxCif --limit 1000 --calib 400
    & $py -m sperm_fluorescence accuracy runs\srv_isx_stems --cache $isxcache --test 3 --title "ImageStream, held-out half of the tube (full run)" --sample 5000 --out figures\srv_isx_accuracy
    & $py -m sperm_fluorescence slides runs\srv_isx_stems --cache $isxcache --test 3 --population sperm --title "ImageStream (60x): four markers predicted from brightfield" --input-name "Brightfield" --um-per-px 0.333 --n 4 --out figures\srv_isx_slides
}
Write-Host "ALL DONE"
