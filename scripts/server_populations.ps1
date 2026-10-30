# Full-population A8 populations on the server: virtual stains for the whole held-out replicate,
# the feature catalogue and aligned stacks, an autoencoder trained on the training replicate,
# clustering at k = 2..5 (latent) and k = 2..4 (features), stain-only galleries, and one zip.
#
#   .\scripts\server_populations.ps1                      # after server_stems.ps1 has made runs\srv_a8_stems and data\server_cache\a8_roles
#   .\scripts\server_populations.ps1 -TrainOnHeldOut      # autoencoder trained on the held-out replicate itself (no replicate-2 pass; faster)
#   .\scripts\server_populations.ps1 -WithFeatures        # also put features.csv (hundreds of MB) in the zip
#   .\scripts\server_populations.ps1 -GalleriesOnly       # runs\pop_srv_a8_* exist: only redraw the galleries (stains, light loss, SSC, measured) and zip
#   .\scripts\server_populations.ps1 -DenoiseRef -GalleriesOnly   # add a features pass with the measured stains denoised (~2 h), k = 2 against that measured side, its confusion matrix
#
# Time, RTX A6000: predict ~10 min per replicate, features ~2 h for 60,000 objects (single process),
# pixels-only for the training replicate ~10 min, autoencoder ~30 min, clustering and galleries ~20 min.
# Outputs: data\server_cache\a8_vstain (store, ~12 GB, stays), runs\pop_srv_a8_*, figures\srv_a8_pop_*,
# and server_populations_<timestamp>.zip with the populations, summaries, embeddings and figures.
param(
    [string]$Cache = "",
    [string]$Run = "runs\srv_a8_stems",
    [switch]$TrainOnHeldOut,
    [switch]$WithFeatures,
    [switch]$GalleriesOnly,
    [switch]$DenoiseRef,
    [int]$Epochs = 60,
    [int]$Latent = 16
)
$ErrorActionPreference = "Stop"
$py = ".venv\Scripts\python.exe"
if ($Cache -eq "") { $Cache = Join-Path (Get-Location) "data\server_cache" }
$a8cache = "$Cache\a8_roles"; $store = "$Cache\a8_vstain"; $store2 = "$Cache\a8_vstain_train"
$feat = "$store\features.csv"; $pix = "$store\features.pixels.npz"; $emb = "$store\emb_dcae.npz"

if (-not $GalleriesOnly) {
Write-Host "== virtual stains for the held-out replicate (3S, 3P, 3SP)"
& $py -m sperm_fluorescence predict $Run --cache $a8cache --classes 3S,3P,3SP --tta --out $store
Write-Host "== feature catalogue + aligned stacks"
& $py -m sperm_fluorescence features $store --out $feat

if (-not $TrainOnHeldOut) {
    Write-Host "== training replicate (2S, 2P, 2SP): virtual stains and aligned stacks only, for the autoencoder"
    & $py -m sperm_fluorescence predict $Run --cache $a8cache --classes 2S,2P,2SP --tta --out $store2
    & $py -m sperm_fluorescence features $store2 --pixels-only --out "$store2\features.csv"
    & $py analysis\autoencoder_embedding.py "$store2\features.pixels.npz" --out "$store2\emb_dcae.npz" --latent $Latent --epochs $Epochs --apply $pix --apply-out $emb
} else {
    & $py analysis\autoencoder_embedding.py $pix --out $emb --latent $Latent --epochs $Epochs
}

}
# one run -> four galleries of the same clusters: virtual stains, light loss, side scatter, measured stains
function Draw-Galleries($run, $title, $outStem) {
    & $py analysis\stain_gallery.py $run --store $store --rows 6 --title $title --out "$outStem.png"
    & $py analysis\stain_gallery.py $run --store $store --rows 6 --input lightloss --title "$title (light loss)" --out "${outStem}_lightloss.png"
    & $py analysis\stain_gallery.py $run --store $store --rows 6 --input ssc --title "$title (side scatter)" --out "${outStem}_ssc.png"
    & $py analysis\stain_gallery.py $run --store $store --rows 6 --measured --title "$title (measured stains)" --out "${outStem}_measured.png"
}
New-Item -ItemType Directory -Force -Path figures | Out-Null
foreach ($k in 2, 3, 4, 5) {
    Write-Host "== latent, k = $k"
    if (-not $GalleriesOnly) { & $py -m sperm_fluorescence populate $feat --set pixels --embedding "e=$emb" --pixel-components $Latent --measured-set stain+intensity --method gmm --k $k --no-gates --store $store --gallery-n 8 --out "runs\pop_srv_a8_latent_k$k" }
    Draw-Galleries "runs\pop_srv_a8_latent_k$k" "A8, full held-out replicate, k = $k on the autoencoder latent" "figures\srv_a8_pop_latent_k$k"
}
foreach ($k in 2, 3, 4) {
    Write-Host "== features, k = $k"
    if (-not $GalleriesOnly) { & $py -m sperm_fluorescence populate $feat --set all --k $k --no-gates --store $store --gallery-n 8 --out "runs\pop_srv_a8_feat_k$k" }
    Draw-Galleries "runs\pop_srv_a8_feat_k$k" "A8, full held-out replicate, k = $k on all features" "figures\srv_a8_pop_feat_k$k"
}
# same k = 2, but the measured side also gets the light-loss / FSC / SSC columns: both sides see identical feature families and only
# the stain source differs (virtual vs measured); the confusion matrix against this side is the apples-to-apples one
if (-not $GalleriesOnly) { & $py -m sperm_fluorescence populate $feat --set all --k 2 --no-gates --measured-inputs --store $store --gallery-n 8 --out "runs\pop_srv_a8_feat_k2_mi" }
& $py analysis\confusion_matrix.py "runs\pop_srv_a8_feat_k2" --names sperm leukocyte --title "A8, full held-out replicate, k = 2 on all features: label-free clusters against the measured stains alone" --out "figures\srv_a8_pop_feat_k2_confusion.png"
if (Test-Path "runs\pop_srv_a8_feat_k2_mi\populations.csv") { & $py analysis\confusion_matrix.py "runs\pop_srv_a8_feat_k2_mi" --names sperm leukocyte --title "A8, full held-out replicate, k = 2 on all features: label-free clusters against the measured stains + the same label-free inputs" --out "figures\srv_a8_pop_feat_k2_mi_confusion.png" }
# measured stains denoised (Gaussian 1 px + noise floor) before the reference masks: a second measured-side truth. On the laptop
# sample it did not improve agreement (99.0 % vs 99.7 %); at 60,000 the question is whether the 247 dim-CD45 leukocytes move.
if ($DenoiseRef) {
    $featDn = "$store\features_dn.csv"
    if (-not (Test-Path $featDn)) { & $py -m sperm_fluorescence features $store --denoise-ref smooth --no-pixels --out $featDn }
    & $py -m sperm_fluorescence populate $featDn --set all --k 2 --no-gates --store $store --gallery-n 8 --out "runs\pop_srv_a8_feat_k2_dn"
    & $py analysis\confusion_matrix.py "runs\pop_srv_a8_feat_k2_dn" --names sperm leukocyte --title "A8, full held-out replicate, k = 2 on all features: label-free clusters against the DENOISED measured stains" --out "figures\srv_a8_pop_feat_k2_dn_confusion.png"
    & $py -m sperm_fluorescence populate $featDn --set all --k 2 --no-gates --measured-inputs --store $store --gallery-n 8 --out "runs\pop_srv_a8_feat_k2_dn_mi"
    & $py analysis\confusion_matrix.py "runs\pop_srv_a8_feat_k2_dn_mi" --names sperm leukocyte --title "A8, full held-out replicate, k = 2: label-free clusters against the DENOISED measured stains + the same inputs" --out "figures\srv_a8_pop_feat_k2_dn_mi_confusion.png"
}
# the objects the k = 2 clustering places against their well, split by what the measured stains say (sperm carried into the
# PBMC preparation, leukocytes in the ejaculate, and the genuinely misassigned); drawn in the same four views
$mis = "runs\pop_srv_a8_feat_k2_misplaced"
& $py analysis\subset_run.py "runs\pop_srv_a8_feat_k2" --out $mis `
    --group "PBMC well, sperm cluster, sperm by measured stains=``id.class`` == '3P' and cluster == 1 and measured_cluster == 1" `
    --group "PBMC well, sperm cluster, leukocyte by measured stains=``id.class`` == '3P' and cluster == 1 and measured_cluster == 0" `
    --group "sperm well, leukocyte cluster, leukocyte by measured stains=``id.class`` == '3S' and cluster == 0 and measured_cluster == 0" `
    --group "sperm well, leukocyte cluster, sperm by measured stains=``id.class`` == '3S' and cluster == 0 and measured_cluster == 1"
$misNames = @("0=PBMC well -> sperm cluster: sperm by measured stains", "1=PBMC well -> sperm cluster: leukocyte by measured stains", "2=sperm well -> leukocyte cluster: leukocyte by measured stains", "3=sperm well -> leukocyte cluster: sperm by measured stains")
$misTitle = "A8, full held-out replicate, k = 2: objects placed against their well"
& $py analysis\stain_gallery.py $mis --store $store --rows 4 --names $misNames --title $misTitle --out "figures\srv_a8_pop_feat_k2_misplaced.png"
& $py analysis\stain_gallery.py $mis --store $store --rows 4 --names $misNames --input lightloss --title "$misTitle (light loss)" --out "figures\srv_a8_pop_feat_k2_misplaced_lightloss.png"
& $py analysis\stain_gallery.py $mis --store $store --rows 4 --names $misNames --input ssc --title "$misTitle (side scatter)" --out "figures\srv_a8_pop_feat_k2_misplaced_ssc.png"
& $py analysis\stain_gallery.py $mis --store $store --rows 4 --names $misNames --measured --title "$misTitle (measured stains)" --out "figures\srv_a8_pop_feat_k2_misplaced_measured.png"
# every object on which the label-free and measured-side k = 2 clusterings disagree, by well and direction, in the four views
$dis = "runs\pop_srv_a8_feat_k2_disagree"
& $py analysis\subset_run.py "runs\pop_srv_a8_feat_k2" --out $dis `
    --group "PBMC well: measured sperm, label-free leukocyte=``id.class`` == '3P' and measured_cluster == 1 and cluster == 0" `
    --group "PBMC well: measured leukocyte, label-free sperm=``id.class`` == '3P' and measured_cluster == 0 and cluster == 1" `
    --group "sperm well: measured sperm, label-free leukocyte=``id.class`` == '3S' and measured_cluster == 1 and cluster == 0" `
    --group "sperm well: measured leukocyte, label-free sperm=``id.class`` == '3S' and measured_cluster == 0 and cluster == 1" `
    --group "mixed well: measured sperm, label-free leukocyte=``id.class`` == '3SP' and measured_cluster == 1 and cluster == 0" `
    --group "mixed well: measured leukocyte, label-free sperm=``id.class`` == '3SP' and measured_cluster == 0 and cluster == 1"
$disNames = @("0=PBMC well: measured sperm, label-free leukocyte", "1=PBMC well: measured leukocyte, label-free sperm", "2=sperm well: measured sperm, label-free leukocyte", "3=sperm well: measured leukocyte, label-free sperm", "4=mixed well: measured sperm, label-free leukocyte", "5=mixed well: measured leukocyte, label-free sperm")
$disTitle = "A8, full held-out replicate, k = 2: label-free and measured-side clusterings disagree"
& $py analysis\stain_gallery.py $dis --store $store --rows 4 --names $disNames --title $disTitle --out "figures\srv_a8_pop_feat_k2_disagree.png"
& $py analysis\stain_gallery.py $dis --store $store --rows 4 --names $disNames --input lightloss --title "$disTitle (light loss)" --out "figures\srv_a8_pop_feat_k2_disagree_lightloss.png"
& $py analysis\stain_gallery.py $dis --store $store --rows 4 --names $disNames --input ssc --title "$disTitle (side scatter)" --out "figures\srv_a8_pop_feat_k2_disagree_ssc.png"
& $py analysis\stain_gallery.py $dis --store $store --rows 4 --names $disNames --measured --title "$disTitle (measured stains)" --out "figures\srv_a8_pop_feat_k2_disagree_measured.png"
& $py analysis\compare_galleries.py "runs\pop_srv_a8_feat_k2" "runs\pop_srv_a8_latent_k2" --store $store --labels "features" "latent" --title "A8, full held-out replicate, k = 2: features versus autoencoder latent" --out "figures\srv_a8_pop_feature_vs_latent_k2.png"
# the sperm cluster is cluster 1 of the feature k = 2 run (the larger one; the 2026-10-29 run used the latent k = 2 cluster 0, which at 60,000 objects was not the sperm)
if (-not $GalleriesOnly) { & $py analysis\compartment_agreement.py $feat --within "runs\pop_srv_a8_feat_k2\populations.csv:1" --prefixes preds refs --measures LDHC_AKAP4.sigma_major_um LDHC_AKAP4.sigma_minor_um LDHC_AKAP4.anisotropy ACRV1.sigma_major_um ACRV1.within_2um CD45.sigma_major_um --title "A8, full held-out replicate: compartment measurements, virtual against measured (sperm cluster)" --out "figures\srv_a8_pop_compartment_agreement.png" }

# ---- zip: populations, summaries, embeddings, figures (features.csv only with -WithFeatures)
$stamp = Get-Date -Format "yyyyMMdd_HHmm"; $zip = "server_populations_$stamp.zip"
$staging = Join-Path $env:TEMP "sf_pop_$stamp"; New-Item -ItemType Directory -Force -Path $staging | Out-Null
foreach ($d in Get-ChildItem runs -Directory -Filter "pop_srv_a8_*") {
    $dest = Join-Path $staging "runs\$($d.Name)"; New-Item -ItemType Directory -Force -Path $dest | Out-Null
    Get-ChildItem $d.FullName -File | Where-Object { $_.Extension -in ".csv", ".json", ".png" } | Copy-Item -Destination $dest
}
New-Item -ItemType Directory -Force -Path (Join-Path $staging "figures") | Out-Null
Get-ChildItem figures -File -Filter "srv_a8_pop_*" | Copy-Item -Destination (Join-Path $staging "figures")
New-Item -ItemType Directory -Force -Path (Join-Path $staging "data\a8_vstain") | Out-Null
Copy-Item $emb -Destination (Join-Path $staging "data\a8_vstain"); Copy-Item "$store\meta.json" -Destination (Join-Path $staging "data\a8_vstain")
Copy-Item "$store\features.columns.json" -Destination (Join-Path $staging "data\a8_vstain")
if ($WithFeatures) { Copy-Item $feat -Destination (Join-Path $staging "data\a8_vstain") }
git rev-parse HEAD | Out-File -Encoding utf8 (Join-Path $staging "commit.txt")
if (Test-Path $zip) { Remove-Item $zip }
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory($staging, (Join-Path (Get-Location) $zip), [System.IO.Compression.CompressionLevel]::Optimal, $false)
Remove-Item $staging -Recurse -Force
Write-Host "ALL DONE -- copy $zip ($([math]::Round((Get-Item $zip).Length / 1MB, 1)) MB) back to the laptop"
