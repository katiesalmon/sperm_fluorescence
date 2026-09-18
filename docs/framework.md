# A per-marker stain-predictor framework

*Design sketch, 2026-09-17. Path 1 from [paths.md](paths.md), but with one network per
marker rather than a shared encoder with four heads.*

---

## The two closest papers disagree, and the closer one trains per marker

| | Structure | Result |
| --- | --- | --- |
| [DeepIFC](https://onlinelibrary.wiley.com/doi/10.1002/cyto.a.24770) — imaging flow cytometry, single-cell crops | **One model per marker.** Inputs the brightfield/darkfield channels, outputs one fluorescent channel | CD45 r = 0.90 |
| [Multi-task label-free phenotyping](https://arxiv.org/html/2605.14717) — single-cell images | **Shared encoder**, task-gated classification + regression heads | CD45 r = 0.799, aggregate 0.726 |

The shared-encoder paper is the more recent and more elaborate. The per-marker paper is
**the closer match to our modality** — single-cell crops from an imaging flow cytometer,
not a microscope field — and it scores higher on the marker both report.

That is not decisive, but it means per-marker is not the naive option. It is the option
with the better modality-matched precedent.

## Four reasons per-marker fits *this* dataset specifically

1. **The markers differ in difficulty by 4x.** DAPI 0.776 against ACRV1 0.209. In a
   shared encoder the easy markers dominate the gradient and the hard one inherits
   whatever representation they built. That is textbook negative transfer, and it lands on
   exactly the marker we most want to move.
2. **One channel is contaminated.** LDHC misbehaves in replicate 2 — its well-separation
   reverses sign between replicates (AUC 0.732 vs 0.179). With a shared encoder that bad
   channel's gradient shapes the representation used by DAPI and CD45 too. Isolation
   contains the damage.
3. **The markers want different targets.** ACRV1 is the natural candidate for ordinal bins
   (intact / intermediate / reacted); DAPI and CD45 are well-behaved continuous
   regressions. A shared trunk with four heads can still do this, but the *loss weighting*
   between a cross-entropy and two regressions is a hyperparameter we would rather not
   own.
4. **The negative controls differ in direction.** Predicted ACRV1 and LDHC on a PBMC
   should be ~0. Predicted **DAPI on a PBMC should be high** — PBMCs have nuclei, and a
   model predicting zero has learned "sperm", not "DNA". Per-marker configs let that
   expectation be declared rather than remembered.

## What the framework is

The framework is not the network. It is **the contract every marker is held to**, so that
adding a marker is a config entry and not a new script — and so the controls cannot be
skipped, because they are not optional.

```
stains/
  spec.py         MarkerSpec: the per-marker contract (below)
  data.py         bundle -> (image, mask, target); compensation applied once, centrally
  transforms.py   asinh / within-acquisition rank / ordinal bins
  models.py       encoder registry + head registry
  train.py        one spec -> one artifact
  evaluate.py     the control battery, identical for every marker
  report.py       results table across markers and runs
markers/
  dapi.yaml  acrv1.yaml  ldhc.yaml  cd45.yaml
```

### The per-marker contract

```yaml
# markers/acrv1.yaml
name:            ACRV1
detector:        BL2-A
compensate:      true              # via the acquisition's own $SPILLOVER
target:          ordinal           # asinh | rank | ordinal
ordinal_bins:    [intact, intermediate, reacted]
head:            ordinal           # scalar | heteroscedastic | ordinal
loss:            ordinal_ce        # smooth_l1_pearson | beta_nll | ordinal_ce

floor:           0.209             # morphology baseline, from docs/results.md
                                   # evaluate.py reports cleared / not cleared

negative_population:               # what SHOULD be near zero
  wells: [2P, 3P]
  expect: low                      # PBMCs have no acrosome
```

`dapi.yaml` differs in the one place that matters:

```yaml
name: DAPI
detector: VL1-A
target: asinh
head: heteroscedastic
loss: beta_nll
floor: 0.776
negative_population:
  wells: [2P, 3P]
  expect: high                     # PBMCs are diploid -- zero here means the model
                                   # learned "sperm", not "DNA"
```

### What is fixed for everyone

Because these are where mistakes happen, they live in the framework, not the config:

- **Input pipeline** — 16-bit crop, masked by the instrument's layer-1 object mask,
  normalised in SNR units from corner patches.
- **Compensation** — applied once, centrally, across every detector in `$SPILLOVER`.
- **The split** — held out by replicate. Train on 2, test on 3. Not a knob.
- **The control battery**, run automatically for every marker:

  | Control | Passes when |
  | --- | --- |
  | Held-out-replicate score | — the headline |
  | Within-replicate score | reported alongside, so the gap is always visible |
  | Permutation null, n runs | score clears the top of the null range |
  | Morphology floor | score clears the number in the config |
  | Biological negative population | matches the declared direction |
  | Background-only control | corner patches alone score near zero |

  A run that does not report all six is not a result.

## The middle path worth testing

Per-marker heads do not have to mean per-marker *encoders*. **Pretrain one encoder,
freeze or fine-tune it, train independent predictors on top.** That keeps the shared
representation — the encoder that finds head, midpiece and tail is genuinely the same for
all four markers — while each marker keeps its own loss, transform, stopping and controls.
It avoids joint training's gradient competition without paying 4x for redundant feature
learning.

Making `encoder: shared_pretrained | per_marker | shared_joint` a config value turns the
whole question into an experiment rather than an architecture commitment. That is the
single most valuable thing the framework buys.

## The honest case against building this now

- **Four markers and one dataset is thin justification for an abstraction.** The usual
  rule is to build the framework at the second instance, not the first.
- **The cost is real**: a week on scaffolding is a week not spent on results, and
  [paths.md](paths.md) argues the cheapest informative work is compartment features, which
  need none of this.
- **4x the training runs and 4x the artifacts**, for markers where two of four have ~0.03
  of headroom.
- **Four pipelines can drift apart** — which is the problem the framework exists to
  prevent, so it only pays off if the contract is actually enforced rather than bypassed.

**The proportionate version:** build `spec.py`, `transforms.py` and `evaluate.py` — the
contract and the control battery — and let `train.py` be a plain script at first. The
controls are the part that must not be optional. The rest can stay small until there is a
second panel, a phase channel, or Tomm20 to add.

---

## Foundation model first, then a head per marker?

*Added 2026-09-17, after a third research pass on what is actually public and what SSL
has done on this exact kind of data.*

### Do we have enough data?

Two different questions hide in that one.

**Enough to pretrain a domain encoder: yes, comfortably.** The closest precedent —
[self-supervised learning for imaging flow cytometry](https://www.nature.com/articles/s41378-026-01236-x)
(2026) — pretrained MoCo v2 / ResNet-18 on **80,347** single-channel grayscale 200 x 200
brightfield IFC crops. We have 282,874 at 248 x 248 and 16-bit. Same modality, same image
style, three and a half times the count.

**Enough to build a *foundation model*: no.** Foundation models need diversity, not
count. Ours is one instrument, one day, two cell types, one operator. What we would get is
an encoder for *this* data, and it should be called that.

### The more important question: would it help?

That same paper is the sobering part. Its SSL encoder, frozen, with a linear head, scored
**0.7 to 2.0 points *below* a supervised ResNet-18** on every downstream task. The benefit
it demonstrated was transfer to cell types absent from pretraining — not accuracy on the
distribution it was trained on. That is the label-efficiency story again, and we have no
label scarcity: every one of the 282,874 crops already carries a measured target.

[ViTally Consistent](https://arxiv.org/html/2411.02572) (Recursion) adds two things.
Domain-pretrained beats generic: even the smallest microscopy MAE (CA-MAE-S/16) outperforms
large ImageNet ViTs. And the representations still carry batch effects that need post-hoc
correction — SSL does not make the acquisition confound go away, it learns it.

So: SSL pretraining on our data would most likely **match** direct supervision, not beat
it, and would not solve the batch problem. Its value would be a reusable embedding, which
is worth something but is not the objective in [GOAL.md](../GOAL.md).

### What is public, ranked by how close it is to our data

| Model | Trained on | Closeness | Notes |
| --- | --- | --- | --- |
| [OpenPhenom](https://huggingface.co/recursionpharma/OpenPhenom) CA-MAE-S/16 | Cell Painting, multi-channel fluorescence | medium | **Channel-agnostic** — takes a single channel natively. Weights on Hugging Face |
| [Cell-DINO](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1013828) | HPA fluorescence single cells | medium | In the official DINOv2 repo |
| [DinoBloom](https://arxiv.org/pdf/2404.05022) | single-cell hematology (stained smears) | medium-high | Single cells, blood — the right *object*, wrong contrast |
| DINOv3 (generic) | 1.7B natural images | low | But the strongest generic dense features; [CytoDINO](https://arxiv.org/abs/2512.17930) adapts it to single-cell cytomorphology with LoRA at 8% trainable parameters |

None was trained on brightfield. The gap between "stained blood smear" and "brightfield
sperm" is real, and the only way to know how much it costs is to probe.

### Four tiers, cheapest first — and the one diagnostic that decides

**Tier 0 — probe frozen public encoders.** Extract features with each of the above,
frozen; fit a linear or small-MLP head per marker; run the full control battery. Hours
of work, no training. This is the decision point: **if frozen DINOv3 plus a linear head
already beats 0.78 on DAPI, representation is not the bottleneck and Tiers 1–3 are not
worth their cost.** If it sits well below the morphology floor, domain adaptation matters.

**Tier 1 — auxiliary-supervised pretraining on everything the instrument measured.** Train
the encoder to predict all ~50 per-event columns at once — every detector, every
morphology and texture measurement — then freeze it and fit a head per marker. This is a
pretext task nobody else has, because nobody else has an instrument that hands over fifty
dense labels per image. It forces the encoder to represent shape and compartment, which is
what the markers need. Cost: one supervised run.

**Tier 2 — continued pretraining of a public encoder on our crops.** The CytoDINO /
Cell-DINO / DINOCell pattern: start from DINOv3 or OpenPhenom, adapt on 282k crops with
LoRA or full fine-tuning under a DINO or MAE objective, then heads per marker. Do not
train from scratch; adaptation is where the recent wins are. Cost: a GPU-day or two — and
whether the server *has* a GPU is currently unknown.

**Tier 3 — image ↔ measurement contrastive pretraining.** CLIP-style: an image encoder and
a small MLP over the 59-dimensional FCS vector, trained to align the same event's two
views ([scPairing](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12664900/) does this for
single-cell multi-omics). The representation is then aligned to *everything* the cytometer
measured, not four channels. This is the genuinely frontier option that fits our data
shape — but it is Tier 1's idea with a harder objective, and it should only follow if
Tier 1 shows the auxiliary signal is worth exploiting.

### Practicalities that bite

- **Single channel vs RGB.** Most encoders expect three channels. OpenPhenom is
  channel-agnostic; for the rest, replicate the channel or learn a 1→3 stem.
- **248 vs 224.** Crop or resize; the object is centred, so a centre crop loses nothing.
- **Batch effects survive pretraining.** Whatever tier, the held-out-replicate split and
  the background-only control stay mandatory. An embedding that separates replicate 2 from
  replicate 3 has learned the acquisition, and probing it will look great on a random
  split.
- **The optical limit still applies.** No encoder recovers phase from absorption. If ACRV1
  does not move at Tier 0, it is unlikely to move at Tier 3.

### Recommendation

Run Tier 0 this week — it is nearly free and it settles whether the rest is worth doing.
My prior is that representation is not the bottleneck for DAPI or CD45, and that ACRV1's
bottleneck is optical rather than representational. If Tier 0 surprises, Tier 1 is the
next step, because it uses the one asset no public model has: fifty free labels per image.
