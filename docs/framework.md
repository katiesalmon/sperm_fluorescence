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
