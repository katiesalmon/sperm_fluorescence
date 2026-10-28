"""Command line: python -m sperm_fluorescence <command> [args]

  prepare              cache any instrument by its config (roles + markers; the modular path)
  prepare-a8           cache an A8 run (fixed frames, legacy)      prepare-isx     cache an ImageStream .cif (ragged, legacy)
  prepare-generalist   one ragged cache from ISX + A8 bundle        train           train a model on a cache
  sheet                redraw a run's IDEAS-style sheet             masks           stain masks, scored against IDEAS / thresholded truth
  slides               presentation panels for one object          accuracy        scatter, event-r bars, mask IoU bars
  subset               re-score a run on DAPI-defined subsets       ablate          score a stems model under role subsets
  views                inputs beside outputs, per input subset     predict-cytpix  run an ISX model on CytPix crops
  spectral             plot a spectral_export.py JSON              denoised        pixel r against a denoised measurement
  compare              within-population event r of several runs, side by side
  predict              virtual stains for every object of a cache -> a store (predicted images + masks)
  features             the feature catalogue over a store -> CSV (+ validation against the .daf)
  panel                submission figure: inputs, measured and predicted stains, one row per object

Every command takes --help.
"""
import sys

COMMANDS = {
    "prepare": ("sperm_fluorescence.data.prepare", "main"),
    "prepare-a8": ("sperm_fluorescence.data.prepare_a8", "main"),
    "prepare-isx": ("sperm_fluorescence.data.prepare_isx", "main"),
    "prepare-generalist": ("sperm_fluorescence.data.prepare_generalist", "main"),
    "train": ("sperm_fluorescence.train.loop", "main"),
    "sheet": ("sperm_fluorescence.eval.load", "main"),
    "masks": ("sperm_fluorescence.eval.masks", "main"),
    "slides": ("sperm_fluorescence.eval.slides", "main"),
    "accuracy": ("sperm_fluorescence.eval.accuracy", "main"),
    "subset": ("sperm_fluorescence.eval.subset", "main"),
    "ablate": ("sperm_fluorescence.eval.ablate", "main"),
    "views": ("sperm_fluorescence.eval.views", "main"),
    "denoised": ("sperm_fluorescence.eval.denoised", "main"),
    "compare": ("sperm_fluorescence.eval.compare", "main"),
    "panel": ("sperm_fluorescence.eval.panel", "main"),
    "predict-cytpix": ("sperm_fluorescence.predict_cytpix", "main"),
    "predict": ("sperm_fluorescence.predict", "main"),
    "features": ("sperm_fluorescence.features.engine", "main"),
    "spectral": ("sperm_fluorescence.eval.spectral", "main"),
}


def main(argv=None):
    import importlib
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__); return 0
    cmd = argv[0]
    if cmd not in COMMANDS:
        print("unknown command %r\n" % cmd + __doc__); return 2
    mod, fn = COMMANDS[cmd]
    return getattr(importlib.import_module(mod), fn)(argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
