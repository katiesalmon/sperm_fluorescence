"""Command line: python -m stainless <command> [args]

  prepare              cache any instrument by its config (roles + markers; the modular path)
  prepare-a8           cache an A8 run (fixed frames, legacy)      prepare-isx     cache an ImageStream .cif (ragged, legacy)
  prepare-generalist   one ragged cache from ISX + A8 bundle        train           train a model on a cache
  sheet                redraw a run's IDEAS-style sheet             masks           stain masks, scored against IDEAS / thresholded truth
  slides               presentation panels for one object          accuracy        scatter, event-r bars, mask IoU bars
  subset               re-score a run on DAPI-defined subsets       ablate          score a stems model under role subsets
  predict-cytpix       run an ISX model on CytPix crops
  spectral             plot a spectral_export.py JSON

Every command takes --help.
"""
import sys

COMMANDS = {
    "prepare": ("stainless.data.prepare", "main"),
    "prepare-a8": ("stainless.data.prepare_a8", "main"),
    "prepare-isx": ("stainless.data.prepare_isx", "main"),
    "prepare-generalist": ("stainless.data.prepare_generalist", "main"),
    "train": ("stainless.train.loop", "main"),
    "sheet": ("stainless.eval.load", "main"),
    "masks": ("stainless.eval.masks", "main"),
    "slides": ("stainless.eval.slides", "main"),
    "accuracy": ("stainless.eval.accuracy", "main"),
    "subset": ("stainless.eval.subset", "main"),
    "ablate": ("stainless.eval.ablate", "main"),
    "predict-cytpix": ("stainless.predict_cytpix", "main"),
    "spectral": ("stainless.eval.spectral", "main"),
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
