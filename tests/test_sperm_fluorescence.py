"""Synthetic-event tests: cache round trip, role selection, stems forward, native-size eval.
Run with `python -m pytest tests` or `python tests/test_sperm_fluorescence.py`."""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sperm_fluorescence.data.cache import open_cache, to_multiple, write_ragged  # noqa: E402
from sperm_fluorescence.data.batching import make_batches, present_of  # noqa: E402
from sperm_fluorescence.models import stems  # noqa: E402


def make_cache(root, roles=("lightloss", "fsc", "ssc"), markers=("A", "B"), n=12, seed=0):
    rng = np.random.default_rng(seed)
    meta = {"layout": "ragged", "roles": list(roles), "target_names": list(markers), "classes": {}}
    for cls in ("2S", "3S"):
        items = []
        for i in range(n):
            h, w = int(rng.integers(20, 40)), int(rng.integers(24, 48))
            x = rng.normal(size=(len(roles), h, w)).astype(np.float32); y = np.abs(rng.normal(size=(len(markers), h, w))).astype(np.float32)
            fx, m = to_multiple(x); fy, _ = to_multiple(y); items.append((fx, fy, m))
        write_ragged(cls, items, root)
        np.save(os.path.join(root, cls + "_avail.npy"), np.ones((n, len(markers)), bool))
        np.save(os.path.join(root, cls + "_present.npy"), np.ones((n, len(roles)), bool))
        np.save(os.path.join(root, cls + "_scalars.npy"), np.zeros((n, 1), np.float32))
        meta["classes"][cls] = {"instrument": "a8", "avail": [True] * len(markers), "present": [True] * len(roles), "repeat": 1}
    json.dump(meta, open(os.path.join(root, "meta.json"), "w"))


def test_cache_round_trip_and_role_selection():
    with tempfile.TemporaryDirectory() as d:
        make_cache(d)
        ds, wells, _ = open_cache(d, ["2S"])
        assert len(ds) == 12 and ds.cin == 3 and ds.cout == 2 and ds.roles == ["lightloss", "fsc", "ssc"]
        x, y, m = ds.item(3)
        assert x.shape[1:] == y.shape[1:] == m.shape and x.shape[1] % 8 == 0 and x.shape[2] % 8 == 0
        assert m.any() and not m.all() or m.all()
        ds2, _, _ = open_cache(d, ["2S"], roles=["ssc", "lightloss"])
        x2, _, _ = ds2.item(3)
        assert ds2.cin == 2 and np.allclose(x2[0], x[2]) and np.allclose(x2[1], x[0])
        X, Y, M = ds2.batch([0, 1, 2])
        assert X.shape[1] == 2 and present_of(ds2, [0, 1, 2]).shape == (3, 2)
        bs = make_batches(ds, np.arange(len(ds)), 4)
        assert sum(len(b) for b in bs) == len(ds)


def test_stems_ignore_absent_roles_and_match_shape():
    import torch
    torch.manual_seed(0)
    m = stems.build("stems", ["lightloss", "fsc", "ssc"], 8, 3, 2).eval()
    x = torch.randn(2, 3, 32, 40)
    on = torch.tensor([[True, False, False]] * 2)
    xa = x.clone(); xa[:, 1:] = 0
    xb = x.clone(); xb[:, 1:] = 7.0
    with torch.no_grad():
        pa, pb = stems.call(m, xa, on), stems.call(m, xb, on)
        full = stems.call(m, x)
    assert pa.shape == (2, 2, 32, 40) and torch.allclose(pa, pb, atol=1e-5) and (full - pa).abs().mean() > 0
    s = stems.build("stack", ["a", "b", "c"], 8, 3, 2)
    assert stems.call(s, x).shape == (2, 2, 32, 40)


def test_native_size_evaluate_crops_to_object():
    import torch
    from sperm_fluorescence.eval.metrics import evaluate
    with tempfile.TemporaryDirectory() as d:
        make_cache(d, n=6)
        ds, wells, _ = open_cache(d, ["3S"])
        m = stems.build("stems", ds.roles, 4, ds.cin, ds.cout).eval()
        scale = torch.ones(1, ds.cout, 1, 1)
        res, preds = evaluate(m, ds, wells, scale, torch.device("cpu"), ["A", "B"], bs=4)
        for i, p in enumerate(preds):
            assert p.shape[1:] == ds.shape(i)
        assert set(res) == {"A", "B"} and all(v["event_r"] is not None for v in res.values())


if __name__ == "__main__":
    for fn in (test_cache_round_trip_and_role_selection, test_stems_ignore_absent_roles_and_match_shape, test_native_size_evaluate_crops_to_object):
        fn(); print("ok", fn.__name__)
