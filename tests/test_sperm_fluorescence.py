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


def test_predict_writes_a_store_that_indexes_like_the_cache():
    """An untrained stems model run through `predict`: the store has one prediction per cache
    object at the cache's frame size, masks inside the valid frame, and item(i) lines up."""
    import torch
    from sperm_fluorescence.data.store import open_store
    from sperm_fluorescence.predict import main as predict_main
    with tempfile.TemporaryDirectory() as d:
        cache, run, store = os.path.join(d, "cache"), os.path.join(d, "run"), os.path.join(d, "store")
        os.makedirs(cache); os.makedirs(run); make_cache(cache, n=7)
        model = stems.build("stems", ["lightloss", "fsc", "ssc"], 8, 3, 2)
        torch.save(model.state_dict(), os.path.join(run, "best.pt"))
        json.dump({"scale": [1.0, 2.0], "roles": ["lightloss", "fsc", "ssc"], "arch": "stems", "width": 8, "train": ["2S"], "test": ["3S"]},
                  open(os.path.join(run, "results.json"), "w"))
        assert predict_main([run, "--cache", cache, "--classes", "3S", "--out", store, "--batch", "4"]) == 0
        st = open_store(store)
        ds, _, _ = open_cache(cache, ["3S"])
        assert len(st) == len(ds) == 7 and st.names == ["A", "B"]
        for i in range(len(st)):
            p, pm, m, im = st.item(i)
            x, y, mc = ds.item(i)
            assert p.shape == y.shape and pm.shape == y.shape and m.shape == mc.shape and im is None
            assert np.array_equal(m, mc) and not (pm & ~m[None]).any()
        meta = json.load(open(os.path.join(store, "meta.json")))
        assert meta["classes"]["3S"]["n"] == 7 and meta["thresholds"]["a8:A"] == ["rel", 0.15]
        # --limit trims the copied index to what was written
        assert predict_main([run, "--cache", cache, "--classes", "3S", "--out", store + "2", "--limit", "3", "--threshold", "B=rel:0.3"]) == 0
        st2 = open_store(store + "2"); assert len(st2) == 3 and st2.meta["thresholds"]["a8:B"] == ["rel", 0.3]
        p2, _, _, _ = st2.item(2); p0, _, _, _ = st.item(2); assert np.allclose(p2, p0, atol=1e-2)




def test_feature_engine_on_a_drawn_sperm():
    """A synthetic sperm (elliptical nucleus, acrosomal cap, a short midpiece, a curved tail)
    through the compartment features: masks derive, lengths come out in the right units, the
    acrosome sits ahead of the nucleus and the tail path is longer than its chord."""
    from sperm_fluorescence.features.engine import compartment_features, qc_features
    px = 0.333; h, w = 96, 64; yy, xx = np.mgrid[:h, :w]
    nucleus = ((yy - 30) / 7.0) ** 2 + ((xx - 32) / 4.5) ** 2 <= 1          # ~4.7 x 3 um
    acro = (((yy - 24) / 5.0) ** 2 + ((xx - 32) / 4.5) ** 2 <= 1) & ~nucleus    # cap on the anterior pole (smaller row)
    mid = (yy >= 37) & (yy <= 49) & (np.abs(xx - 32) <= 1)                 # 13 px = 4.3 um
    t = np.linspace(0, 1, 400); ty = 50 + 40 * t; tx = 32 + 12 * np.sin(2 * np.pi * t)
    tail = np.zeros((h, w), bool); tail[np.clip(ty.astype(int), 0, h - 1), np.clip(tx.astype(int), 0, w - 1)] = True
    tail = __import__("scipy.ndimage", fromlist=["binary_dilation"]).binary_dilation(tail, iterations=1)
    valid = np.ones((h, w), bool)
    parts = {"nucleus": nucleus, "acrosome": acro, "midpiece": mid, "flagellum": tail | mid}
    rng = np.random.default_rng(0)
    chans = {"DAPI": nucleus * 5.0 + rng.normal(0, 0.1, (h, w)), "ACRV1": acro * 3.0, "TOMM20": mid * 4.0, "LDHC/AKAP4": (tail | mid) * 2.0}
    bf = -3.0 * (nucleus | acro | mid | tail).astype(np.float32) + rng.normal(0, 0.3, (h, w))
    docs = {}
    row, M, S, K = compartment_features("pred", parts, chans, {"brightfield": bf}, valid, px, docs)
    assert set(M) >= {"nucleus", "acrosome", "head", "midpiece", "principal", "tail", "cell"}
    assert 4.0 < row["pred.nucleus.feret_max_um"] < 5.5 and 2.5 < row["pred.nucleus.feret_min_um"] < 3.6
    assert 3.0 < row["pred.midpiece.skel.path_um"] < 5.0      # 13 px drawn; skeletonising trims an end pixel each side
    assert row["pred.tail.skel.tortuosity"] > 1.3 and row["pred.tail.skel.path_um"] > row["pred.tail.skel.end_to_end_um"]
    assert row["pred.x.acrosome_anterior_um"] < 0 and 0.2 < row["pred.x.acrosome_frac"] < 0.6
    assert row["pred.x.head_midpiece_angle_deg"] < 15 and row["pred.x.insertion_offset_um"] < 1.5
    assert row["pred.DAPI.nucleus.frac_in_mask"] > 0.9
    qc = qc_features(M, K, row, valid); assert qc["n_nuclei"] == 1 and qc["n_objects"] == 1
    assert all(c in docs for c in row)



def test_populate_recovers_planted_clusters():
    """A feature table with three planted populations in four head columns (plus noise
    columns and quality flags): the gates report their losses, PCA keeps a few components,
    k-means finds the three groups (ARI > 0.9 against the plant) and they are stable."""
    import csv
    from sklearn.metrics import adjusted_rand_score
    from sperm_fluorescence.population.engine import main as populate_main
    rng = np.random.default_rng(1); n = 600; plant = rng.integers(0, 3, n)
    centres = np.array([[4.0, 2.8, 8.5, 0.5], [5.2, 2.4, 9.6, 0.45], [3.6, 3.4, 9.0, 0.6]])
    head = centres[plant] + rng.normal(0, [0.12, 0.08, 0.3, 0.02], (n, 4))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "features.csv")
        cols = ["id.class", "id.index", "id.pop_pred", "qc.n_objects", "qc.n_nuclei", "qc.flagellum_edge_frac", "qc.focus_gradient_rms",
                "pred.head.feret_max_um", "pred.head.feret_min_um", "pred.head.area_um2", "pred.x.acrosome_frac", "pred.head.centroid_x", "pred.tail.skel.path_um", "pred.head.n_components"]
        with open(path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(cols)
            for i in range(n):
                w.writerow(["3I", i, "sperm" if i % 10 else "round", 1 if i % 25 else 2, 1, 0.0 if i % 40 else 0.2, rng.normal(10, 1),
                            *head[i], rng.uniform(0, 60), rng.normal(20, 5), 1])
        out = os.path.join(d, "pop")
        assert populate_main([path, "--set", "head", "--k", "3", "--within", "sperm", "--out", out, "--seed", "0"]) == 0
        summ = json.load(open(os.path.join(out, "summary.json")))
        assert [g["gate"] for g in summ["gates"]] == ["within", "single", "in_frame", "focus"] and summ["gates"][0]["lost"] == 60
        assert "pred.head.centroid_x" not in summ["columns"] and "pred.head.n_components" in summ["dropped"]
        assert summ["cluster_sizes"] and len(summ["cluster_sizes"]) == 3
        import pandas as pd
        tab = pd.read_csv(os.path.join(out, "populations.csv")); ok = tab["gate.passed"].to_numpy(bool)
        assert adjusted_rand_score(plant[ok], tab["cluster"].to_numpy()[ok]) > 0.9
        assert summ["stability"]["bootstrap_ari_mean"] > 0.8 and all(v > 0.8 for v in summ["stability"]["halves_ari"])
        assert os.path.exists(os.path.join(out, "embedding.png"))

if __name__ == "__main__":
    for fn in (test_cache_round_trip_and_role_selection, test_stems_ignore_absent_roles_and_match_shape, test_native_size_evaluate_crops_to_object,
               test_predict_writes_a_store_that_indexes_like_the_cache, test_feature_engine_on_a_drawn_sperm,
               test_populate_recovers_planted_clusters):
        fn(); print("ok", fn.__name__)
