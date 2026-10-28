"""Skeleton geometry of one mask: the longest path through its skeleton, its length,
straightness, curvature profile, bends, branches, and whether it coils. Built for tails and
midpieces; the definitions are generic."""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

N_PROFILE = 16
DOC = {
    "skel_px": "skeleton pixels", "n_endpoints": "skeleton pixels with one neighbour", "n_branchpoints": "skeleton pixels with three or more neighbours",
    "path_um": "length of the longest skeleton path (sum of pixel steps, diagonal = sqrt 2), um",
    "end_to_end_um": "straight distance between the path's ends, um", "tortuosity": "path_um / end_to_end_um",
    "turning_total_deg": "sum of absolute turning angles along the smoothed path, degrees (360 = one full coil)",
    "curv_max_per_um": "largest absolute curvature along the smoothed path, 1/um", "curv_mean_per_um": "mean absolute curvature, 1/um",
    "n_sharp_bends": "turns of more than 60 degrees within 1.5 um of path",
    "self_proximity_um": "smallest distance between path points more than 4 um apart along the path, um (small = folded or coiled)",
    "coiled": "1 if turning_total_deg > 360", "path_frac": "path pixels / skeleton pixels (1 = no side branches)",
    "end_y": "row of the path end nearest the frame edge", "end_x": "column of that end",
}
for i in range(N_PROFILE): DOC["curv_%02d" % i] = "signed curvature at %d/%d of the path, 1/um" % (i, N_PROFILE)
NAN = {k: np.nan for k in DOC}
OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _neighbours(sk):
    k = np.ones((3, 3)); k[1, 1] = 0
    return ndi.convolve(sk.astype(np.int16), k, mode="constant") * sk


def _longest_path(sk):
    """Farthest-pair path on the skeleton pixel graph (two BFS passes), as an ordered list of (y, x)."""
    pts = list(zip(*np.nonzero(sk))); idx = {p: i for i, p in enumerate(pts)}
    if not pts: return []
    adj = [[] for _ in pts]
    for p, i in idx.items():
        for dy, dx in OFFS:
            q = (p[0] + dy, p[1] + dx)
            if q in idx: adj[i].append((idx[q], np.hypot(dy, dx)))
    def bfs(s):
        dist = np.full(len(pts), np.inf); prev = np.full(len(pts), -1); dist[s] = 0; order = [s]
        for u in order:
            for v, w in adj[u]:
                if dist[u] + w < dist[v]: dist[v] = dist[u] + w; prev[v] = u; order.append(v)
        return dist, prev
    d0, _ = bfs(0); a = int(np.nanargmax(np.where(np.isfinite(d0), d0, -1)))
    d1, prev = bfs(a); b = int(np.nanargmax(np.where(np.isfinite(d1), d1, -1)))
    path = []; u = b
    while u != -1: path.append(pts[u]); u = int(prev[u])
    return path[::-1]


def skeleton_features(mask, px_um, frame_shape=None):
    """dict of DOC's keys; NaNs for a mask too small to skeletonise."""
    if mask.sum() < 4: return dict(NAN, skel_px=0, n_endpoints=0, n_branchpoints=0)
    sk = skeletonize(mask)
    nb = _neighbours(sk)
    out = {"skel_px": int(sk.sum()), "n_endpoints": int((nb == 1).sum()), "n_branchpoints": int((nb >= 3).sum())}
    path = _longest_path(sk)
    if len(path) < 3: return dict(NAN, **out)
    P = np.array(path, np.float64)
    steps = np.hypot(*np.diff(P, axis=0).T); s = np.concatenate([[0], np.cumsum(steps)]); L = float(s[-1])
    out["path_um"] = L * px_um; out["end_to_end_um"] = float(np.hypot(*(P[-1] - P[0]))) * px_um
    out["tortuosity"] = L / max(np.hypot(*(P[-1] - P[0])), 1e-6); out["path_frac"] = len(path) / float(sk.sum())
    # resample by arc length, smooth over ~1 um, then turning angles
    n = max(int(round(L / 1.0)), 8); t = np.linspace(0, L, n)
    R = np.stack([np.interp(t, s, P[:, 0]), np.interp(t, s, P[:, 1])], 1)
    w = max(int(round(1.0 / px_um / (L / (n - 1)))), 1)
    if w > 1: R = ndi.uniform_filter1d(R, size=w, axis=0, mode="nearest")
    d = np.diff(R, axis=0); ang = np.arctan2(d[:, 1], d[:, 0]); turn = np.diff(np.unwrap(ang))
    seg = np.hypot(*d.T); ds_um = seg[1:] * px_um
    curv = turn / np.maximum(ds_um, 1e-6)
    out["turning_total_deg"] = float(np.rad2deg(np.abs(turn).sum()))
    out["curv_max_per_um"] = float(np.abs(curv).max()) if len(curv) else np.nan; out["curv_mean_per_um"] = float(np.abs(curv).mean()) if len(curv) else np.nan
    win = max(int(round(1.5 / (L * px_um / max(n - 1, 1)))), 1)
    cum = np.convolve(turn, np.ones(win), mode="valid") if len(turn) >= win else turn
    out["n_sharp_bends"] = int((np.abs(cum) > np.deg2rad(60)).sum())
    out["coiled"] = int(out["turning_total_deg"] > 360)
    # self proximity: closest approach between points far apart along the path
    if L * px_um > 8:
        tt = t * px_um; D = np.hypot(R[:, None, 0] - R[None, :, 0], R[:, None, 1] - R[None, :, 1]) * px_um
        far = np.abs(tt[:, None] - tt[None, :]) > 4.0
        out["self_proximity_um"] = float(D[far].min()) if far.any() else np.nan
    else:
        out["self_proximity_um"] = np.nan
    prof = np.interp(np.linspace(0, 1, N_PROFILE), np.linspace(0, 1, len(curv)), curv) if len(curv) >= 2 else np.full(N_PROFILE, np.nan)
    for i in range(N_PROFILE): out["curv_%02d" % i] = float(prof[i])
    if frame_shape is not None:
        h, w_ = frame_shape; ends = [P[0], P[-1]]
        dist = [min(e[0], h - 1 - e[0], e[1], w_ - 1 - e[1]) for e in ends]; e = ends[int(np.argmin(dist))]
        out["end_y"] = float(e[0]); out["end_x"] = float(e[1])
    else:
        out["end_y"] = out["end_x"] = np.nan
    return out
