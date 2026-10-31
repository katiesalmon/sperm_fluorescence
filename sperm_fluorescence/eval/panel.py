"""Submission figure: one row per held-out object, the label-free inputs, then for every
marker the measured stain beside the prediction, all on white with a legend written next
to it.

    python -m sperm_fluorescence panel runs/srv_isx_stems --cache data/isx_roles --test 3 \\
        --rows "best:2,typical:2,dim:1" --um-per-px 0.333 --out figures/aua_isx_panel.png
        ... --rows best:12 --exclude 6709,9794 --out candidates.png      # a sheet to choose from by eye
        ... --objects 766,2395,640,1234,5678 --out final.png           # then pin the chosen objects in row order

Rows are chosen by rule, not by eye. `best:n`: sperm (ISX: by the DAPI scalar; A8: the S
well) stained above the median on every marker, ranked by mean per-object pixel r, top n.
`typical:n`: sperm whose mean pixel r is closest to the sperm median. With a pixel size and
DAPI + LDHC/AKAP4 among the markers, a sperm row must also look like one sperm (nucleus
< 30 um^2 at half maximum, flagellum stain centred >= 2 um from it). Any other name is a
population from `accuracy.groups_for` (ISX: dim, round; A8: S, SP, P) sampled at random. An
object whose stain touches the frame edge is never chosen. Each marker is drawn on one
intensity range for the whole figure (the 99.5th percentile of the measured stain over the
chosen objects), so a dim fragment looks dim. Writes the PNG, a .txt legend and a .json of
the chosen objects. An all-marker overlay pair (measured, predicted) closes each row unless
--no-overlay.
"""
import argparse
import json
import os

import numpy as np

from . import render
from .accuracy import groups_for
from .load import load_run
from .metrics import evaluate
from .slides import crop, window

ROLE_LABEL = {"lightloss": "Light loss", "fsc": "FSC", "ssc": "SSC", "brightfield": "Brightfield 1", "brightfield2": "Brightfield 2", "darkfield": "Darkfield (SSC)"}
WHERE = {"LDHC/AKAP4": "Flagellum", "LDHC": "Flagellum", "TOMM20": "Midpiece", "DAPI": "Nucleus", "ACRV1": "Acrosome", "CD45": "Leukocytes"}
DISPLAY = {"LDHC/AKAP4": "LDHC"}          # the marker name as the figure prints it (the antibody cocktail stays LDHC/AKAP4 in the data)
GROUP_LABEL = {"best": "Sperm,\nbest cases", "typical": "Sperm,\ntypical", "dim": "Debris /\nDAPI-dim", "round": "Round\ncells",
               "S": "Sperm well", "SP": "Mixed well", "P": "PBMC well"}


def per_object(ds, P, idx, av):
    """Mean per-object pixel r over the stained markers, per-marker event sums, and an
    in-frame flag (stain within 3 px of the border < 4 % of the total)."""
    r = np.full((len(idx), ds.cout), np.nan); ev = np.full((len(idx), ds.cout), np.nan); inside = np.ones(len(idx), bool)
    for q, i in enumerate(idx):
        _, y, m = crop(ds, i); tot = np.abs(y).sum(axis=0) * m; b = 3
        edge = tot[:b].sum() + tot[-b:].sum() + tot[b:-b, :b].sum() + tot[b:-b, -b:].sum()
        inside[q] = edge <= 0.04 * max(tot.sum(), 1e-9)
        for k in range(ds.cout):
            if not av[q, k]: continue
            ev[q, k] = (y[k] * m).sum(); a, b_ = P[q][k][m > 0], y[k][m > 0]
            r[q, k] = np.corrcoef(a, b_)[0, 1] if a.std() > 0 and b_.std() > 0 else 0.0
    return r, ev, inside


def sperm_shaped(ds, idx, names, av, um_per_px):
    """True where the object looks like a single sperm, when the markers allow the test: the
    nucleus (DAPI above half its maximum) covers < 30 um^2 and the flagellum stain
    (LDHC/AKAP4) is centred >= 2 um from it. Without DAPI or a flagellum marker, or without
    a pixel size, every object passes."""
    ok = np.ones(len(idx), bool)
    flag = next((n for n in ("LDHC/AKAP4", "LDHC") if n in names), None)
    if not um_per_px or "DAPI" not in names or flag is None: return ok
    kd, kf = names.index("DAPI"), names.index(flag)
    for q, i in enumerate(idx):
        if not (av[q, kd] and av[q, kf]): continue
        _, y, m = crop(ds, i); d = y[kd] * m; f = np.clip(y[kf] * m, 0, None)
        nuc = d > 0.5 * d.max()
        if nuc.sum() * um_per_px ** 2 > 30 or d.max() <= 0 or f.sum() <= 0: ok[q] = False; continue
        yy, xx = np.indices(m.shape)
        cd = (yy[nuc].mean(), xx[nuc].mean()); cf = ((yy * f).sum() / f.sum(), (xx * f).sum() / f.sum())
        ok[q] = np.hypot(cd[0] - cf[0], cd[1] - cf[1]) * um_per_px >= 2.0
    return ok


def choose(spec, g, r, ev, inside, main, rng, shaped=None, use=None):
    """[(group name, q)] in row order, by the rule in the module docstring. `use`: the
    marker columns that count for "stained above the median" and for the ranking."""
    use = list(range(r.shape[1])) if use is None else list(use)
    score = np.nanmean(r[:, use], axis=1); taken = set(); rows = []
    sperm = (g == main) & inside & np.isfinite(score)
    if shaped is not None: sperm &= shaped
    med = np.nanmedian(np.where(sperm[:, None], ev[:, use], np.nan), axis=0)
    well = sperm & np.all(np.isnan(ev[:, use]) | (ev[:, use] > med), axis=1)
    for name, n in spec:
        if name == "best":
            order = [q for q in np.argsort(-score) if well[q] and q not in taken][:n]
        elif name == "typical":
            m_ = np.nanmedian(score[sperm]); order = [q for q in np.argsort(np.abs(score - m_)) if sperm[q] and q not in taken][:n]
        else:
            cand = [q for q in np.flatnonzero((g == name) & inside) if q not in taken]
            if not cand: raise SystemExit("no in-frame objects in population %r (have %s)" % (name, sorted(set(g))))
            order = [int(q) for q in rng.choice(cand, min(n, len(cand)), replace=False)]
        taken.update(order); rows += [(name, int(q)) for q in order]
    return rows


def pad_to(a, size, fill):
    """Centre (C, h, w) or (h, w) in a size x size frame."""
    h, w = a.shape[-2:]; t, l = (size - h) // 2, (size - w) // 2
    out = np.full(a.shape[:-2] + (size, size), fill, dtype=a.dtype) if np.isscalar(fill) else np.empty(a.shape[:-2] + (size, size), a.dtype)
    if not np.isscalar(fill):
        for c in range(a.shape[0]): out[c] = fill[c]
    out[..., t:t + h, l:l + w] = a
    return out


def draw(path, rows, objs, roles, names, ks, rng_, um_per_px, title, size, overlay=True, approx=False):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib import patches
    groups = []
    for name, _ in rows:
        if not groups or groups[-1][0] != name: groups.append([name, 0])
        groups[-1][1] += 1
    gap = 0.22
    widths = [1.0] * len(roles) + [gap * 1.4]
    for j in range(len(ks)): widths += [1.0, 1.0] + ([gap] if j < len(ks) - 1 or overlay else [])
    if overlay: widths += [1.0, 1.0]
    heights = []
    for gi, (_, n) in enumerate(groups): heights += [1.0] * n + ([gap] if gi < len(groups) - 1 else [])
    pw = 0.95; left, right, top, bottom = (0.95 if len(groups) > 1 else 0.12), 0.12, (0.78 if title else 0.6), 0.12
    figw = pw * sum(widths) + left + right; figh = pw * sum(heights) + top + bottom
    fig = plt.figure(figsize=(figw, figh), facecolor="white")
    gs = fig.add_gridspec(len(heights), len(widths), width_ratios=widths, height_ratios=heights,
                          left=left / figw, right=1 - right / figw, top=1 - top / figh, bottom=bottom / figh, wspace=0.05, hspace=0.05)
    col_of = {}; c = 0
    for r_i in range(len(roles)): col_of[("in", r_i)] = c; c += 1
    c += 1
    for j, k in enumerate(ks):
        col_of[("t", k)] = c; col_of[("p", k)] = c + 1; c += 3
    if overlay: col_of[("t", "all")] = c; col_of[("p", "all")] = c + 1
    row_of = []; rr = 0
    for gi, (_, n) in enumerate(groups):
        for _ in range(n): row_of.append(rr); rr += 1
        rr += 1
    axes = {}
    for i, ((name, q), (x, y, m, p, r)) in enumerate(zip(rows, objs)):
        R = row_of[i]
        for r_i in range(len(roles)):
            ax = fig.add_subplot(gs[R, col_of[("in", r_i)]]); ax.imshow(render.grey(x[r_i]), interpolation="nearest"); axes[(R, col_of[("in", r_i)])] = ax
        tl, pl = [], []
        for k in ks:
            col = render.color_for(names[k], k); lo, hi = rng_[k]
            tl.append(render.colorize(y[k], col, lo, hi)); pl.append(render.colorize(np.clip(p[k], 0, None), col, lo, hi))
            ax = fig.add_subplot(gs[R, col_of[("t", k)]]); ax.imshow(tl[-1], interpolation="nearest"); axes[(R, col_of[("t", k)])] = ax
            ax = fig.add_subplot(gs[R, col_of[("p", k)]]); ax.imshow(pl[-1], interpolation="nearest"); axes[(R, col_of[("p", k)])] = ax
            if np.isfinite(r[k]):
                ax.text(0.04, 0.04, "r %.2f" % r[k], transform=ax.transAxes, color="white", fontsize=5.2, va="bottom", ha="left")
        if overlay:
            ax = fig.add_subplot(gs[R, col_of[("t", "all")]]); ax.imshow(render.composite(tl), interpolation="nearest"); axes[(R, col_of[("t", "all")])] = ax
            ax = fig.add_subplot(gs[R, col_of[("p", "all")]]); ax.imshow(render.composite(pl), interpolation="nearest"); axes[(R, col_of[("p", "all")])] = ax
        if i == 0 and um_per_px:
            ax = axes[(R, 0)]; px = 5.0 / um_per_px
            ink = "black" if render.grey(x[0])[-12:, -int(px) - 6:].mean() > 0.5 else "white"     # light loss is a dark cell on a bright field
            ax.add_patch(patches.Rectangle((size - px - 3, size - 5), px, 1.6, color=ink))
            ax.text(size - px / 2 - 3, size - 7, ("≈5 µm" if approx else "5 µm"), color=ink, fontsize=5.2, ha="center", va="bottom")
    for ax in axes.values():
        ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("black")
        for sp in ax.spines.values(): sp.set_edgecolor("#c8c8c8"); sp.set_linewidth(0.5)
    # column headers: the inputs, then one header per marker over its measured/predicted pair
    for r_i, role in enumerate(roles):
        ax = axes[(0, col_of[("in", r_i)])]; ax.set_title(ROLE_LABEL.get(role, role), fontsize=7, fontweight="bold", pad=3)
    for k in ks + (["all"] if overlay else []):
        at, ap_ = axes[(0, col_of[("t", k)])], axes[(0, col_of[("p", k)])]
        at.set_title("Measured", fontsize=6.2, color="#444444", pad=2); ap_.set_title("Predicted", fontsize=6.2, color="#444444", pad=2)
        bt, bp = at.get_position(), ap_.get_position(); xm = (bt.x0 + bp.x1) / 2
        col = tuple(0.7 * v for v in render.color_for(names[k], k)) if k != "all" else "black"
        fig.text(xm, bt.y1 + 0.17 / figh, DISPLAY.get(names[k], names[k]) if k != "all" else "All markers", fontsize=8, fontweight="bold", color=col, ha="center", va="bottom")
        fig.text(xm, bt.y1 + 0.17 / figh + 0.14 / figh, WHERE.get(names[k], "") if k != "all" else "Overlay", fontsize=6.2, color="#444444", ha="center", va="bottom")
    b0 = axes[(0, 0)].get_position(); b1 = axes[(0, col_of[("in", len(roles) - 1)])].get_position()
    fig.text((b0.x0 + b1.x1) / 2, b0.y1 + 0.17 / figh + 0.14 / figh, "Microscopy Imaging", fontsize=8, fontweight="bold", ha="center", va="bottom")
    # row-group labels (none when every row is the same group: the legend says what they are)
    i = 0
    for name, n in (groups if len(groups) > 1 else []):
        top_ax = axes[(row_of[i], 0)]; bot_ax = axes[(row_of[i + n - 1], 0)]
        bt, bb = top_ax.get_position(), bot_ax.get_position()
        fig.text(bt.x0 - 0.08 / figw, (bt.y1 + bb.y0) / 2, GROUP_LABEL.get(name, name), fontsize=7, fontweight="bold", ha="right", va="center", linespacing=1.1)
        i += n
    if title: fig.suptitle(title, fontsize=9, fontweight="bold", y=1 - 0.06 / figh, va="top")
    fig.savefig(path, dpi=300, facecolor="white"); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sperm_fluorescence panel", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run"); ap.add_argument("--cache", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--include", default=None); ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--denoise-measured", default="none", choices=["none", "floor", "smooth"], help="read the measured stains denoised (noise floor removed; smooth = 1-px Gaussian first), so the prediction is scored and shown against the stain without its read noise"); ap.add_argument("--tta", action="store_true")
    ap.add_argument("--rows", default="best:2,typical:2,dim:1", help="comma list of group:n in row order")
    ap.add_argument("--main", default=None, help="the sperm population for best/typical (default: 'sperm' if present, else 'S')")
    ap.add_argument("--sample", type=int, default=1500, help="held-out objects scored before choosing"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=72, help="window side in px around each object")
    ap.add_argument("--um-per-px", type=float, default=0.0); ap.add_argument("--title", default="")
    ap.add_argument("--no-overlay", action="store_true", help="omit the all-marker overlay pair at the right")
    ap.add_argument("--stained", default=None, help="comma list of markers that count for 'stained above the median' and the ranking (default: all; on the A8 leave CD45 out)")
    ap.add_argument("--approx-scale", action="store_true", help="prefix the scale bar with ~ (A8: the pixel size is approximate)")
    ap.add_argument("--range", default="rows", choices=["rows", "sample"], help="display range per marker from the rows shown (default) or from the whole scored sample (A8: so CD45, absent on sperm, is scaled by the leukocytes that carry it)")
    ap.add_argument("--objects", default=None, help="comma list of held-out object ids to show, in row order, instead of the rule (chosen by eye from a candidates panel); the legend says so")
    ap.add_argument("--exclude", default=None, help="comma list of object ids the rule must not pick")
    ap.add_argument("--marker-order", default=None, help="comma list of marker names, the column order left to right (default: the model's order), e.g. ACRV1,DAPI,TOMM20,LDHC/AKAP4")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    R = load_run(a.run, a.cache, a.test, a.include, a.checkpoint, denoise=a.denoise_measured); te, wte, names, roles = R.te, R.wte, R.names, list(R.te.roles)
    rng = np.random.default_rng(a.seed)
    idx = np.sort(rng.choice(len(te), min(a.sample, len(te)), replace=False))
    pinned = [int(v) for v in a.objects.split(",")] if a.objects else []
    if pinned: idx = np.sort(np.union1d(idx, np.array(pinned)))                   # pinned objects join the sample so they are scored and ranged like the rest
    excluded = {int(v) for v in a.exclude.split(",")} if a.exclude else set()
    _, P = evaluate(R.model, te, wte[idx], R.scale, R.dev, names, idx=idx, tta=a.tta)
    av = te.avail[idx] if te.avail is not None else np.ones((len(idx), te.cout), bool)
    g = groups_for(te, wte, idx).astype(str)
    main_ = a.main or ("sperm" if (g == "sperm").any() else "S")
    r, ev, inside = per_object(te, P, idx, av)
    spec = [(s.split(":")[0].strip(), int(s.split(":")[1])) for s in a.rows.split(",") if s.strip()]
    use = [names.index(s.strip()) for s in a.stained.split(",")] if a.stained else None
    if pinned:
        pos = {int(i): q for q, i in enumerate(idx)}; rows = [("best", pos[o]) for o in pinned]
    else:
        keep = np.array([int(i) not in excluded for i in idx]); inside = inside & keep
        rows = choose(spec, g, r, ev, inside, main_, rng, sperm_shaped(te, idx, names, av, a.um_per_px), use)
    objs = []
    for name, q in rows:
        x, y, m = crop(te, idx[q]); x, y, m, p = window(x, y, m, P[q], a.size)
        fill = np.array([np.median(np.concatenate([x[c, 0], x[c, -1], x[c, :, 0], x[c, :, -1]])) for c in range(x.shape[0])], np.float32)
        objs.append((pad_to(x, a.size, fill), pad_to(y, a.size, 0.0), pad_to(m, a.size, 0.0), pad_to(p, a.size, 0.0), r[q]))
        print("%-8s object %6d  %s" % (name, idx[q], "  ".join("%s %.2f" % (names[k], r[q][k]) for k in range(te.cout) if av[q, k])))
    ks = [k for k in range(te.cout) if all(av[q, k] for _, q in rows)]
    if a.marker_order:
        want = [n.strip() for n in a.marker_order.split(",")]
        for n in want:
            if n not in names: raise SystemExit("--marker-order: %r is not a marker here (have %s)" % (n, names))
        ks = [names.index(n) for n in want if names.index(n) in ks] + [k for k in ks if names[k] not in want]
    if a.range == "sample":
        # per marker: the brightest quarter of the sampled objects by event sum, then the usual
        # 99.5th percentile of their pixels, so a marker absent on the rows shown (CD45 on sperm)
        # is scaled by the cells that carry it and reads as dark on the rows that do not
        sub = idx[:: max(1, len(idx) // 400)]; items = [crop(te, i) for i in sub]
        rng_ = []
        for k in range(te.cout):
            sums = np.array([(y[k] * m).sum() for _, y, m in items]); top = sums >= np.percentile(sums, 75)
            rng_.append(render.ranges([y for (_, y, m), t_ in zip(items, top) if t_], [m for (_, y, m), t_ in zip(items, top) if t_])[k])
    else:
        rng_ = render.ranges([o[1] for o in objs], [o[2] for o in objs])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    draw(a.out, rows, objs, roles, names, ks, rng_, a.um_per_px, a.title, a.size, overlay=not a.no_overlay and len(ks) > 1, approx=a.approx_scale)
    stem = os.path.splitext(a.out)[0]
    json.dump({"run": a.run, "checkpoint": a.checkpoint, "sample": int(len(idx)), "seed": a.seed, "size_px": a.size, "main_population": main_,
               "rows": [{"group": name, "object": int(idx[q]), "pixel_r": {names[k]: (float(r[q][k]) if np.isfinite(r[q][k]) else None) for k in range(te.cout)}} for name, q in rows],
               "display_range_max": {names[k]: rng_[k][1] for k in ks}}, open(stem + ".json", "w"), indent=1)
    if pinned: spec = []                                          # the legend describes the pinned rows, not the unused rule
    n_best = sum(n for s, n in spec if s == "best"); n_typ = sum(n for s, n in spec if s == "typical")
    others = ", ".join("%d %s" % (n, GROUP_LABEL.get(s, s).replace("\n", " ").lower()) for s, n in spec if s not in ("best", "typical"))
    parts = []
    on = "every marker" if not a.stained else ("%s" % " and ".join(s.strip() for s in a.stained.split(",")))
    if pinned: parts.append("%d sperm chosen by eye from the highest-ranked candidates (per-object pixel correlation among sperm stained above the median on %s)" % (len(pinned), on))
    elif n_best: parts.append("%d sperm with the highest per-object pixel correlation among sperm stained above the median on %s" % (n_best, on))
    if n_typ: parts.append("%d sperm closest to the median correlation" % n_typ)
    if others: parts.append(others + " chosen at random")
    legend = ("Held-out objects, one per row: %s label-free inputs (left) and, for each marker, the measured fluorescence beside the "
              "image predicted from the inputs alone%s. Rows were chosen %sfrom %d randomly sampled held-out objects: %s"
              "; objects whose stain touches the frame edge were excluded%s. Each marker is shown on one intensity range across all "
              "rows (black = 0, full colour = the 99.5th percentile of the measured stain over %s). r = per-object pixel correlation "
              "between predicted and measured images within the cell. Scale bar %s5 µm." % (" + ".join(ROLE_LABEL.get(x, x).lower() for x in roles), "" if (a.no_overlay or len(ks) < 2) else ", with all markers overlaid at the right", "" if pinned else "by rule ", len(idx), "; ".join(parts),
                 ", and sperm rows were required to show one nucleus under 30 \u00b5m\u00b2 with the flagellum stain centred at least 2 \u00b5m from it" if (a.um_per_px and "DAPI" in names and "LDHC/AKAP4" in names) else "", "the objects shown" if a.range == "rows" else "the brightest quarter of the sampled held-out objects, of every population, for that marker", "approximately " if a.approx_scale else ""))
    if a.denoise_measured != "none": legend += " Measured stains are shown and scored after noise-floor removal (%s)." % ("1-px Gaussian, then the frame's noise floor subtracted" if a.denoise_measured == "smooth" else "the frame's noise floor subtracted")
    open(stem + ".txt", "w", encoding="utf-8").write(legend + "\n")
    print("wrote", a.out, "(+ .txt legend, .json picks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
