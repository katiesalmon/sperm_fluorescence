"""Read BD FACSDiscover A8 CellView event images and their targets from an A8 bundle.

The one thing worth knowing before touching these files: each six-page TIFF carries a
per-page `{"shape": [h, w]}` description, which makes `tifffile.imread` treat every page
as a separate series and return page 1 alone, silently. Iterate the pages.

Pages, in order (verified against the FCS on the server, docs/task_brief.md):

    0  LightLoss (Imaging)           label-free; extinction -- cells are DARKER than background
    1  FSC                           label-free
    2  SSC (Imaging)                 label-free
    3  Imaging_BP/534/46/LP/505      AF488  -> LDHC_AKAP4
    4  Imaging_BP/598/60/LP/570      PE     -> CD45
    5  Imaging_BP/788/225/LP/675     PerCP-eF710 -> ACRV1

Fluorescence pages are already background-subtracted (zero-centred, negative noise).
The three fluorescence pages are RAW filter channels, not unmixed; see mixing_matrix().
"""

import csv
import io
import json
import zipfile

import numpy as np
import tifffile

PAGES = ["LightLoss", "FSC", "SSC", "AF488_LDHC", "PE_CD45", "PerCP_ACRV1"]
LABEL_FREE = [0, 1, 2]
FLUOR = [3, 4, 5]
# Unmixed scalar columns in targets.csv, in the same order as the fluorescence pages.
SCALARS = ["LDHC_AKAP4-A", "CD45-A", "ACRV-1-A"]


def read_stack(data):
    """(6, h, w) float32 plus the page names, from the bytes of one event's TIFF."""
    with tifffile.TiffFile(io.BytesIO(data)) as tf:
        pages = [p.asarray() for p in tf.pages]
        names = [(p.tags["PageName"].value if "PageName" in p.tags else "") for p in tf.pages]
    return np.stack(pages).astype(np.float32), names


class Bundle:
    """An a8_bundle_*.zip: per-class targets and images, keyed by FCS row."""

    def __init__(self, path):
        self.zf = zipfile.ZipFile(path)
        self.manifest = json.loads(self.zf.read("MANIFEST.json"))
        self.targets = {}
        for cls, entry in self.manifest["classes"].items():
            if not entry.get("rows"):
                continue
            text = self.zf.read("%s/targets.csv" % cls).decode("utf-8")
            rows = list(csv.DictReader(io.StringIO(text)))
            self.targets[cls] = {int(r["row"]): r for r in rows}

    @property
    def classes(self):
        return sorted(self.targets)

    def rows(self, cls):
        return sorted(self.targets[cls])

    def image(self, cls, row):
        return read_stack(self.zf.read("%s/images/%08d.tiff" % (cls, row)))[0]

    def scalars(self, cls, row, names=SCALARS):
        t = self.targets[cls][row]
        return np.array([float(t[n]) for n in names], dtype=np.float64)

    def events(self, classes=None):
        for cls in (classes or self.classes):
            for row in self.rows(cls):
                yield cls, row


def spillover(bundle):
    """Estimate the imaging-channel spillover under the one constraint physics supplies.

    Emission only spills red-ward: AF488 (534 nm filter) -> PE (598) -> PerCP-eF710 (788),
    never back. So the mixing is lower-triangular with three free coefficients, and each
    is estimable from a well where the upstream fluorophore dominates -- no single-stain
    control needed, which is good because the experiment has none.

    An unconstrained regression on the unmixed scalars does not work here: the scalars
    are collinear by cell type (sperm carry AF488 and PerCP together, PBMCs carry PE), so
    it returns well-fitting but physically impossible coefficients.

    Returns (M, fits) with raw_pages ~= M @ true_pages, M = [[1,0,0],[a,1,0],[b,c,1]].
    """
    def sums(classes):
        return np.array([bundle.image(c, r)[FLUOR].sum(axis=(1, 2)) for c, r in bundle.events(classes)])

    def slope(y, x):
        b = np.linalg.lstsq(np.c_[x, np.ones(len(x))], y, rcond=None)[0][0]
        return b, np.corrcoef(x, y)[0, 1]

    S = sums(["2S", "3S"])            # AF488-dominant; sperm carry no CD45
    P = sums(["2P", "3P"])            # PE-dominant; PBMCs carry no acrosome
    a, ra = slope(S[:, 1], S[:, 0])   # AF488 -> PE
    b, rb = slope(S[:, 2], S[:, 0])   # AF488 -> PerCP
    # PE -> PerCP: remove AF488's contribution first (PBMCs do bind the LDHC antibody).
    c, rc = slope(P[:, 2] - b * P[:, 0], P[:, 1])
    M = np.array([[1.0, 0.0, 0.0], [a, 1.0, 0.0], [b, c, 1.0]])
    return M, {"AF488->PE": (a, ra), "AF488->PerCP": (b, rb), "PE->PerCP": (c, rc)}


def unmix(stack, M):
    """Apply inv(M) pixel-wise to the three fluorescence pages of a (6, h, w) stack."""
    f = stack[FLUOR].reshape(3, -1)
    out = stack.copy()
    out[FLUOR] = (np.linalg.inv(M) @ f).reshape(3, *stack.shape[1:])
    return out
