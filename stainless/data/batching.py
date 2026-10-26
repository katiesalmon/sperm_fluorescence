"""Datasets over the cache arrays, and how batches are formed from them."""
import numpy as np


def avail_of(ds, idx):
    import torch
    if getattr(ds, "avail", None) is None:
        return torch.ones(len(idx), ds.cout)
    return torch.tensor(ds.avail[np.asarray(idx)].astype(np.float32))


class Cached:
    """Fixed-frame cache: index several (N, C, H, W) memmaps as one dataset."""
    ragged = False
    avail = None

    def __init__(self, xs, ys, ms):
        self.xs, self.ys, self.ms = xs, ys, ms
        self.offsets = np.cumsum([0] + [len(x) for x in xs])
        self.cin, self.cout = xs[0].shape[1], ys[0].shape[1]
    def __len__(self): return int(self.offsets[-1])
    def shape(self, i):
        k = np.searchsorted(self.offsets, i, side="right") - 1
        return tuple(int(v) for v in self.xs[k].shape[2:])
    def sample_idx(self, n):
        return np.unique(np.linspace(0, len(self) - 1, n).astype(int))
    def sample_targets(self, n):
        return np.concatenate([np.asarray(y[:n]) for y in self.ys]).astype(np.float32)
    def batch(self, idx):
        import torch
        out = [[], [], []]
        for i in idx:
            k = np.searchsorted(self.offsets, i, side="right") - 1; j = i - self.offsets[k]
            out[0].append(self.xs[k][j]); out[1].append(self.ys[k][j]); out[2].append(self.ms[k][j])
        return (torch.tensor(np.stack(out[0]).astype(np.float32)), torch.tensor(np.stack(out[1]).astype(np.float32)),
                torch.tensor(np.stack(out[2])[:, None].astype(np.float32)))


class RaggedCached:
    """Native-size cache: every object keeps its own (h, w). A batch is padded to the largest
    shape in it, so batches are assembled from objects of like shape (make_batches).
    Nothing is cropped at any point, so evaluation is on whole objects."""
    ragged = True
    avail = None

    def __init__(self, xs, ys, ms, extra, target_k=None):
        self.xs, self.ys, self.ms, self.target_k = xs, ys, ms, target_k
        self.shapes = np.concatenate([e[0] for e in extra]); self.per_class = extra
        self.counts = np.cumsum([0] + [len(e[0]) for e in extra])
        self.cin = xs[0].shape[0]; self.cout = 1 if target_k is not None else ys[0].shape[0]
    def __len__(self): return int(self.counts[-1])
    def shape(self, i): return tuple(int(v) for v in self.shapes[i])
    def item(self, i):
        k = np.searchsorted(self.counts, i, side="right") - 1; j = i - self.counts[k]
        shapes, offsets = self.per_class[k]; h, w = shapes[j]; a, b = offsets[j], offsets[j + 1]
        x = np.asarray(self.xs[k][:, a:b]).reshape(self.cin, h, w).astype(np.float32)
        y = np.asarray(self.ys[k][:, a:b]).reshape(-1, h, w).astype(np.float32)
        if self.target_k is not None: y = y[self.target_k:self.target_k + 1]
        return x, y, np.asarray(self.ms[k][a:b]).reshape(h, w)
    def sample_idx(self, n):
        return np.unique(np.linspace(0, len(self) - 1, n).astype(int))   # spread over every class
    def sample_targets(self, n):
        flat = np.concatenate([self.item(i)[1].reshape(self.cout, -1) for i in self.sample_idx(n)], axis=1)
        return flat[None, :, :, None]   # (1, C, pixels, 1): same reduction axes as a fixed frame
    def batch(self, idx):
        import torch
        items = [self.item(i) for i in idx]
        H = max(x.shape[1] for x, _, _ in items); W = max(x.shape[2] for x, _, _ in items); n = len(items)
        X = np.zeros((n, self.cin, H, W), np.float32); Y = np.zeros((n, self.cout, H, W), np.float32); M = np.zeros((n, 1, H, W), np.float32)
        for q, (x, y, m) in enumerate(items):
            h, w = x.shape[1:]; t, l = (H - h) // 2, (W - w) // 2
            X[q, :, t:t + h, l:l + w] = x; Y[q, :, t:t + h, l:l + w] = y; M[q, 0, t:t + h, l:l + w] = m
        return torch.tensor(X), torch.tensor(Y), torch.tensor(M)


def make_batches(ds, idx, batch_size, rng=None):
    """Batches of indices. Fixed-frame: random chunks. Ragged: sort by shape -- with a
    little jitter when training, so size-neighbours mix across epochs -- so each batch
    pads almost nothing, then shuffle batch order."""
    idx = np.asarray(idx)
    if not ds.ragged:
        if rng is not None: idx = rng.permutation(idx)
        return [idx[i:i + batch_size] for i in range(0, len(idx), batch_size)]
    key = ds.shapes[idx].astype(np.float64); key = key[:, 0] * 1000 + key[:, 1]
    if rng is not None: key = key + rng.uniform(0, 12, len(key))
    order = idx[np.argsort(key, kind="stable")]
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if rng is not None: rng.shuffle(batches)
    return batches

