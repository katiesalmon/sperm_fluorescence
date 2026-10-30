"""Denoising of a measured stain frame, shared by `prepare --denoise-targets` (baked into a cache)
and `open_cache(denoise=...)` (applied on the fly when a cache is read, so an existing run can be
scored and drawn against denoised measurements without retraining).

    floor   subtract the frame's noise floor (median + 3 x MAD of the pixels at or below the
            median) and clip at 0: removes read noise, keeps resolution
    smooth  a 1-px Gaussian first, then the floor: single-pixel noise cannot survive the floor
"""
import numpy as np

NOISE_K = 3.0


def noise_floor(img, k=NOISE_K):
    v = np.asarray(img, dtype=np.float32).ravel()
    if v.size == 0: return 0.0
    med = float(np.median(v)); noise = 1.4826 * float(np.median(np.abs(v[v <= med] - med))) + 1e-6
    return med + k * noise


def denoise_target(img, how):
    if how in (None, "none"): return img
    img = np.asarray(img, dtype=np.float32)
    if how == "smooth":
        from scipy.ndimage import gaussian_filter
        img = gaussian_filter(img, 1.0)
    return np.clip(img - noise_floor(img), 0, None)
