"""Device choice and memory hygiene."""


def pick_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))


def release_device_memory(dev):
    """Return the caching allocator's free blocks. With ragged batches every distinct
    (H, W) gets its own cached block, and on MPS, where device memory is the laptop's RAM,
    the cache grew to 25 GB on a 16 GB machine during the joint run and the process
    thrashed in swap instead of training."""
    import gc, torch
    gc.collect()
    if dev.type == "mps": torch.mps.empty_cache()
    elif dev.type == "cuda": torch.cuda.empty_cache()

