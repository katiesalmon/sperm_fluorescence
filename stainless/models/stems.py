"""Modular input: one stem per label-free role, summed into the trunk.

    x (B, R, H, W)  roles in the cache's order     present (B, R) bool: which roles this object has
    feats = mean over present roles of  stem_r(x[:, r]) + bias_r      ->  U-Net trunk  ->  stain image(s)

A role the instrument lacks, or one dropped during training (`role_dropout`), contributes
nothing; the trunk never sees a zero image standing in for a missing view. This is the
convolutional form of per-channel tokens with a learned channel embedding (ChannelViT,
ChAda-ViT): order-invariant, any subset of roles at inference, one trained model for every
input configuration. Trained with role dropout it answers "light loss only" and "all three
views" from the same weights.
"""


def build_stem_model(roles, width, cout):
    import torch, torch.nn as nn
    from .unet import build_model

    class Stem(nn.Module):
        def __init__(s, w):
            super().__init__()
            s.net = nn.Sequential(nn.Conv2d(1, w, 3, padding=1, padding_mode="reflect"), nn.BatchNorm2d(w), nn.ReLU(True),
                                  nn.Conv2d(w, w, 3, padding=1, padding_mode="reflect"))
        def forward(s, x): return s.net(x)

    class RoleStemUNet(nn.Module):
        arch = "stems"

        def __init__(s, roles, w, cout):
            super().__init__()
            s.roles = list(roles)
            s.stems = nn.ModuleList([Stem(w) for _ in s.roles])
            s.role_bias = nn.Parameter(torch.zeros(len(s.roles), w))
            s.trunk = build_model(w, cin=w, cout=cout)

        def forward(s, x, present=None):
            B, R = x.shape[:2]
            if present is None:
                present = torch.ones(B, R, dtype=torch.bool, device=x.device)
            present = present.to(x.device)
            acc = None; count = present.float().sum(1).clamp(min=1.0).view(B, 1, 1, 1)
            for r, stem in enumerate(s.stems):
                on = present[:, r]
                if not bool(on.any()): continue
                f = stem(x[:, r:r + 1]) + s.role_bias[r].view(1, -1, 1, 1)
                f = f * on.float().view(B, 1, 1, 1)
                acc = f if acc is None else acc + f
            if acc is None:
                acc = torch.zeros(B, s.role_bias.shape[1], *x.shape[2:], device=x.device)
            return s.trunk(acc / count)

    return RoleStemUNet(roles, width, cout)


def build(arch, roles, width, cin, cout):
    """arch 'stack': the fixed-channel U-Net (older checkpoints). arch 'stems': role stems."""
    if arch == "stems":
        return build_stem_model(roles, width, cout)
    from .unet import build_model
    m = build_model(width, cin, cout); m.arch = "stack"
    return m


def call(model, x, present=None):
    """Forward through either architecture."""
    if getattr(model, "arch", "stack") == "stems":
        return model(x, present)
    return model(x)
