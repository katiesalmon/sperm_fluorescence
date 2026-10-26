"""The trunk: a 4-level U-Net, reflect padding, no global pooling, so it runs at any size."""


def build_model(width, cin=3, cout=3):
    import torch, torch.nn as nn, torch.nn.functional as F

    class Block(nn.Module):
        def __init__(s, i, o):
            super().__init__()
            s.net = nn.Sequential(nn.Conv2d(i, o, 3, padding=1, padding_mode="reflect"), nn.BatchNorm2d(o), nn.ReLU(True),
                                  nn.Conv2d(o, o, 3, padding=1, padding_mode="reflect"), nn.BatchNorm2d(o), nn.ReLU(True))
        def forward(s, x): return s.net(x)

    class UNet(nn.Module):
        def __init__(s, w, cin=3, cout=3):
            super().__init__()
            s.e1, s.e2, s.e3, s.e4 = Block(cin, w), Block(w, 2 * w), Block(2 * w, 4 * w), Block(4 * w, 8 * w)
            s.d3, s.d2, s.d1 = Block(8 * w + 4 * w, 4 * w), Block(4 * w + 2 * w, 2 * w), Block(2 * w + w, w)
            s.out = nn.Conv2d(w, cout, 1)
        def forward(s, x):
            e1 = s.e1(x); e2 = s.e2(F.max_pool2d(e1, 2)); e3 = s.e3(F.max_pool2d(e2, 2)); e4 = s.e4(F.max_pool2d(e3, 2))
            up = lambda t, ref: F.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=False)
            d3 = s.d3(torch.cat([up(e4, e3), e3], 1)); d2 = s.d2(torch.cat([up(d3, e2), e2], 1)); d1 = s.d1(torch.cat([up(d2, e1), e1], 1))
            return s.out(d1)
    return UNet(width, cin, cout)

