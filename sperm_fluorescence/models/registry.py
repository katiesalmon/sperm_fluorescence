"""Instrument and marker definitions, read from configs/*.toml."""
import glob
import os

from .. import ROOT

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

CONFIGS = os.path.join(ROOT, "configs")


def _load(path):
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def instrument(name_or_path):
    """configs/instruments/<name>.toml, or a path to one."""
    p = name_or_path if name_or_path.endswith(".toml") else os.path.join(CONFIGS, "instruments", name_or_path + ".toml")
    cfg = _load(p)
    cfg.setdefault("roles", {}); cfg.setdefault("stains", {}); cfg.setdefault("classes", {})
    return cfg


def marker(name_or_path):
    """configs/markers/<file>.toml by file stem or by marker name (e.g. 'LDHC/AKAP4')."""
    if name_or_path.endswith(".toml"):
        return _load(name_or_path)
    p = os.path.join(CONFIGS, "markers", name_or_path + ".toml")
    if os.path.exists(p):
        return _load(p)
    for q in glob.glob(os.path.join(CONFIGS, "markers", "*.toml")):
        m = _load(q)
        if m.get("name") == name_or_path:
            return m
    raise KeyError("no marker config for %r" % name_or_path)


def markers():
    """All marker configs, keyed by marker name."""
    out = {}
    for q in sorted(glob.glob(os.path.join(CONFIGS, "markers", "*.toml"))):
        m = _load(q); out[m["name"]] = m
    return out


def colour(name):
    try:
        return tuple(marker(name)["colour"])
    except KeyError:
        return None
