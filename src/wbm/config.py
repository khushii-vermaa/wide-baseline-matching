"""Config loading and per-matcher settings."""
from pathlib import Path
import yaml


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = str(path)
    return cfg


def settings_for(cfg, protocol, matcher):
    """Return the settings dict (resize, max_kp, ransac_thr) for a matcher.

    protocol 'main'      -> one common setting for all matchers (equal-protocol arm, N2)
    protocol 'published' -> per-matcher published defaults (sensitivity arm / reproduction)
    """
    if protocol == "main":
        return dict(cfg["main"])
    return dict(cfg["protocols"][protocol][matcher])


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    return Path(p)


def progress(it, **kw):
    try:
        from tqdm import tqdm
        return tqdm(it, **kw)
    except ImportError:
        return it
