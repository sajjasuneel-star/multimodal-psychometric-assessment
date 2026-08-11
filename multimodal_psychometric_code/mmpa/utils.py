import os, json, random, hashlib
from pathlib import Path
import numpy as np
import torch


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(pref='auto'):
    if pref == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(pref)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)


def save_json(obj, path):
    ensure_dir(Path(path).parent)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)


def stable_token_id(token: str, vocab_size: int):
    h = hashlib.blake2b(token.encode('utf-8'), digest_size=8).digest()
    return 2 + (int.from_bytes(h, 'little') % max(1, vocab_size - 2))


def mean_ci95(values):
    a = np.asarray(values, dtype=float)
    if len(a) < 2:
        return float(a.mean()) if len(a) else np.nan, np.nan, np.nan
    m = float(a.mean()); sd = float(a.std(ddof=1))
    half = 1.96 * sd / np.sqrt(len(a))
    return m, m-half, m+half
