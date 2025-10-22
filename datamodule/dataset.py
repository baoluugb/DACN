import io
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
import cv2


def resize_keep_ratio(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = target_h / float(h)
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)

def split_caption(line: str) -> Tuple[str, str]:
    if "\t" in line:
        k, v = line.split("\t", 1)
        return k.strip(), v.strip()
    if "	" in line:
        k, v = line.split("	", 1)
        return k.strip(), v.strip()
    parts = line.split(None, 1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]

class HMEDataset(Dataset):
    def __init__(self, source_path: Path, vocab_obj: str):
        self.source_path = Path(source_path)
        self.target_height = 128

        self.token_to_id: Dict[str, int] = vocab_obj["token_to_id"]
        self.special = vocab_obj["special"]
        self.pad_id = self.token_to_id[self.special["pad"]]
        self.bos_id = self.token_to_id[self.special["bos"]]
        self.eos_id = self.token_to_id[self.special["eos"]]
        self.unk_id = self.token_to_id[self.special["unk"]]

        self.items: List[Dict[str, Any]] = []
        self._images: Dict[str, np.ndarray] = {}

        self.load_data(self.source_path)


    def load_data(self, dir_path: Path):
        pkl_path = dir_path / "images.pkl"
        cap_path = dir_path / "caption.txt"
        if not pkl_path.exists() or not cap_path.exists():
            raise FileNotFoundError(f"Expected {pkl_path} and {cap_path}")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            self._images = data
        else:
            self._images = {k: v for k, v in data}
        for line in Path(cap_path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line: continue
            key, latex = split_caption(line)

            tokens = [t for t in latex.strip().split(" ") if t != ""] if " " in latex.strip() else list(latex)
            self.items.append({"key": key, "latex": latex, "tokens": tokens})
        self.items = [it for it in self.items if it["key"] in self._images]


    def __len__(self) -> int:
        return len(self.items)


    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.items[idx]
        key = rec["key"]
        tokens = rec["tokens"]
        img = self._images[key]
        if img.ndim == 3: img = img[..., 0]
        img = resize_keep_ratio(img, self.target_height)
        h, w = img.shape[:2]
        img = (img.astype("float32") / 255.0)[None, ...]
        ids = [self.bos_id] + [self.token_to_id.get(t, self.unk_id) for t in tokens] + [self.eos_id]
        return {"image": torch.from_numpy(img),
                "target_ids": torch.tensor(ids, dtype=torch.long),
                "length": len(ids),
                "width": w,
                "path": str(key),
                "tokens": tokens}

def collate_fn(batch: List[Dict[str, Any]], pad_id: int) -> Dict[str, Any]:
    max_w = max(x["image"].shape[-1] for x in batch)
    H = batch[0]["image"].shape[-2]
    B = len(batch)
    images = torch.full((B, 1, H, max_w), 1.0, dtype=torch.float32)
    target_lens = [x["target_ids"].shape[0] for x in batch]
    max_L = max(target_lens)
    target = torch.full((B, max_L), pad_id, dtype=torch.long)
    widths, paths, tokens_list = [], [], []
    for i, x in enumerate(batch):
        c, h, w = x["image"].shape
        images[i, :, :, :w] = x["image"]
        L = x["target_ids"].shape[0]
        target[i, :L] = x["target_ids"]
        widths.append(x["width"]); paths.append(x["path"]); tokens_list.append(x["tokens"])
    return {"images": images,
            "target_ids": target,
            "target_lens": torch.tensor(target_lens, dtype=torch.long),
            "widths": torch.tensor(widths, dtype=torch.long),
            "paths": paths,
            "tokens": tokens_list}

