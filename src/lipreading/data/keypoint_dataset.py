import json
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class KeypointTextDataset(Dataset):
    """
    Dataset of paired (keypoints, transcript) samples for lip reading.
    - manifest: JSONL with fields {"text": str, "kp_path": str}
      where kp_path points to a .npy file of shape [T, 478, 3].
    Tokenization/padding happens in collate.
    """

    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        text: str = rec["text"]
        kp_path: str = rec["kp_path"]

        kp_np = np.load(kp_path)  # [T, 478, 3]
        if kp_np.ndim != 3 or kp_np.shape[1] != 478 or kp_np.shape[2] != 3:
            raise ValueError(f"Bad keypoint shape at {kp_path}: {kp_np.shape}")

        keypoints = torch.from_numpy(kp_np).to(torch.float32)  # [T, 478, 3]
        return {"text": text, "face_kp": keypoints}


def load_manifest(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "text" not in obj or "kp_path" not in obj:
                continue
            records.append({"text": obj["text"], "kp_path": obj["kp_path"]})
    if not records:
        raise ValueError(f"No valid records in manifest {path}")
    return records


def split_records(
    records: List[Dict[str, Any]],
    val_ratio: float,
    seed: int = 1337,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(records))
    rng.shuffle(idx)
    cut = int(len(records) * (1.0 - val_ratio))
    train_idx = idx[:cut]
    val_idx = idx[cut:]
    train = [records[i] for i in train_idx]
    val = [records[i] for i in val_idx]
    return train, val


def collate_keypoint_batch(batch: List[Dict[str, Any]], tokenizer, seq_len: int, pad_token_id: int, eos_token_id: Optional[int]):
    # Text tokenization with padding/truncation
    texts = [b["text"] for b in batch]
    enc = tokenizer(
        texts,
        add_special_tokens=False,
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    # Append EOS where possible if not already present and room exists
    if eos_token_id is not None:
        for i in range(input_ids.size(0)):
            if attention_mask[i].sum() < seq_len:
                pos = int(attention_mask[i].sum().item())
                input_ids[i, pos] = eos_token_id
                attention_mask[i, pos] = 1

    labels = input_ids.clone()
    # Standard CausalLM ignore padding tokens
    labels[attention_mask == 0] = -100

    # Keypoints padding on T
    kp_list: List[torch.Tensor] = [b["face_kp"] for b in batch]  # [T_i, 478, 3]
    T_max = max(k.shape[0] for k in kp_list)
    B = len(kp_list)
    L = kp_list[0].shape[1]
    C = kp_list[0].shape[2]
    kp_padded = torch.zeros(B, T_max, L, C, dtype=torch.float32)
    kp_mask = torch.zeros(B, T_max, dtype=torch.bool)
    for i, k in enumerate(kp_list):
        T = k.shape[0]
        kp_padded[i, :T] = k
        kp_mask[i, :T] = True

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "face_kp": kp_padded,
        "face_kp_mask": kp_mask,
    }


