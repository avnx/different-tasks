"""
Utilities and toy data generation for the LAS-inspired RL task.

The dataset is intentionally tiny and synthetic. Each transcript is mapped to a
sequence of feature frames generated deterministically from character templates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

__all__ = [
    "VOCAB",
    "token_to_id",
    "id_to_token",
    "encode_text",
    "decode_tokens",
    "MiniLASDataset",
    "get_dataloaders",
    "compute_wer",
    "recompute_wer_from_pairs",
    "set_seed",
]

_BASE_VOCAB = list("abcdefghijklmnopqrstuvwxyz '")
SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>"]
VOCAB = _BASE_VOCAB + SPECIAL_TOKENS

PAD_IDX = VOCAB.index("<pad>")
SOS_IDX = VOCAB.index("<sos>")
EOS_IDX = VOCAB.index("<eos>")

FEAT_DIM = 24
FRAMES_PER_CHAR = 4

_TRAIN_TRANSCRIPTS = [
    "triple a emergency",
    "saint mary clinic",
    "cancel cancel cancel",
    "call mom please",
    "play rock music",
    "navigate home now",
    "what is weather",
    "turn lights on",
]

_VAL_TRANSCRIPTS = [
    "cancel the ride",
    "play aaa song",
    "call for help",
    "navigate saint mary",
]


def set_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)


@dataclass(frozen=True)
class MiniLASSample:
    features: torch.Tensor  # (T, FEAT_DIM)
    transcript: str


def token_to_id(token: str) -> int:
    if token not in VOCAB:
        raise KeyError(f"Unknown token: {token!r}")
    return VOCAB.index(token)


def id_to_token(idx: int) -> str:
    return VOCAB[idx]


def encode_text(text: str) -> torch.Tensor:
    indices = [token_to_id("<sos>")]
    for ch in text:
        if ch not in _BASE_VOCAB:
            raise ValueError(f"Out-of-vocabulary character {ch!r}.")
        indices.append(token_to_id(ch))
    indices.append(token_to_id("<eos>"))
    return torch.tensor(indices, dtype=torch.long)


def decode_tokens(token_ids: Sequence[int]) -> str:
    chars = []
    for idx in token_ids:
        token = id_to_token(idx)
        if token in SPECIAL_TOKENS:
            continue
        chars.append(token)
    return "".join(chars).strip()


def _char_template(ch: str) -> torch.Tensor:
    """
    Deterministically produce a (FRAMES_PER_CHAR, FEAT_DIM) template for a character.
    """
    base_idx = _BASE_VOCAB.index(ch)
    generator = torch.Generator().manual_seed(10_000 + base_idx)

    # Generate a base vector and create phase-shifted sine waves to introduce structure.
    phases = torch.linspace(0, 1, FEAT_DIM, dtype=torch.float32)
    template = []
    for step in range(FRAMES_PER_CHAR):
        shift = (step + 1) * 0.5
        vec = torch.sin(phases * (base_idx + 1) + shift)
        noise = torch.randn(FEAT_DIM, generator=generator) * 0.05
        template.append(vec + noise)
    return torch.stack(template, dim=0)


def _synthesize_features(transcript: str) -> torch.Tensor:
    frames = []
    for ch in transcript:
        if ch not in _BASE_VOCAB:
            raise ValueError(f"Out-of-vocabulary character {ch!r} in transcript {transcript!r}.")
        template = _char_template(ch)
        frames.append(template)
    stacked = torch.cat(frames, dim=0)
    time_steps = stacked.size(0)
    positional = torch.linspace(0, 1, time_steps).unsqueeze(1)
    return stacked + positional


class MiniLASDataset(Dataset[MiniLASSample]):
    def __init__(self, split: str) -> None:
        if split == "train":
            transcripts = _TRAIN_TRANSCRIPTS
        elif split == "val":
            transcripts = _VAL_TRANSCRIPTS
        else:
            raise ValueError(f"Unknown split {split!r}.")
        self._samples = [MiniLASSample(_synthesize_features(t), t) for t in transcripts]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> MiniLASSample:
        return self._samples[idx]


def _collate_fn(batch: Sequence[MiniLASSample]) -> dict[str, torch.Tensor | list[str]]:
    lengths = [sample.features.size(0) for sample in batch]
    max_len = max(lengths)

    features = torch.zeros(len(batch), max_len, FEAT_DIM, dtype=torch.float32)
    feature_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

    targets = []
    transcripts = []

    for i, sample in enumerate(batch):
        seq_len = sample.features.size(0)
        features[i, :seq_len] = sample.features
        feature_mask[i, :seq_len] = 1
        targets.append(encode_text(sample.transcript))
        transcripts.append(sample.transcript)

    target_lengths = torch.tensor([t.size(0) for t in targets], dtype=torch.long)
    max_target_len = max(t.size(0) for t in targets)
    padded_targets = torch.full((len(batch), max_target_len), PAD_IDX, dtype=torch.long)
    for i, seq in enumerate(targets):
        padded_targets[i, : seq.size(0)] = seq

    return {
        "features": features,
        "feature_mask": feature_mask,
        "targets": padded_targets,
        "target_lengths": target_lengths,
        "transcripts": transcripts,
    }


def get_dataloaders(batch_size: int = 2) -> tuple[DataLoader, DataLoader]:
    train_ds = MiniLASDataset("train")
    val_ds = MiniLASDataset("val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn)
    return train_loader, val_loader


def compute_wer(reference: str, prediction: str) -> float:
    ref_words = reference.split()
    hyp_words = prediction.split()

    if not ref_words:
        return float(len(hyp_words) > 0)

    dp = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]

    for i in range(len(ref_words) + 1):
        dp[i][0] = i
    for j in range(len(hyp_words) + 1):
        dp[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],     # deletion
                    dp[i][j - 1],     # insertion
                    dp[i - 1][j - 1], # substitution
                )

    return dp[len(ref_words)][len(hyp_words)] / len(ref_words)


def recompute_wer_from_pairs(pairs: Iterable[dict[str, str]]) -> float:
    references = []
    predictions = []
    for entry in pairs:
        if not isinstance(entry, dict):
            raise TypeError("Each prediction entry must be a dict.")
        ref = entry.get("reference")
        hyp = entry.get("prediction")
        if not isinstance(ref, str) or not isinstance(hyp, str):
            raise TypeError("Prediction entries must include 'reference' and 'prediction' strings.")
        references.append(ref.strip())
        predictions.append(hyp.strip())

    if not references:
        raise ValueError("val_predictions must not be empty.")

    wers = [compute_wer(r, h) for r, h in zip(references, predictions)]
    return float(sum(wers) / len(wers))

