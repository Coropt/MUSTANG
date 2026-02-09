import math
from typing import Iterable, List, Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader, Sampler


class NonOverlappingBatchSampler(Sampler[List[int]]):
    """
    Batch sampler that enforces non-overlapping windows within each batch.

    Assumes dataset indices map to contiguous window start positions (true for
    train/valid datasets in this project).
    """

    def __init__(
        self,
        data_source,
        batch_size: int,
        window_length: int,
        drop_last: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.data_source = data_source
        self.batch_size = int(batch_size)
        self.window_length = int(window_length)
        self.drop_last = drop_last
        self._rng = np.random.default_rng(seed)
        self._dataset_size = len(data_source)
        self._exclusion = max(0, self.window_length - 1)
        if self._dataset_size > 0 and self.window_length > 0:
            self._max_nonoverlap = max(1, math.ceil(self._dataset_size / self.window_length))
        else:
            self._max_nonoverlap = 0

    def __len__(self) -> int:
        if self._dataset_size <= 0:
            return 0
        max_full = self._max_nonoverlap // self.batch_size
        if self.drop_last:
            return max_full
        if self._max_nonoverlap <= 0:
            return 0
        return max_full + (1 if self._max_nonoverlap % self.batch_size else 0)

    def __iter__(self) -> Iterable[List[int]]:
        if self._dataset_size <= 0:
            return

        candidates = np.arange(self._dataset_size)
        self._rng.shuffle(candidates)

        remaining = candidates.tolist()
        while remaining:
            batch = self._greedy_batch(remaining)
            if len(batch) < self.batch_size:
                if batch and not self.drop_last:
                    yield batch
                break
            yield batch
            selected_set = set(batch)
            remaining = [idx for idx in remaining if idx not in selected_set]

    def _greedy_batch(self, candidates: List[int]) -> List[int]:
        selected: List[int] = []
        blocked = np.zeros(self._dataset_size, dtype=bool)

        for idx in candidates:
            if blocked[idx]:
                continue
            selected.append(int(idx))
            left = max(0, idx - self._exclusion)
            right = min(self._dataset_size - 1, idx + self._exclusion)
            blocked[left:right + 1] = True
            if len(selected) >= self.batch_size:
                break

        return selected


def build_nonoverlap_loaders(
    dataset_class,
    sequence_length: int,
    batch_size: int,
    seed: int,
    training_missing: str,
    val_len: float,
    test_len: float,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = dataset_class(
        eval_length=sequence_length,
        seed=seed,
        mode="train",
        val_len=val_len,
        test_len=test_len,
        test_missing="point",
        training_missing=training_missing,
        missing_ratio=0.1,
    )
    valid_dataset = dataset_class(
        eval_length=sequence_length,
        seed=seed,
        mode="valid",
        val_len=val_len,
        test_len=test_len,
        test_missing="point",
        training_missing=training_missing,
        missing_ratio=0.1,
    )

    train_sampler = NonOverlappingBatchSampler(
        train_dataset,
        batch_size=batch_size,
        window_length=sequence_length,
        drop_last=True,
        seed=seed,
    )
    valid_sampler = NonOverlappingBatchSampler(
        valid_dataset,
        batch_size=batch_size,
        window_length=sequence_length,
        drop_last=True,
        seed=seed + 1,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_sampler=valid_sampler,
        num_workers=num_workers,
    )

    return train_loader, valid_loader
