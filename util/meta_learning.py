from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn

try:
    # PyTorch 2.x
    from torch.func import functional_call
except ImportError:  # pragma: no cover - fallback for older versions
    from torch.nn.utils.stateless import functional_call


@dataclass
class MetaTask:
    """
    Represents a single meta-task for self-imputation.
    
    Self-imputation: Use a single dataset (e.g., discharge with all 31 stations)
    to impute itself. All stations are used together, split by TIME into 
    support and query sets.

    Attributes:
        support_batch: Dict containing support set data (for inner loop training).
        query_batch: Dict containing query set data (for meta-objective).
    """
    support_batch: dict
    query_batch: dict


@dataclass
class MetaTaskV2:
    """
    V2: Uses FULL batches without shape modification.
    
    Each inner loop step uses a DIFFERENT support batch, rather than
    reusing the same batch multiple times.
    
    Attributes:
        support_batches: List of dicts, one for each inner loop step.
        query_batches: List of dicts for meta-objective (multiple query batches).
    """
    support_batches: List[dict]  # List of support batches, one per inner step
    query_batches: List[dict]    # List of query batches for meta-objective


@dataclass
class MetaTaskV3:
    """
    V3: Mask-based split without changing data shape.
    
    Unlike V1 which physically splits the batch by time (changing shape from
    [B, T, D] to [B, T_support, D] and [B, T_query, D]), V3 keeps the full
    shape [B, T, D] and uses masks to indicate which time steps are 
    "visible" for support vs query.
    
    This approach:
    - Preserves the original data shape
    - Uses masks to control visibility (more aligned with conditional diffusion's design)
    - Maintains complete time embeddings
    
    Attributes:
        support_batch: Dict with full data but mask only on support time steps.
        query_batch: Dict with full data but mask only on query time steps.
    """
    support_batch: dict
    query_batch: dict


def split_support_query_by_mask(
    observed_data: torch.Tensor,
    observed_mask: torch.Tensor,
    cond_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    support_frac: float = 0.5,
) -> Tuple[dict, dict]:
    """
    V3: Splits data into support and query sets using MASKS (not shape change).
    
    Unlike split_support_query_by_time which physically truncates the batch,
    this function keeps the full shape and uses masks to indicate visibility.
    
    Args:
        observed_data: [B, T, D] tensor of observed data
        observed_mask: [B, T, D] observation mask
        cond_mask: [B, T, D] conditioning mask
        gt_mask: [B, T, D] ground truth mask
        support_frac: Fraction of time points for support set

    Returns:
        support_batch, query_batch: Dicts with full shape but different masks
    """
    B, T, D = observed_data.shape
    device = observed_data.device
    
    # Determine number of support and query time steps
    num_support = max(1, int(T * support_frac))
    num_query = T - num_support
    
    if num_query < 1:
        num_support = T - 1
        num_query = 1
    
    # Random permutation of time indices
    perm = torch.randperm(T, device=device)
    support_indices = perm[:num_support]
    query_indices = perm[num_support:]
    
    # Create time-based masks [B, T, D]
    # Support mask: 1 at support time steps, 0 elsewhere
    support_time_mask = torch.zeros(B, T, D, device=device)
    support_time_mask[:, support_indices, :] = 1.0
    
    # Query mask: 1 at query time steps, 0 elsewhere
    query_time_mask = torch.zeros(B, T, D, device=device)
    query_time_mask[:, query_indices, :] = 1.0
    
    # For support batch:
    # - observed_mask: original mask AND support time mask (only see support times)
    # - cond_mask: original cond_mask AND support time mask (preserves random masking for training)
    # - gt_mask: same as observed_mask for support (used for loss calculation)
    # 
    # IMPORTANT: cond_mask must be different from observed_mask to create target_mask!
    # target_mask = observed_mask - cond_mask, if they are equal, target_mask = 0, no gradient!
    support_visible_mask = observed_mask * support_time_mask
    support_cond_mask = cond_mask * support_time_mask  # Use original cond_mask, not support_visible_mask!
    support_target_mask = support_visible_mask - support_cond_mask  # Points to predict in support set
    
    support_batch = {
        "observed_data": observed_data.clone(),  # Full data
        "observed_mask": support_visible_mask,    # Only support times visible
        "gt_mask": support_visible_mask,          # Same as observed_mask
        "cond_mask": support_cond_mask,           # Random mask within support times (allows gradient flow!)
        "timepoints": torch.arange(T, device=device).unsqueeze(0).expand(B, -1).float(),
        "cut_length": torch.zeros(B, dtype=torch.long, device=device),
    }
    
    # For query batch:
    # - The model should predict query time steps
    # - observed_mask: original mask (what the model thinks is observed)
    # - cond_mask: support cond_mask (condition on support, predict query)
    # - gt_mask: query times (evaluate on query times)
    query_visible_mask = observed_mask * query_time_mask
    query_cond_mask = cond_mask * query_time_mask
    
    # Construct observed_mask_for_query: exclude support_target_mask from observed_mask
    # observed_mask_for_query = observed_mask - support_target_mask
    # = support_cond_mask + query_target_mask + query_cond_mask
    # This ensures target_mask = observed_mask_for_query - cond_mask = query_target_mask + query_cond_mask
    # Loss will be computed on both query_target_mask and query_cond_mask
    observed_mask_for_query = observed_mask - support_target_mask
    
    query_batch = {
        "observed_data": observed_data.clone(),   # Full data
        "observed_mask": observed_mask_for_query,  # support_cond_mask + query_target_mask + query_cond_mask
        "gt_mask": query_visible_mask,            # Evaluate only on query times
        "cond_mask": support_cond_mask,           # Use support_cond_mask as known conditions
        "timepoints": torch.arange(T, device=device).unsqueeze(0).expand(B, -1).float(),
        "cut_length": torch.zeros(B, dtype=torch.long, device=device),
    }

    # query_batch = { 
    #     "observed_data": observed_data.clone(), # Full data 
    #     "observed_mask": observed_mask.clone(), # Original observation mask 
    #     "gt_mask": query_visible_mask, # Evaluate only on query times 
    #     "cond_mask": support_cond_mask.clone(), # Use support_cond_mask (consistent with support_batch) 
    #     "timepoints": torch.arange(T, device=device).unsqueeze(0).expand(B, -1).float(), 
    #     "cut_length": torch.zeros(B, dtype=torch.long, device=device), 
    # }
    
    return support_batch, query_batch


def split_support_query_by_time(
    observed_data: torch.Tensor,
    observed_mask: torch.Tensor,
    cond_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    support_frac: float = 0.5,
    sequence_length: int = 16,
) -> Tuple[dict, dict]:
    """
    Splits data into support and query sets BY TIME (not by station).
    
    This preserves conditional diffusion's multi-station joint modeling - all stations are 
    used together, just on different time segments.

    Args:
        observed_data: [B, T, D] tensor of observed data
        observed_mask: [B, T, D] observation mask
        cond_mask: [B, T, D] conditioning mask
        gt_mask: [B, T, D] ground truth mask
        support_frac: Fraction of time points for support set
        sequence_length: Length of each sequence

    Returns:
        support_batch, query_batch: Dicts matching conditional diffusion's expected input format
    """
    B, T, D = observed_data.shape
    device = observed_data.device
    
    # Split time points into support and query
    num_support = max(1, int(T * support_frac))
    num_query = T - num_support
    
    if num_query < 1:
        num_support = T - 1
        num_query = 1
    
    # Random permutation of time indices
    perm = torch.randperm(T, device=device)
    support_indices = perm[:num_support].sort().values
    query_indices = perm[num_support:].sort().values
    
    # Create support batch
    support_batch = {
        "observed_data": observed_data[:, support_indices, :],
        "observed_mask": observed_mask[:, support_indices, :],
        "gt_mask": gt_mask[:, support_indices, :],
        "cond_mask": cond_mask[:, support_indices, :],
        "timepoints": support_indices.unsqueeze(0).expand(B, -1).float(),
        "cut_length": torch.zeros(B, dtype=torch.long, device=device),
    }
    
    # Create query batch
    query_batch = {
        "observed_data": observed_data[:, query_indices, :],
        "observed_mask": observed_mask[:, query_indices, :],
        "gt_mask": gt_mask[:, query_indices, :],
        "cond_mask": cond_mask[:, query_indices, :],
        "timepoints": query_indices.unsqueeze(0).expand(B, -1).float(),
        "cut_length": torch.zeros(B, dtype=torch.long, device=device),
    }
    
    return support_batch, query_batch


class MetaTaskLoader:
    """
    Simple wrapper for single-dataset MUSTANG.
    Internally uses MultiDatasetMetaTaskLoader with one dataset.
    
    For proper cross-dataset MUSTANG, use MultiDatasetMetaTaskLoader directly.
    """

    def __init__(
        self,
        data_loader,
        support_frac: float = 0.5,
        tasks_per_batch: int = 4,
    ):
        # Wrap single loader in a dict and delegate to MultiDatasetMetaTaskLoader
        self._multi_loader = MultiDatasetMetaTaskLoader(
            data_loaders={"default": data_loader},
            support_frac=support_frac,
            tasks_per_batch=tasks_per_batch,
        )

    def __iter__(self) -> Iterable[List[MetaTask]]:
        return iter(self._multi_loader)


class MultiDatasetMetaTaskLoader:
    """
    Meta-task loader that samples from MULTIPLE datasets (RANDOM sampling).
    
    This enables true cross-dataset MUSTANG:
    - Tasks are RANDOMLY sampled from different datasets (TP, NO3, NH4, SRP, etc.)
    - The model learns a general imputation strategy across all datasets
    - The meta-learned initialization can then be fine-tuned on a new dataset
    
    Note: This uses random sampling, so tasks_per_batch tasks may come from
    the same dataset. For round-robin sampling, use RoundRobinMetaTaskLoader.
    
    Example:
        loaders = {
            'tp': tp_train_loader,
            'no3': no3_train_loader,
            'nh4': nh4_train_loader,
            'srp': srp_train_loader,
        }
        meta_loader = MultiDatasetMetaTaskLoader(loaders, support_frac=0.5)
    """

    def __init__(
        self,
        data_loaders: dict,
        support_frac: float = 0.5,
        tasks_per_batch: int = 4,
    ):
        """
        Args:
            data_loaders: Dict mapping dataset names to DataLoaders
            support_frac: Fraction of time points for support set
            tasks_per_batch: Number of tasks per meta-batch
        """
        self.data_loaders = data_loaders
        self.dataset_names = list(data_loaders.keys())
        self.support_frac = support_frac
        self.tasks_per_batch = tasks_per_batch
        
        # Create iterators for each dataset
        self.data_iters = {
            name: iter(loader) for name, loader in data_loaders.items()
        }
        
        # Track dataset sizes for logging
        self.dataset_sizes = {
            name: len(loader) for name, loader in data_loaders.items()
        }
        
        print(f"MultiDatasetMetaTaskLoader initialized with {len(self.dataset_names)} datasets:")
        for name in self.dataset_names:
            print(f"  - {name}: {self.dataset_sizes[name]} batches")

    def _get_batch(self, dataset_name: str):
        """Get a batch from the specified dataset, resetting if exhausted."""
        try:
            return next(self.data_iters[dataset_name])
        except StopIteration:
            self.data_iters[dataset_name] = iter(self.data_loaders[dataset_name])
            return next(self.data_iters[dataset_name])

    def _sample_dataset(self) -> str:
        """Uniformly sample which dataset to use for the next task."""
        import random
        return random.choice(self.dataset_names)

    def __iter__(self) -> Iterable[List[MetaTask]]:
        while True:
            yield [self._sample_task() for _ in range(self.tasks_per_batch)]

    def _sample_task(self) -> MetaTask:
        """Sample a task from a randomly selected dataset."""
        # Choose which dataset to sample from
        dataset_name = self._sample_dataset()
        batch = self._get_batch(dataset_name)
        
        # Extract data from batch
        observed_data = batch["observed_data"]
        observed_mask = batch["observed_mask"]
        cond_mask = batch["cond_mask"]
        gt_mask = batch["gt_mask"]
        
        # Convert to tensors and force float dtype (like conditional diffusion.process_data)
        # This ensures masks are always float32, allowing subtraction operations
        if not isinstance(observed_data, torch.Tensor):
            observed_data = torch.tensor(observed_data, dtype=torch.float32)
        else:
            observed_data = observed_data.float()
        if not isinstance(observed_mask, torch.Tensor):
            observed_mask = torch.tensor(observed_mask, dtype=torch.float32)
        else:
            observed_mask = observed_mask.float()
        if not isinstance(cond_mask, torch.Tensor):
            cond_mask = torch.tensor(cond_mask, dtype=torch.float32)
        else:
            cond_mask = cond_mask.float()
        if not isinstance(gt_mask, torch.Tensor):
            gt_mask = torch.tensor(gt_mask, dtype=torch.float32)
        else:
            gt_mask = gt_mask.float()
        
        # Split by time into support and query
        support_batch, query_batch = split_support_query_by_time(
            observed_data=observed_data,
            observed_mask=observed_mask,
            cond_mask=cond_mask,
            gt_mask=gt_mask,
            support_frac=self.support_frac,
        )
        
        return MetaTask(support_batch=support_batch, query_batch=query_batch)


class RoundRobinMetaTaskLoader:
    """
    Meta-task loader that samples from MULTIPLE datasets using ROUND-ROBIN.
    
    Key differences from MultiDatasetMetaTaskLoader:
    - Each epoch yields exactly one task from EACH dataset (round-robin, not random)
    - tasks_per_batch = number of datasets (one from each)
    - Expects DataLoaders with shuffle=False for chronological order
    
    Example:
        loaders = {
            'tp': tp_train_loader,      # shuffle=False
            'no3': no3_train_loader,    # shuffle=False
            'nh4': nh4_train_loader,    # shuffle=False
            'srp': srp_train_loader,    # shuffle=False
        }
        meta_loader = RoundRobinMetaTaskLoader(loaders, support_frac=0.5)
        # Each epoch yields 4 tasks: [tp_task, no3_task, nh4_task, srp_task]
    """

    def __init__(
        self,
        data_loaders: dict,
        support_frac: float = 0.5,
    ):
        """
        Args:
            data_loaders: Dict mapping dataset names to DataLoaders (should use shuffle=False)
            support_frac: Fraction of time points for support set
        """
        self.data_loaders = data_loaders
        self.dataset_names = list(data_loaders.keys())
        self.support_frac = support_frac
        
        # Create iterators for each dataset
        self.data_iters = {
            name: iter(loader) for name, loader in data_loaders.items()
        }
        
        # Track dataset sizes for logging
        self.dataset_sizes = {
            name: len(loader) for name, loader in data_loaders.items()
        }
        
        print(f"RoundRobinMetaTaskLoader initialized with {len(self.dataset_names)} datasets:")
        print(f"  Each epoch = {len(self.dataset_names)} tasks (one from each dataset)")
        for name in self.dataset_names:
            print(f"  - {name}: {self.dataset_sizes[name]} batches")

    def _get_batch(self, dataset_name: str):
        """Get a batch from the specified dataset, resetting if exhausted."""
        try:
            return next(self.data_iters[dataset_name])
        except StopIteration:
            self.data_iters[dataset_name] = iter(self.data_loaders[dataset_name])
            return next(self.data_iters[dataset_name])

    def _create_task_from_batch(self, batch: dict) -> MetaTask:
        """Create a MetaTask from a batch."""
        # Extract data from batch
        observed_data = batch["observed_data"]
        observed_mask = batch["observed_mask"]
        cond_mask = batch["cond_mask"]
        gt_mask = batch["gt_mask"]
        
        # Convert to tensors and force float dtype (like conditional diffusion.process_data)
        # This ensures masks are always float32, allowing subtraction operations
        if not isinstance(observed_data, torch.Tensor):
            observed_data = torch.tensor(observed_data, dtype=torch.float32)
        else:
            observed_data = observed_data.float()
        if not isinstance(observed_mask, torch.Tensor):
            observed_mask = torch.tensor(observed_mask, dtype=torch.float32)
        else:
            observed_mask = observed_mask.float()
        if not isinstance(cond_mask, torch.Tensor):
            cond_mask = torch.tensor(cond_mask, dtype=torch.float32)
        else:
            cond_mask = cond_mask.float()
        if not isinstance(gt_mask, torch.Tensor):
            gt_mask = torch.tensor(gt_mask, dtype=torch.float32)
        else:
            gt_mask = gt_mask.float()
        
        # Split by time into support and query
        support_batch, query_batch = split_support_query_by_time(
            observed_data=observed_data,
            observed_mask=observed_mask,
            cond_mask=cond_mask,
            gt_mask=gt_mask,
            support_frac=self.support_frac,
        )
        
        return MetaTask(support_batch=support_batch, query_batch=query_batch)

    def __iter__(self) -> Iterable[List[MetaTask]]:
        while True:
            # Get one batch from EACH dataset (round-robin)
            tasks = []
            for dataset_name in self.dataset_names:
                batch = self._get_batch(dataset_name)
                task = self._create_task_from_batch(batch)
                tasks.append(task)
            yield tasks


# =============================================================================
# V3: Mask-Based Meta-Learning (same shape, different masks)
# =============================================================================

class MultiDatasetMetaTaskLoaderV3:
    """
    V3: Mask-based split without changing data shape.
    
    Unlike V1 which physically splits batches by time (changing shape),
    V3 keeps the full shape [B, T, D] and uses masks to indicate which
    time steps are visible for support vs query.
    
    Key features:
    - Data shape preserved: [B, T, D] for both support and query
    - Uses masks to control visibility
    - Support mask: only support time steps visible
    - Query mask: evaluates only on query time steps
    - Same inner loop as V1 (single support batch, multiple steps)
    
    Example:
        loaders = {'tp': tp_loader, 'no3': no3_loader, ...}
        meta_loader = MultiDatasetMetaTaskLoaderV3(loaders, support_frac=0.5)
        # Each task uses one batch with mask-based support/query split
    """

    def __init__(
        self,
        data_loaders: dict,
        support_frac: float = 0.5,
        tasks_per_batch: int = 4,
    ):
        """
        Args:
            data_loaders: Dict mapping dataset names to DataLoaders
            support_frac: Fraction of time points for support set
            tasks_per_batch: Number of tasks per meta-batch
        """
        self.data_loaders = data_loaders
        self.dataset_names = list(data_loaders.keys())
        self.support_frac = support_frac
        self.tasks_per_batch = tasks_per_batch
        
        # Create iterators for each dataset
        self.data_iters = {
            name: iter(loader) for name, loader in data_loaders.items()
        }
        
        # Track dataset sizes for logging
        self.dataset_sizes = {
            name: len(loader) for name, loader in data_loaders.items()
        }
        
        print(f"MultiDatasetMetaTaskLoaderV3 initialized with {len(self.dataset_names)} datasets:")
        print(f"  V3: Mask-based split (shape preserved)")
        print(f"  Support fraction: {support_frac}")
        for name in self.dataset_names:
            print(f"  - {name}: {self.dataset_sizes[name]} batches")

    def _get_batch(self, dataset_name: str):
        """Get a batch from the specified dataset, resetting if exhausted."""
        try:
            return next(self.data_iters[dataset_name])
        except StopIteration:
            self.data_iters[dataset_name] = iter(self.data_loaders[dataset_name])
            return next(self.data_iters[dataset_name])

    def _sample_dataset(self) -> str:
        """Uniformly sample which dataset to use for the next task."""
        import random
        return random.choice(self.dataset_names)

    def __iter__(self) -> Iterable[List[MetaTaskV3]]:
        while True:
            yield [self._sample_task() for _ in range(self.tasks_per_batch)]

    def _sample_task(self) -> MetaTaskV3:
        """Sample a task from a randomly selected dataset."""
        # Choose which dataset to sample from
        dataset_name = self._sample_dataset()
        batch = self._get_batch(dataset_name)
        
        # Extract data from batch
        observed_data = batch["observed_data"]
        observed_mask = batch["observed_mask"]
        cond_mask = batch["cond_mask"]
        gt_mask = batch["gt_mask"]
        
        # Convert to tensors and force float dtype (like conditional diffusion.process_data)
        # This ensures masks are always float32, allowing subtraction operations
        if not isinstance(observed_data, torch.Tensor):
            observed_data = torch.tensor(observed_data, dtype=torch.float32)
        else:
            observed_data = observed_data.float()
        if not isinstance(observed_mask, torch.Tensor):
            observed_mask = torch.tensor(observed_mask, dtype=torch.float32)
        else:
            observed_mask = observed_mask.float()
        if not isinstance(cond_mask, torch.Tensor):
            cond_mask = torch.tensor(cond_mask, dtype=torch.float32)
        else:
            cond_mask = cond_mask.float()
        if not isinstance(gt_mask, torch.Tensor):
            gt_mask = torch.tensor(gt_mask, dtype=torch.float32)
        else:
            gt_mask = gt_mask.float()
        
        # Split by mask (V3 style - preserves shape)
        support_batch, query_batch = split_support_query_by_mask(
            observed_data=observed_data,
            observed_mask=observed_mask,
            cond_mask=cond_mask,
            gt_mask=gt_mask,
            support_frac=self.support_frac,
        )
        
        return MetaTaskV3(support_batch=support_batch, query_batch=query_batch)


# =============================================================================
# V2: Full-Batch Meta-Learning (no time-based splitting)
# =============================================================================

def _convert_batch_to_tensors(batch: dict) -> dict:
    """Convert batch data to tensors if needed."""
    result = {}
    for key, value in batch.items():
        if key in ["observed_data", "observed_mask", "cond_mask", "gt_mask"]:
            if not isinstance(value, torch.Tensor):
                result[key] = torch.tensor(value, dtype=torch.float32)
            else:
                result[key] = value
        else:
            result[key] = value
    return result


class MultiDatasetMetaTaskLoaderV2:
    """
    V2: Uses FULL batches without time-based splitting.
    
    Each task consists of:
    - `inner_steps` support batches (number of different batches to use)
    - `query_batches` query batches (for meta-objective)
    
    Total batches per task = inner_steps + query_batches (e.g., 8 + 16 = 24 batches)
    
    Training order: With inner_epochs=2 and batches [A, B, C], 
    the training sequence is: A, B, C, A, B, C
    
    Key differences from V1:
    - No time-based splitting of batches
    - inner_steps: number of different batches to use (not training steps)
    - inner_epochs: number of times to iterate through all batches
    - Multiple query batches for more stable meta-gradient
    - Batches retain their original shape [B, T, D]
    
    Example:
        loaders = {'tp': tp_loader, 'no3': no3_loader, ...}
        meta_loader = MultiDatasetMetaTaskLoaderV2(loaders, inner_steps=8, query_batches=16)
        # Each task uses 24 different batches from the same dataset
        # With inner_epochs=2, inner loop trains: A,B,C,D,E,F,G,H, A,B,C,D,E,F,G,H
    """

    def __init__(
        self,
        data_loaders: dict,
        inner_steps: int = 8,
        query_batches: int = 16,
        tasks_per_batch: int = 4,
    ):
        """
        Args:
            data_loaders: Dict mapping dataset names to DataLoaders
            inner_steps: Number of different support batches to use (not training steps)
            query_batches: Number of query batches for meta-objective
            tasks_per_batch: Number of tasks per meta-batch
        """
        self.data_loaders = data_loaders
        self.dataset_names = list(data_loaders.keys())
        self.inner_steps = inner_steps
        self.query_batches = query_batches
        self.tasks_per_batch = tasks_per_batch
        
        # Create iterators for each dataset
        self.data_iters = {
            name: iter(loader) for name, loader in data_loaders.items()
        }
        
        # Track dataset sizes for logging
        self.dataset_sizes = {
            name: len(loader) for name, loader in data_loaders.items()
        }
        
        print(f"MultiDatasetMetaTaskLoaderV2 initialized with {len(self.dataset_names)} datasets:")
        print(f"  inner_steps={inner_steps}, query_batches={query_batches}, tasks_per_batch={tasks_per_batch}")
        print(f"  Each task uses {inner_steps + query_batches} batches ({inner_steps} support + {query_batches} query)")
        for name in self.dataset_names:
            print(f"  - {name}: {self.dataset_sizes[name]} batches")

    def _get_batch(self, dataset_name: str) -> dict:
        """Get a batch from the specified dataset, resetting if exhausted."""
        try:
            batch = next(self.data_iters[dataset_name])
        except StopIteration:
            self.data_iters[dataset_name] = iter(self.data_loaders[dataset_name])
            batch = next(self.data_iters[dataset_name])
        return _convert_batch_to_tensors(batch)

    def _sample_dataset(self) -> str:
        """Uniformly sample which dataset to use for the next task."""
        import random
        return random.choice(self.dataset_names)

    def _sample_task(self) -> MetaTaskV2:
        """Sample a task: inner_steps support batches + query_batches query batches."""
        dataset_name = self._sample_dataset()
        
        # Get inner_steps support batches + query_batches query batches
        support_batches = [self._get_batch(dataset_name) for _ in range(self.inner_steps)]
        query_batches = [self._get_batch(dataset_name) for _ in range(self.query_batches)]
        
        return MetaTaskV2(support_batches=support_batches, query_batches=query_batches)

    def __iter__(self) -> Iterable[List[MetaTaskV2]]:
        while True:
            yield [self._sample_task() for _ in range(self.tasks_per_batch)]


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move all tensors in a batch to the specified device."""
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def _task_to_device(task: MetaTask, device: torch.device) -> MetaTask:
    """Move a MetaTask to the specified device."""
    return MetaTask(
        support_batch=_batch_to_device(task.support_batch, device),
        query_batch=_batch_to_device(task.query_batch, device),
    )


def _task_to_device_v2(task: MetaTaskV2, device: torch.device) -> MetaTaskV2:
    """Move a MetaTaskV2 to the specified device."""
    return MetaTaskV2(
        support_batches=[_batch_to_device(b, device) for b in task.support_batches],
        query_batches=[_batch_to_device(b, device) for b in task.query_batches],
    )


def _task_to_device_v3(task: MetaTaskV3, device: torch.device) -> MetaTaskV3:
    """Move a MetaTaskV3 to the specified device."""
    return MetaTaskV3(
        support_batch=_batch_to_device(task.support_batch, device),
        query_batch=_batch_to_device(task.query_batch, device),
    )


def _forward_with_params(
    model: nn.Module,
    batch: dict,
    params: Optional[OrderedDict],
    buffers: Optional[OrderedDict],
    is_train: int = 1,
) -> torch.Tensor:
    """
    Forward pass using either model's own params or provided params.
    This calls conditional diffusion's forward directly - the original conditional diffusion training
    """
    if params is None:
        return model(batch, is_train=is_train)
    buffer_dict = buffers if buffers is not None else OrderedDict(model.named_buffers())
    return functional_call(model, (params, buffer_dict), (batch, is_train))


def inner_loop_adapt(
    model: nn.Module,
    task: MetaTask,
    inner_steps: int,
    inner_lr: float,
    buffers: Optional[OrderedDict] = None,
) -> OrderedDict:
    """
    First-Order MAML inner loop adaptation.
    
    Performs K steps of gradient descent on the SUPPORT set using conditional diffusion directly.
    This is exactly like normal conditional diffusion training, but on the support time points only.
    
    Args:
        model: The conditional diffusion model.
        task: MetaTask with support_batch and query_batch.
        inner_steps: Number of inner-loop gradient steps.
        inner_lr: Inner-loop learning rate.
        buffers: Optional buffer dict for functional_call.
    
    Returns:
        Adapted parameters as OrderedDict.
    """
    device = next(model.parameters()).device
    task = _task_to_device(task, device)
    buffer_dict = buffers if buffers is not None else OrderedDict(model.named_buffers())
    
    # Start with cloned parameters
    fast_params = OrderedDict(
        (name, param.clone()) for name, param in model.named_parameters()
    )

    for step in range(inner_steps):
        # Use conditional diffusion directly on support batch - this is the original conditional diffusion forward
        support_loss = _forward_with_params(
            model, task.support_batch, fast_params, buffer_dict, is_train=1
        )
        
        grads = torch.autograd.grad(
            support_loss, 
            fast_params.values(), 
            create_graph=False,  # First-order: no graph through inner loop
            allow_unused=True,
        )
        
        # Detach and update - for first-order MAML
        fast_params = OrderedDict(
            (name, (param - inner_lr * grad).detach().requires_grad_(True) 
             if grad is not None else param.detach().requires_grad_(True))
            for (name, param), grad in zip(fast_params.items(), grads)
        )

    return fast_params


def inner_loop_adapt_v2(
    model: nn.Module,
    task: MetaTaskV2,
    inner_lr: float,
    inner_epochs: int = 1,
    buffers: Optional[OrderedDict] = None,
) -> OrderedDict:
    """
    V2: First-Order MAML inner loop with multiple epochs over batches.
    
    Training order: if batches are [A, B, C] and inner_epochs=2,
    the training sequence is: A, B, C, A, B, C
    
    Args:
        model: The conditional diffusion model.
        task: MetaTaskV2 with support_batches (list) and query_batches.
        inner_lr: Inner-loop learning rate.
        inner_epochs: Number of times to iterate through all support batches.
        buffers: Optional buffer dict for functional_call.
    
    Returns:
        Adapted parameters as OrderedDict.
    """
    device = next(model.parameters()).device
    task = _task_to_device_v2(task, device)
    buffer_dict = buffers if buffers is not None else OrderedDict(model.named_buffers())
    
    # Start with cloned parameters
    fast_params = OrderedDict(
        (name, param.clone()) for name, param in model.named_parameters()
    )

    # Iterate through all batches multiple times (inner_epochs)
    # Example: inner_epochs=2, batches=[A,B,C] -> A, B, C, A, B, C
    total_steps = len(task.support_batches) * inner_epochs
    for epoch in range(inner_epochs):
        for batch_idx, support_batch in enumerate(task.support_batches):
            # Use conditional diffusion directly on this batch
            support_loss = _forward_with_params(
                model, support_batch, fast_params, buffer_dict, is_train=1
            )
            
            grads = torch.autograd.grad(
                support_loss, 
                fast_params.values(), 
                create_graph=False,  # First-order: no graph through inner loop
                allow_unused=True,
            )
            
            # Detach and update - for first-order MAML
            fast_params = OrderedDict(
                (name, (param - inner_lr * grad).detach().requires_grad_(True) 
                 if grad is not None else param.detach().requires_grad_(True))
                for (name, param), grad in zip(fast_params.items(), grads)
            )

    return fast_params


def meta_train(
    model: nn.Module,
    meta_loader: Iterable[List[MetaTask]],
    optimizer: torch.optim.Optimizer,
    inner_steps: int,
    inner_lr: float,
    num_outer_steps: int,
    grad_clip: float = 1.0,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    log_interval: int = 10,
    query_batches: int = 1,
    wandb_run=None,
    val_loader: Optional[Iterable[List[MetaTask]]] = None,
    val_interval: int = 20,
    num_val_tasks: int = 10,
) -> List[float]:
    """
    Runs First-Order MAML (FOMAML) for self-imputation.
    
    Args:
        model: The conditional diffusion model to meta-train.
        meta_loader: MetaTaskLoader yielding batches of MetaTasks.
        optimizer: Optimizer for outer-loop updates.
        inner_steps: Number of inner loop steps.
        inner_lr: Learning rate for inner loop.
        num_outer_steps: Number of outer-loop iterations.
        grad_clip: Max gradient norm for clipping. Set to 0 to disable.
        scheduler: Optional learning rate scheduler.
        log_interval: How often to print progress.
        query_batches: (unused, kept for compatibility)
    
    Returns:
        List of meta-losses for each outer step.
    """
    device = next(model.parameters()).device
    buffer_dict = OrderedDict(model.named_buffers())
    task_iter = iter(meta_loader)
    meta_losses_history = []

    model.train()
    for outer_step in range(num_outer_steps):
        tasks = next(task_iter)
        optimizer.zero_grad()
        
        # First-Order MAML: Accumulate gradients from query losses
        total_query_loss = 0.0
        
        for task in tasks:
            task_on_device = _task_to_device(task, device)
            
            fast_params = inner_loop_adapt(
                model=model,
                task=task_on_device,
                inner_steps=inner_steps,
                inner_lr=inner_lr,
                buffers=buffer_dict,
            )
            
            # Compute query loss with adapted params
            query_loss = _forward_with_params(
                model, task_on_device.query_batch, fast_params, buffer_dict, is_train=1
            )
            total_query_loss += query_loss.item()
            
            # Compute gradient of query loss w.r.t. fast_params
            query_grads = torch.autograd.grad(
                query_loss, 
                fast_params.values(),
                allow_unused=True,
            )
            
            # Apply these gradients to the original model parameters
            # This is the first-order approximation
            for (name, param), grad in zip(model.named_parameters(), query_grads):
                if grad is not None and param.grad is not None:
                    param.grad.add_(grad / len(tasks))
                elif grad is not None:
                    param.grad = grad.clone() / len(tasks)
        
        meta_loss_value = total_query_loss / len(tasks)
        
        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        meta_losses_history.append(meta_loss_value)
        
        # Print every outer step to monitor convergence
        lr = optimizer.param_groups[0]['lr']
        print(f"[meta step {outer_step + 1}/{num_outer_steps}] "
              f"meta-loss={meta_loss_value:.4f} lr={lr:.2e}")
        if wandb_run is not None:
            wandb_run.log(
                {"train/meta_loss": meta_loss_value, "train/lr": lr},
                step=outer_step,
            )
        if val_loader is not None and (outer_step + 1) % val_interval == 0:
            val_loss = meta_validate(
                model=model,
                meta_loader=val_loader,
                inner_steps=inner_steps,
                inner_lr=inner_lr,
                num_val_tasks=num_val_tasks,
                query_batches=query_batches,
            )
            print(f"  [Validation] val_loss={val_loss:.4f}")
            if wandb_run is not None:
                wandb_run.log(
                    {"meta/val_loss": val_loss},
                    step=outer_step,
                )
    
    return meta_losses_history


def meta_validate(
    model: nn.Module,
    meta_loader: Iterable[List[MetaTask]],
    inner_steps: int,
    inner_lr: float,
    num_val_tasks: int = 50,
    query_batches: int = 1,
) -> float:
    """
    Evaluate MUSTANG performance on validation tasks.
    
    Args:
        model: The MUSTANG-trained conditional diffusion model.
        meta_loader: MetaTaskLoader yielding batches of MetaTasks.
        inner_steps: Number of inner loop steps.
        inner_lr: Inner-loop learning rate.
        num_val_tasks: Number of task batches to evaluate.
        query_batches: (unused, kept for compatibility)
    
    Returns:
        Average query loss across all validation tasks.
    """
    device = next(model.parameters()).device
    buffer_dict = OrderedDict(model.named_buffers())
    task_iter = iter(meta_loader)
    
    model.eval()
    total_loss = 0.0
    total_tasks = 0
    
    for batch_idx in range(num_val_tasks):
        try:
            tasks = next(task_iter)
        except StopIteration:
            break
        
        for task in tasks:
            task_on_device = _task_to_device(task, device)
            
            # Need gradients for inner loop even in eval mode
            with torch.enable_grad():
                fast_params = inner_loop_adapt(
                    model=model,
                    task=task_on_device,
                    inner_steps=inner_steps,
                    inner_lr=inner_lr,
                    buffers=buffer_dict,
                )
            
            # Evaluate on query set without gradients
            with torch.no_grad():
                query_loss = _forward_with_params(
                    model, task_on_device.query_batch, fast_params, buffer_dict, is_train=0
                )
                total_loss += query_loss.item()
                total_tasks += 1
    
    model.train()
    return total_loss / max(total_tasks, 1)


# =============================================================================
# V2: Meta-Training and Validation with Full Batches
# =============================================================================

def meta_train_v2(
    model: nn.Module,
    meta_loader: Iterable[List[MetaTaskV2]],
    optimizer: torch.optim.Optimizer,
    inner_lr: float,
    inner_epochs: int = 1,
    num_outer_steps: int = 200,
    grad_clip: float = 1.0,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    log_interval: int = 10,
    wandb_run=None,
    val_loader: Optional[Iterable[List[MetaTaskV2]]] = None,
    val_interval: int = 20,
    num_val_tasks: int = 10,
) -> List[float]:
    """
    V2: First-Order MAML with multiple epochs over support batches.
    
    Each task uses:
    - len(task.support_batches) different batches for inner loop
    - inner_epochs: number of times to iterate through all batches (e.g., 2 means A,B,C,A,B,C)
    - len(task.query_batches) batches for query (meta-objective)
    
    Args:
        model: The conditional diffusion model to meta-train.
        meta_loader: MetaTaskLoaderV2 yielding batches of MetaTaskV2.
        optimizer: Optimizer for outer-loop updates.
        inner_lr: Learning rate for inner loop.
        inner_epochs: Number of times to iterate through all support batches.
        num_outer_steps: Number of outer-loop iterations.
        grad_clip: Max gradient norm for clipping. Set to 0 to disable.
        scheduler: Optional learning rate scheduler.
        log_interval: How often to print progress.
    
    Returns:
        List of meta-losses for each outer step.
    """
    device = next(model.parameters()).device
    buffer_dict = OrderedDict(model.named_buffers())
    task_iter = iter(meta_loader)
    meta_losses_history = []

    model.train()
    for outer_step in range(num_outer_steps):
        tasks = next(task_iter)
        optimizer.zero_grad()
        
        # First-Order MAML: Accumulate gradients from query losses
        total_query_loss = 0.0
        total_query_batches = 0
        
        for task in tasks:
            task_on_device = _task_to_device_v2(task, device)
            
            # V2: inner loop iterates through batches multiple times
            fast_params = inner_loop_adapt_v2(
                model=model,
                task=task_on_device,
                inner_lr=inner_lr,
                inner_epochs=inner_epochs,
                buffers=buffer_dict,
            )
            
            # Compute query loss with adapted params over ALL query batches
            task_query_loss = 0.0
            for query_batch in task_on_device.query_batches:
                query_loss = _forward_with_params(
                    model, query_batch, fast_params, buffer_dict, is_train=1
                )
                task_query_loss += query_loss
                total_query_loss += query_loss.item()
                total_query_batches += 1
            
            # Average query loss for this task
            avg_task_query_loss = task_query_loss / len(task_on_device.query_batches)
            
            # Compute gradient of average query loss w.r.t. fast_params
            query_grads = torch.autograd.grad(
                avg_task_query_loss, 
                fast_params.values(),
                allow_unused=True,
            )
            
            # Apply these gradients to the original model parameters
            # This is the first-order approximation
            for (name, param), grad in zip(model.named_parameters(), query_grads):
                if grad is not None and param.grad is not None:
                    param.grad.add_(grad / len(tasks))
                elif grad is not None:
                    param.grad = grad.clone() / len(tasks)
        
        meta_loss_value = total_query_loss / total_query_batches
        
        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        meta_losses_history.append(meta_loss_value)
        
        # Print every outer step to monitor convergence
        lr = optimizer.param_groups[0]['lr']
        print(f"[meta step {outer_step + 1}/{num_outer_steps}] "
              f"meta-loss={meta_loss_value:.4f} lr={lr:.2e}")
        if wandb_run is not None:
            wandb_run.log(
                {"train/meta_loss": meta_loss_value, "train/lr": lr},
                step=outer_step,
            )
        if val_loader is not None and (outer_step + 1) % val_interval == 0:
            val_loss = meta_validate_v2(
                model=model,
                meta_loader=val_loader,
                inner_lr=inner_lr,
                inner_epochs=inner_epochs,
                num_val_tasks=num_val_tasks,
            )
            print(f"  [Validation] val_loss={val_loss:.4f}")
            if wandb_run is not None:
                wandb_run.log(
                    {"meta/val_loss": val_loss},
                    step=outer_step,
                )
    
    return meta_losses_history


def meta_validate_v2(
    model: nn.Module,
    meta_loader: Iterable[List[MetaTaskV2]],
    inner_lr: float,
    inner_epochs: int = 1,
    num_val_tasks: int = 50,
) -> float:
    """
    V2: Evaluate MUSTANG with multiple epochs over support batches.
    
    Reports validation loss at every step.
    
    Args:
        model: The MUSTANG-trained conditional diffusion model.
        meta_loader: MetaTaskLoaderV2 yielding batches of MetaTaskV2.
        inner_lr: Inner-loop learning rate.
        inner_epochs: Number of times to iterate through all support batches.
        num_val_tasks: Number of task batches to evaluate.
    
    Returns:
        Average query loss across all validation tasks.
    """
    device = next(model.parameters()).device
    buffer_dict = OrderedDict(model.named_buffers())
    task_iter = iter(meta_loader)
    
    model.eval()
    total_loss = 0.0
    total_query_batches = 0
    
    for batch_idx in range(num_val_tasks):
        try:
            tasks = next(task_iter)
        except StopIteration:
            break
        
        batch_loss = 0.0
        batch_query_count = 0
        
        for task in tasks:
            task_on_device = _task_to_device_v2(task, device)
            
            # Need gradients for inner loop even in eval mode
            with torch.enable_grad():
                fast_params = inner_loop_adapt_v2(
                    model=model,
                    task=task_on_device,
                    inner_lr=inner_lr,
                    inner_epochs=inner_epochs,
                    buffers=buffer_dict,
                )
            
            # Evaluate on ALL query batches without gradients
            with torch.no_grad():
                for query_batch in task_on_device.query_batches:
                    query_loss = _forward_with_params(
                        model, query_batch, fast_params, buffer_dict, is_train=0
                    )
                    batch_loss += query_loss.item()
                    batch_query_count += 1
        
        total_loss += batch_loss
        total_query_batches += batch_query_count
        
        # Report validation loss at every step
        avg_batch_loss = batch_loss / batch_query_count if batch_query_count > 0 else 0.0
        print(f"[val step {batch_idx + 1}/{num_val_tasks}] val-loss={avg_batch_loss:.4f}")
    
    model.train()
    avg_val_loss = total_loss / max(total_query_batches, 1)
    print(f"[validation complete] avg val-loss={avg_val_loss:.4f}")
    return avg_val_loss


# =============================================================================
# V3: Meta-Training and Validation with Mask-Based Splitting
# =============================================================================

def inner_loop_adapt_v3(
    model: nn.Module,
    task: MetaTaskV3,
    inner_steps: int,
    inner_lr: float,
    buffers: Optional[OrderedDict] = None,
) -> OrderedDict:
    """
    V3: First-Order MAML inner loop with mask-based support/query split.
    
    Similar to V1's inner_loop_adapt, but uses MetaTaskV3 which has
    full-shape batches with mask-based visibility.
    
    Args:
        model: The conditional diffusion model.
        task: MetaTaskV3 with support_batch and query_batch (both full shape).
        inner_steps: Number of inner-loop gradient steps.
        inner_lr: Inner-loop learning rate.
        buffers: Optional buffer dict for functional_call.
    
    Returns:
        Adapted parameters as OrderedDict.
    """
    device = next(model.parameters()).device
    task = _task_to_device_v3(task, device)
    buffer_dict = buffers if buffers is not None else OrderedDict(model.named_buffers())
    
    # Start with cloned parameters
    fast_params = OrderedDict(
        (name, param.clone()) for name, param in model.named_parameters()
    )

    for step in range(inner_steps):
        # Use conditional diffusion directly on support batch (full shape, mask controls visibility)
        support_loss = _forward_with_params(
            model, task.support_batch, fast_params, buffer_dict, is_train=1
        )
        
        grads = torch.autograd.grad(
            support_loss, 
            fast_params.values(), 
            create_graph=False,  # First-order: no graph through inner loop
            allow_unused=True,
        )
        
        # Detach and update - for first-order MAML
        fast_params = OrderedDict(
            (name, (param - inner_lr * grad).detach().requires_grad_(True) 
             if grad is not None else param.detach().requires_grad_(True))
            for (name, param), grad in zip(fast_params.items(), grads)
        )

    return fast_params


def meta_train_v3(
    model: nn.Module,
    meta_loader: Iterable[List[MetaTaskV3]],
    optimizer: torch.optim.Optimizer,
    inner_steps: int,
    inner_lr: float,
    num_outer_steps: int,
    grad_clip: float = 1.0,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    log_interval: int = 10,
    wandb_run=None,
    # Early stopping parameters
    val_loader: Optional[Iterable[List[MetaTaskV3]]] = None,
    val_interval: int = 20,
    patience: int = 10,
    save_path: Optional[str] = None,
) -> List[float]:
    """
    V3: First-Order MAML with mask-based support/query split.
    
    Uses mask-based splitting to preserve data shape while still
    separating support and query time steps.
    
    Args:
        model: The conditional diffusion model to meta-train.
        meta_loader: MetaTaskLoaderV3 yielding batches of MetaTaskV3.
        optimizer: Optimizer for outer-loop updates.
        inner_steps: Number of inner loop steps.
        inner_lr: Learning rate for inner loop.
        num_outer_steps: Number of outer-loop iterations.
        grad_clip: Max gradient norm for clipping. Set to 0 to disable.
        scheduler: Optional learning rate scheduler.
        log_interval: How often to print progress.
        val_loader: Optional validation loader for early stopping.
        val_interval: How often to validate (in outer steps).
        patience: Number of validations without improvement before stopping.
        save_path: Path to save best model checkpoint.
    
    Returns:
        List of meta-losses for each outer step.
    """
    device = next(model.parameters()).device
    buffer_dict = OrderedDict(model.named_buffers())
    task_iter = iter(meta_loader)
    meta_losses_history = []
    
    # Early stopping state
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    use_early_stopping = val_loader is not None

    model.train()
    for outer_step in range(num_outer_steps):
        tasks = next(task_iter)
        optimizer.zero_grad()
        
        # First-Order MAML: Accumulate gradients from query losses
        total_query_loss = 0.0
        
        for task in tasks:
            task_on_device = _task_to_device_v3(task, device)
            
            fast_params = inner_loop_adapt_v3(
                model=model,
                task=task_on_device,
                inner_steps=inner_steps,
                inner_lr=inner_lr,
                buffers=buffer_dict,
            )
            
            # Compute query loss with adapted params
            query_loss = _forward_with_params(
                model, task_on_device.query_batch, fast_params, buffer_dict, is_train=1
            )
            total_query_loss += query_loss.item()
            
            # Compute gradient of query loss w.r.t. fast_params
            query_grads = torch.autograd.grad(
                query_loss, 
                fast_params.values(),
                allow_unused=True,
            )
            
            # Apply these gradients to the original model parameters
            # This is the first-order approximation
            for (name, param), grad in zip(model.named_parameters(), query_grads):
                if grad is not None and param.grad is not None:
                    param.grad.add_(grad / len(tasks))
                elif grad is not None:
                    param.grad = grad.clone() / len(tasks)
        
        meta_loss_value = total_query_loss / len(tasks)
        
        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        meta_losses_history.append(meta_loss_value)
        
        # Print every outer step to monitor convergence
        lr = optimizer.param_groups[0]['lr']
        print(f"[meta step {outer_step + 1}/{num_outer_steps}] "
              f"meta-loss={meta_loss_value:.4f} lr={lr:.2e}")
        if wandb_run is not None:
            wandb_run.log(
                {"train/meta_loss": meta_loss_value, "train/lr": lr},
                step=outer_step,
            )
        
        # Early stopping validation
        if use_early_stopping and (outer_step + 1) % val_interval == 0:
            val_loss = meta_validate_v3(
                model=model,
                meta_loader=val_loader,
                inner_steps=inner_steps,
                inner_lr=inner_lr,
                num_val_tasks=10,  # Use fewer tasks for speed
            )
            print(f"  [Validation] val_loss={val_loss:.4f} best={best_val_loss:.4f}")
            if wandb_run is not None:
                wandb_run.log(
                    {"meta/val_loss": val_loss},
                    step=outer_step,
                )
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                if save_path:
                    torch.save(best_model_state, save_path)
                print(f"  [Validation] New best model saved!")
            else:
                patience_counter += 1
                print(f"  [Validation] No improvement. Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"\n[Early Stopping] No improvement for {patience} validations. "
                      f"Stopping at step {outer_step + 1}.")
                break
            
            model.train()  # Switch back to training mode
    
    # Restore best model if early stopping was used
    if use_early_stopping and best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n[Early Stopping] Restored best model with val_loss={best_val_loss:.4f}")

    return meta_losses_history


def meta_validate_v3(
    model: nn.Module,
    meta_loader: Iterable[List[MetaTaskV3]],
    inner_steps: int,
    inner_lr: float,
    num_val_tasks: int = 50,
) -> float:
    """
    V3: Evaluate MUSTANG with mask-based support/query split.
    
    Args:
        model: The MUSTANG-trained conditional diffusion model.
        meta_loader: MetaTaskLoaderV3 yielding batches of MetaTaskV3.
        inner_steps: Number of inner loop steps.
        inner_lr: Inner-loop learning rate.
        num_val_tasks: Number of task batches to evaluate.
    
    Returns:
        Average query loss across all validation tasks.
    """
    device = next(model.parameters()).device
    buffer_dict = OrderedDict(model.named_buffers())
    task_iter = iter(meta_loader)
    
    model.eval()
    total_loss = 0.0
    total_tasks = 0
    
    for batch_idx in range(num_val_tasks):
        try:
            tasks = next(task_iter)
        except StopIteration:
            break
        
        batch_loss = 0.0
        batch_count = 0
        
        for task in tasks:
            task_on_device = _task_to_device_v3(task, device)
            
            # Need gradients for inner loop even in eval mode
            with torch.enable_grad():
                fast_params = inner_loop_adapt_v3(
                    model=model,
                    task=task_on_device,
                    inner_steps=inner_steps,
                    inner_lr=inner_lr,
                    buffers=buffer_dict,
                )
            
            # Evaluate on query set without gradients
            with torch.no_grad():
                query_loss = _forward_with_params(
                    model, task_on_device.query_batch, fast_params, buffer_dict, is_train=0
                )
                batch_loss += query_loss.item()
                batch_count += 1
        
        total_loss += batch_loss
        total_tasks += batch_count
        
        # Report validation loss at every step
        avg_batch_loss = batch_loss / batch_count if batch_count > 0 else 0.0
        print(f"[val step {batch_idx + 1}/{num_val_tasks}] val-loss={avg_batch_loss:.4f}")
    
    model.train()
    avg_val_loss = total_loss / max(total_tasks, 1)
    print(f"[validation complete] avg val-loss={avg_val_loss:.4f}")
    return avg_val_loss
