from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple
import random

import torch
from torch import nn

try:
    # PyTorch 2.x
    from torch.func import functional_call
except ImportError:  # pragma: no cover - fallback for older versions
    from torch.nn.utils.stateless import functional_call


@dataclass
class MetaTask:
    """A single meta-task with support/query batches."""

    support_batch: dict
    query_batch: dict


def _ensure_float_tensor(value: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.float()
    return torch.tensor(value, dtype=torch.float32)


def split_support_query_by_mask(
    observed_data: torch.Tensor,
    observed_mask: torch.Tensor,
    cond_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    support_frac: float = 0.5,
) -> Tuple[dict, dict]:
    """Split support/query by time masks while preserving full tensor shape."""
    del gt_mask

    batch_size, num_steps, num_features = observed_data.shape
    device = observed_data.device

    num_support = max(1, int(num_steps * support_frac))
    num_query = num_steps - num_support
    if num_query < 1:
        num_support = num_steps - 1
        num_query = 1

    perm = torch.randperm(num_steps, device=device)
    support_indices = perm[:num_support]
    query_indices = perm[num_support:]

    support_time_mask = torch.zeros(batch_size, num_steps, num_features, device=device)
    support_time_mask[:, support_indices, :] = 1.0
    query_time_mask = torch.zeros(batch_size, num_steps, num_features, device=device)
    query_time_mask[:, query_indices, :] = 1.0

    support_visible_mask = observed_mask * support_time_mask
    support_cond_mask = cond_mask * support_time_mask
    support_target_mask = support_visible_mask - support_cond_mask

    support_batch = {
        "observed_data": observed_data.clone(),
        "observed_mask": support_visible_mask,
        "gt_mask": support_visible_mask,
        "cond_mask": support_cond_mask,
        "timepoints": torch.arange(num_steps, device=device).unsqueeze(0).expand(batch_size, -1).float(),
        "cut_length": torch.zeros(batch_size, dtype=torch.long, device=device),
    }

    query_visible_mask = observed_mask * query_time_mask
    observed_mask_for_query = observed_mask - support_target_mask

    query_batch = {
        "observed_data": observed_data.clone(),
        "observed_mask": observed_mask_for_query,
        "gt_mask": query_visible_mask,
        "cond_mask": support_cond_mask,
        "timepoints": torch.arange(num_steps, device=device).unsqueeze(0).expand(batch_size, -1).float(),
        "cut_length": torch.zeros(batch_size, dtype=torch.long, device=device),
    }

    return support_batch, query_batch


class MultiDatasetMetaTaskLoader:
    """Randomly samples tasks from one or more datasets."""

    def __init__(
        self,
        data_loaders: dict,
        support_frac: float = 0.5,
        tasks_per_batch: int = 4,
    ):
        self.data_loaders = data_loaders
        self.dataset_names = list(data_loaders.keys())
        self.support_frac = support_frac
        self.tasks_per_batch = tasks_per_batch

        self.data_iters = {name: iter(loader) for name, loader in data_loaders.items()}
        self.dataset_sizes = {name: len(loader) for name, loader in data_loaders.items()}

        print(
            f"MultiDatasetMetaTaskLoader initialized with {len(self.dataset_names)} datasets:"
        )
        print("  Support/query split: mask-based (shape preserved)")
        print(f"  Support fraction: {support_frac}")
        for name in self.dataset_names:
            print(f"  - {name}: {self.dataset_sizes[name]} batches")

    def _get_batch(self, dataset_name: str):
        try:
            return next(self.data_iters[dataset_name])
        except StopIteration:
            self.data_iters[dataset_name] = iter(self.data_loaders[dataset_name])
            return next(self.data_iters[dataset_name])

    def _sample_dataset(self) -> str:
        return random.choice(self.dataset_names)

    def _sample_task(self) -> MetaTask:
        dataset_name = self._sample_dataset()
        batch = self._get_batch(dataset_name)

        observed_data = _ensure_float_tensor(batch["observed_data"])
        observed_mask = _ensure_float_tensor(batch["observed_mask"])
        cond_mask = _ensure_float_tensor(batch["cond_mask"])
        gt_mask = _ensure_float_tensor(batch["gt_mask"])

        support_batch, query_batch = split_support_query_by_mask(
            observed_data=observed_data,
            observed_mask=observed_mask,
            cond_mask=cond_mask,
            gt_mask=gt_mask,
            support_frac=self.support_frac,
        )
        return MetaTask(support_batch=support_batch, query_batch=query_batch)

    def __iter__(self) -> Iterable[List[MetaTask]]:
        while True:
            yield [self._sample_task() for _ in range(self.tasks_per_batch)]


class MetaTaskLoader:
    """Single-dataset wrapper around ``MultiDatasetMetaTaskLoader``."""

    def __init__(
        self,
        data_loader,
        support_frac: float = 0.5,
        tasks_per_batch: int = 4,
    ):
        self._multi_loader = MultiDatasetMetaTaskLoader(
            data_loaders={"default": data_loader},
            support_frac=support_frac,
            tasks_per_batch=tasks_per_batch,
        )

    def __iter__(self) -> Iterable[List[MetaTask]]:
        return iter(self._multi_loader)


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _task_to_device(task: MetaTask, device: torch.device) -> MetaTask:
    return MetaTask(
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
    """First-order MAML adaptation on the support batch."""
    device = next(model.parameters()).device
    task = _task_to_device(task, device)
    buffer_dict = buffers if buffers is not None else OrderedDict(model.named_buffers())

    fast_params = OrderedDict((name, param.clone()) for name, param in model.named_parameters())

    for _ in range(inner_steps):
        support_loss = _forward_with_params(
            model, task.support_batch, fast_params, buffer_dict, is_train=1
        )
        grads = torch.autograd.grad(
            support_loss,
            fast_params.values(),
            create_graph=False,
            allow_unused=True,
        )
        fast_params = OrderedDict(
            (
                name,
                (param - inner_lr * grad).detach().requires_grad_(True)
                if grad is not None
                else param.detach().requires_grad_(True),
            )
            for (name, param), grad in zip(fast_params.items(), grads)
        )

    return fast_params


def meta_validate(
    model: nn.Module,
    meta_loader: Iterable[List[MetaTask]],
    inner_steps: int,
    inner_lr: float,
    num_val_tasks: int = 50,
) -> float:
    """Evaluate meta-objective on sampled validation tasks."""
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
            task_on_device = _task_to_device(task, device)
            with torch.enable_grad():
                fast_params = inner_loop_adapt(
                    model=model,
                    task=task_on_device,
                    inner_steps=inner_steps,
                    inner_lr=inner_lr,
                    buffers=buffer_dict,
                )

            with torch.no_grad():
                query_loss = _forward_with_params(
                    model,
                    task_on_device.query_batch,
                    fast_params,
                    buffer_dict,
                    is_train=0,
                )
                batch_loss += query_loss.item()
                batch_count += 1

        total_loss += batch_loss
        total_tasks += batch_count
        avg_batch_loss = batch_loss / batch_count if batch_count > 0 else 0.0
        print(f"[val step {batch_idx + 1}/{num_val_tasks}] val-loss={avg_batch_loss:.4f}")

    model.train()
    avg_val_loss = total_loss / max(total_tasks, 1)
    print(f"[validation complete] avg val-loss={avg_val_loss:.4f}")
    return avg_val_loss


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
    wandb_run=None,
    val_loader: Optional[Iterable[List[MetaTask]]] = None,
    val_interval: int = 20,
    patience: int = 10,
    save_path: Optional[str] = None,
    num_val_tasks: int = 10,
) -> List[float]:
    """Run first-order MAML with optional early stopping."""
    device = next(model.parameters()).device
    buffer_dict = OrderedDict(model.named_buffers())
    task_iter = iter(meta_loader)
    meta_losses_history: List[float] = []

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    use_early_stopping = val_loader is not None

    model.train()
    for outer_step in range(num_outer_steps):
        tasks = next(task_iter)
        optimizer.zero_grad()

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

            query_loss = _forward_with_params(
                model,
                task_on_device.query_batch,
                fast_params,
                buffer_dict,
                is_train=1,
            )
            total_query_loss += query_loss.item()

            query_grads = torch.autograd.grad(
                query_loss,
                fast_params.values(),
                allow_unused=True,
            )

            for (_, param), grad in zip(model.named_parameters(), query_grads):
                if grad is not None and param.grad is not None:
                    param.grad.add_(grad / len(tasks))
                elif grad is not None:
                    param.grad = grad.clone() / len(tasks)

        meta_loss_value = total_query_loss / len(tasks)

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        meta_losses_history.append(meta_loss_value)

        should_log = (outer_step + 1) % max(1, log_interval) == 0 or outer_step == 0
        if should_log:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"[meta step {outer_step + 1}/{num_outer_steps}] "
                f"meta-loss={meta_loss_value:.4f} lr={lr:.2e}"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {"train/meta_loss": meta_loss_value, "train/lr": lr},
                    step=outer_step,
                )

        if use_early_stopping and (outer_step + 1) % max(1, val_interval) == 0:
            val_loss = meta_validate(
                model=model,
                meta_loader=val_loader,
                inner_steps=inner_steps,
                inner_lr=inner_lr,
                num_val_tasks=num_val_tasks,
            )
            print(f"  [Validation] val_loss={val_loss:.4f} best={best_val_loss:.4f}")
            if wandb_run is not None:
                wandb_run.log({"meta/val_loss": val_loss}, step=outer_step)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                if save_path:
                    torch.save(best_model_state, save_path)
                print("  [Validation] New best model saved!")
            else:
                patience_counter += 1
                print(
                    f"  [Validation] No improvement. "
                    f"Patience: {patience_counter}/{patience}"
                )

            if patience_counter >= patience:
                print(
                    f"\n[Early Stopping] No improvement for {patience} validations. "
                    f"Stopping at step {outer_step + 1}."
                )
                break

            model.train()

    if use_early_stopping and best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(
            f"\n[Early Stopping] Restored best model with val_loss={best_val_loss:.4f}"
        )

    return meta_losses_history
