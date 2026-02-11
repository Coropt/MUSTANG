import torch
import numpy as np

def apply_pretrained_mask(observed_mask, target_missing_rate, rng=None):
    if target_missing_rate is None:
        return observed_mask
    if target_missing_rate < 0 or target_missing_rate > 1:
        raise ValueError("target_missing_rate must be in [0, 1].")
    total = observed_mask.size
    if total == 0:
        return observed_mask
    mask = observed_mask.astype(bool)
    current_missing = total - int(mask.sum())
    target_missing = int(round(total * float(target_missing_rate)))
    if target_missing <= current_missing:
        return mask
    num_to_mask = target_missing - current_missing
    flat = mask.reshape(-1)
    observed_indices = np.flatnonzero(flat)
    if observed_indices.size == 0:
        return mask
    if rng is None:
        rng = np.random.default_rng()
    if num_to_mask > observed_indices.size:
        num_to_mask = observed_indices.size
    chosen = rng.choice(observed_indices, size=num_to_mask, replace=False)
    flat = flat.copy()
    flat[chosen] = False
    return flat.reshape(mask.shape)

def get_randmask(observed_mask, min_miss_ratio=0.0, max_miss_ratio=1.0):
    rand_for_mask = torch.rand_like(observed_mask) * observed_mask
    rand_for_mask = rand_for_mask.reshape(-1)
    sample_ratio = np.random.rand()
    sample_ratio = sample_ratio * (max_miss_ratio - min_miss_ratio) + min_miss_ratio
    num_observed = observed_mask.sum().item()
    num_masked = round(num_observed * sample_ratio)
    rand_for_mask[rand_for_mask.topk(num_masked).indices] = -1
    cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
    return cond_mask


def get_block_mask(observed_mask, eval_length=16):
    rand_sensor_mask = torch.rand_like(observed_mask)
    randint = np.random.randint
    sample_ratio = np.random.rand()
    sample_ratio = sample_ratio * 0.15
    mask = rand_sensor_mask < sample_ratio
    min_seq = int(eval_length / 2)
    max_seq = int(eval_length * 2)
    for col in range(observed_mask.shape[1]):
        idxs = np.flatnonzero(mask[:, col])
        if not len(idxs):
            continue
        fault_len = min_seq
        if max_seq > min_seq:
            fault_len = fault_len + int(randint(max_seq - min_seq))
        idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs])
        idxs = np.unique(idxs_ext)
        idxs = np.clip(idxs, 0, observed_mask.shape[0] - 1)
        mask[idxs, col] = True
    rand_base_mask = torch.rand_like(observed_mask) < 0.05
    reverse_mask = mask | rand_base_mask
    block_mask = 1 - reverse_mask.to(torch.float32)

    cond_mask = observed_mask.clone()
    cond_mask = block_mask * cond_mask

    return cond_mask
