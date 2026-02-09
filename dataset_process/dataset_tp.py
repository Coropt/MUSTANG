import os
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import pickle
import torch
from dataset_process.utils import apply_pretrained_mask


def get_randmask(observed_mask, min_miss_ratio=0., max_miss_ratio=1.):
    rand_for_mask = torch.rand_like(observed_mask) * observed_mask
    rand_for_mask = rand_for_mask.reshape(-1)
    sample_ratio = np.random.rand()
    sample_ratio = sample_ratio * (max_miss_ratio-min_miss_ratio) + min_miss_ratio
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

def sample_mask(shape, p=0.0015, p_noise=0.05, max_seq=1, min_seq=1, rng=None):
    # shape：输出掩码数组的形状。
    # p：初始选择某个位置的概率。
    # p_noise：添加随机噪声的概率。
    # max_seq 和 min_seq：故障序列的最大和最小长度。
    # rng：随机数生成器。如果没有提供，则使用默认的 NumPy 随机数生成器。
    if rng is None:
        rand = np.random.random
        randint = np.random.randint
    else:
        rand = rng.random
        randint = rng.integers
    mask = rand(shape) < p # 概率 p 初始化一个布尔掩码数组 mask，形状为 shape。mask 中的值根据概率 p 进行随机选择，True 表示被选中。
    for col in range(mask.shape[1]):
        idxs = np.flatnonzero(mask[:, col]) # 找出当前列中所有值为 True 的索引 idxs。
        if not len(idxs): # # 如果没有值为True的索引，跳过该列
            continue
        fault_len = min_seq # 5
        if max_seq > min_seq: # 15
            fault_len = fault_len + int(randint(max_seq - min_seq)) # 5+x
        idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs]) # 扩展索引以包括故障序列中的所有索引
        idxs = np.unique(idxs_ext) # 去重并确保索引不超过数组边界
        idxs = np.clip(idxs, 0, shape[0] - 1)
        mask[idxs, col] = True # 将这些索引位置的值设为True
    mask = mask | (rand(mask.shape) < p_noise) # 增加随机噪声，根据概率p_noise添加更多True值
    print("eval_mask True值的个数:", np.sum(mask))
    return mask.astype('uint8')



class TP_Dataset(Dataset):
    def __init__(self, eval_length=16, seed=1, mode="train", val_len=0.1, test_len=0.2, test_missing='point',
                training_missing='point', missing_ratio=0.1, pretrain_missing_rate=None):
        self.eval_length = eval_length
        self.training_missing = training_missing
        self.mode = mode
        self.seed = seed
        self.use_index = []
        self.cut_length = []

        # 筛选12个站点
        selected_stations = [
            '04178000', '04182000', '04183000', '04183500', '04185318', '04186500',
            '04188100', '04190000', '04191058', '04191444', '04191500', '04192500'
        ]
        
        df = pd.read_csv('./original_data/Phosphorus/TP_pooled.csv', index_col=0)
        df.index = pd.to_datetime(df.index) # 将索引转换为日期时间格式
        df.columns = df.columns.astype(str)
        df = df[selected_stations]
        
        datetime_idx = sorted(df.index)
        date_range = pd.date_range(datetime_idx[0], datetime_idx[-1], freq='1D')
        df = df.reindex(index=date_range) 

        start_date = '2015/4/15'
        end_date = '2022/9/30'  # 统一时间范围
        
        #### log transformation #####
        df = df.loc[start_date:end_date, :] # 特定时间范围的数据
        
        data_shape = df.values.shape

        # 原始观测 mask（用于评估时的 ground truth）
        original_ob_mask = ~np.isnan(df.values)

        total_entries = np.prod(df.values.shape)
        original_missing = np.isnan(df.values).sum()
        original_missing_rate = original_missing / total_entries

        ob_mask = original_ob_mask.copy()
        pretrain_mask_enabled = pretrain_missing_rate is not None and pretrain_missing_rate > 0

        df.fillna(method='ffill', axis=0, inplace=True)
        # print("重新索引并填充缺失值后的数据:")
        # print(df.head())
        

        # SEED = 56789
        SEED = seed
        self.rng = np.random.default_rng(SEED)
        if pretrain_missing_rate is not None:
            mask_rng = np.random.default_rng(SEED)
            ob_mask = apply_pretrained_mask(ob_mask, pretrain_missing_rate, rng=mask_rng)
        num_features = data_shape[1]
        if pretrain_mask_enabled:
            if mode == 'valid':
                if test_missing == 'block':
                    eval_mask = sample_mask(shape=data_shape, p=0.0015, p_noise=0.05, min_seq=int(self.eval_length / 2), max_seq=int(self.eval_length * 2), rng=self.rng)
                elif test_missing == 'point':
                    eval_mask = sample_mask(shape=data_shape, p=0.0, p_noise=missing_ratio, min_seq=1, max_seq=1, rng=self.rng)
            else:
                eval_mask = np.zeros_like(ob_mask, dtype='uint8')
        else:
            if test_missing == 'block':
                eval_mask = sample_mask(shape=data_shape, p=0.0015, p_noise=0.05, min_seq=int(self.eval_length / 2), max_seq=int(self.eval_length * 2), rng=self.rng)
            elif test_missing == 'point':
                eval_mask = sample_mask(shape=data_shape, p=0.0, p_noise=missing_ratio, min_seq=1, max_seq=1, rng=self.rng)
        
        eval_mask_applied = eval_mask.astype(bool)
        total_eval_entries = np.prod(eval_mask.shape)
        eval_missing = np.sum(eval_mask_applied)

        gt_mask = (1-(eval_mask | (1-ob_mask))).astype('uint8')

        self.train_mean = np.zeros(num_features)
        self.train_std = np.zeros(num_features)
        for k in range(num_features):
            tmp_data = df.iloc[:, k][ob_mask[:, k] == 1]
            self.train_mean[k] = tmp_data.mean()
            self.train_std[k] = tmp_data.std()
        path = "./original_data/Phosphorus/tp_meanstd.pk"
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump([self.train_mean, self.train_std], f)

        val_start = int((1 - val_len - test_len) * len(df))
        test_start = int((1 - test_len) * len(df))
        
        full_data = (
             (df.fillna(0).values - self.train_mean) / self.train_std
        )
        c_data = full_data * ob_mask

        eval_observed_mask = original_ob_mask if pretrain_mask_enabled else ob_mask
        
        if mode == 'train':
            self.observed_mask = ob_mask[:val_start]
            self.gt_mask = gt_mask[:val_start]
            self.observed_data = c_data[:val_start]
            self.target_data = full_data[:val_start]
        elif mode == 'valid':
            self.observed_mask = eval_observed_mask[val_start: test_start]
            self.gt_mask = gt_mask[val_start: test_start]
            self.observed_data = c_data[val_start: test_start]
            self.target_data = full_data[val_start: test_start]
        elif mode == 'test':
            self.observed_mask = eval_observed_mask[test_start:]
            self.gt_mask = gt_mask[test_start:]
            self.observed_data = c_data[test_start:]
            self.target_data = full_data[test_start:]
        current_length = len(self.observed_mask) - self.eval_length + 1

        if mode == "test":
            # 如果数据长度不是 eval_length 的倍数，最后一段是不完整的 → 补一个“伪样本”，但会记录其中有多少位置是填充的
            # 这个“填充的长度”就存到了 cut_length 中
            n_sample = len(self.observed_data) // self.eval_length
            c_index = np.arange(
                0, 0 + self.eval_length * n_sample, self.eval_length
            )
            self.use_index += c_index.tolist()
            self.cut_length += [0] * len(c_index)
            if len(self.observed_data) % self.eval_length != 0:
                self.use_index += [current_length - 1]
                self.cut_length += [self.eval_length - len(self.observed_data) % self.eval_length]
        elif mode != "test":
            self.use_index = np.arange(current_length)
            self.cut_length = [0] * len(self.use_index)

    def __getitem__(self, org_index):
        index = self.use_index[org_index]
        ob_data = self.observed_data[index: index + self.eval_length]
        target_data = self.target_data[index: index + self.eval_length]
        ob_mask = self.observed_mask[index: index + self.eval_length]
        ob_mask_t = torch.tensor(ob_mask).float()
        gt_mask = self.gt_mask[index: index + self.eval_length]
        
        if self.mode == 'test':
            # test 时使用 gt_mask，模型看不到评估点（与 CD2-TSI 保持一致）
            cond_mask = torch.tensor(gt_mask).to(torch.float32)
        elif self.mode == 'valid':
            cond_mask = torch.tensor(gt_mask).to(torch.float32)
        else:
            if self.training_missing != 'point':
                cond_mask = get_block_mask(ob_mask_t, eval_length=self.eval_length)
            else:
                cond_mask = get_randmask(ob_mask_t)
        
        s = {
            "observed_data": ob_data,
            "target_data": target_data,
            "observed_mask": ob_mask,
            "gt_mask": gt_mask,
            "timepoints": np.arange(self.eval_length),
            "cut_length": self.cut_length[org_index],
            "cond_mask": cond_mask,
        }

        return s

    def __len__(self):
        return len(self.use_index)


def get_dataloader_TP(sequence_length, batch_size, device, seed=45678, val_len=0.1, test_len=0.2, num_workers=4, 
                   test_missing='block', training_missing='block', missing_ratio=0.1, pretrain_missing_rate=None):
    dataset = TP_Dataset(eval_length=sequence_length, seed=seed, mode="train", val_len=val_len, test_len=test_len,
                             test_missing=test_missing, training_missing=training_missing, missing_ratio=missing_ratio,
                             pretrain_missing_rate=pretrain_missing_rate)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    dataset_test = TP_Dataset(eval_length=sequence_length, seed=seed, mode="test", val_len=val_len, test_len=test_len,
                             test_missing=test_missing, training_missing=training_missing, missing_ratio=missing_ratio,
                             pretrain_missing_rate=pretrain_missing_rate)
    test_loader = DataLoader(
        dataset_test, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    dataset_valid = TP_Dataset(eval_length=sequence_length, seed=seed, mode="valid", val_len=val_len, test_len=test_len,
                             test_missing=test_missing, training_missing=training_missing, missing_ratio=missing_ratio,
                             pretrain_missing_rate=pretrain_missing_rate)
    valid_loader = DataLoader(
        dataset_valid, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    scaler = torch.from_numpy(dataset.train_std).to(device).float()
    mean_scaler = torch.from_numpy(dataset.train_mean).to(device).float()

    return train_loader, valid_loader, test_loader, scaler, mean_scaler
