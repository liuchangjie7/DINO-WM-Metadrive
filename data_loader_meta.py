import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset


# ============================================================
# 1. 训练专用数据集 (切片模式 - Sliced)
# ============================================================
class MetaDriveDataset(Dataset):
    def __init__(self, files, num_hist, num_pred, frameskip=1):
        self.files = files
        self.num_hist = num_hist
        self.num_pred = num_pred
        self.frameskip = frameskip
        self.seq_len = (num_hist + num_pred) * frameskip

        self.indices = []
        for file_idx, fpath in enumerate(self.files):
            try:
                with np.load(fpath) as data:
                    L = data['action'].shape[0]
                if L > self.seq_len:
                    for t in range(0, L - self.seq_len + 1):
                        self.indices.append((file_idx, t))
            except Exception as e:
                print(f"Skipping broken file {fpath}: {e}")

        np.random.shuffle(self.indices)
        print(f"[Train Dataset] Loaded {len(self.indices)} slices (Shuffled).")

        # 【核心修正 1】定义 raw 维度，并计算拼接后的 action_dim
        self.raw_action_dim = 2
        # action_dim 必须是拼接后的长度，这样 train.py 才能正确初始化 action_encoder
        self.action_dim = self.raw_action_dim * self.frameskip
        self.proprio_dim = 3

        self.compute_stats()

    def compute_stats(self):
        self.transform = lambda x: x

        sample_files = self.files[:100]
        all_acts = []
        all_states = []

        for f in sample_files:
            try:
                with np.load(f) as data:
                    all_acts.append(data['action'])
                    all_states.append(data['state'])
            except:
                pass

        if len(all_acts) > 0:
            all_acts = np.concatenate(all_acts, axis=0)
            all_states = np.concatenate(all_states, axis=0)

            # 【核心修正 2】动作的均值和方差需要 repeat 扩展到拼接后的维度
            raw_action_mean = torch.from_numpy(all_acts.mean(axis=0)).float()
            raw_action_std = torch.from_numpy(all_acts.std(axis=0)).float() + 1e-6

            self.action_mean = raw_action_mean.repeat(self.frameskip)
            self.action_std = raw_action_std.repeat(self.frameskip)

            self.state_mean = torch.from_numpy(all_states.mean(axis=0)).float()
            self.state_std = torch.from_numpy(all_states.std(axis=0)).float() + 1e-6
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.proprio_dim)
            self.state_std = torch.ones(self.proprio_dim)

        self.proprio_mean = self.state_mean
        self.proprio_std = self.state_std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_idx, start_t = self.indices[idx]
        fpath = self.files[file_idx]

        with np.load(fpath) as data:
            all_imgs = data['image']
            all_acts = data['action']
            all_states = data['state']

        end_t = start_t + self.seq_len

        # 【核心修正 3】对不同数据的 frameskip 处理策略

        # 1. 观察 (Images & States)：直接跳帧采样
        imgs = all_imgs[start_t:end_t:self.frameskip]
        states = all_states[start_t:end_t:self.frameskip]

        # 2. 动作 (Actions)：保留所有帧，并在特征维度上进行合并展平
        acts_raw = all_acts[start_t:end_t]  # 取出完整连续的动作
        # reshape 拼接: 比如 seq_len 是 20, frameskip 是 5, 原始 dim 是 2
        # (20, 2) -> (4, 10)，也就是每个时间步包含了 5 帧的动作
        acts = acts_raw.reshape(-1, self.frameskip * self.raw_action_dim)

        # 归一化与 Tensor 转换
        imgs = torch.from_numpy(imgs).float() / 255.0
        imgs = imgs.permute(0, 3, 1, 2)
        acts = torch.from_numpy(acts).float()
        states = torch.from_numpy(states).float()

        obs = {"visual": imgs, "proprio": states}

        return obs, acts, states


# ============================================================
# 2. 验证专用数据集 (全量模式 - Full Trajectory)
# ============================================================
class MetaDriveTrajectoryDataset(Dataset):
    def __init__(self, files, num_hist, num_pred, frameskip=1):
        self.files = files
        self.frameskip = frameskip
        print(f"[Valid Dataset] Loaded {len(self.files)} full trajectories.")

        self.raw_action_dim = 2
        # 虽然 trajectory dataset 原则上返回 raw 数据，但为了与 train 统一接口属性，这里也标明 action_dim
        self.action_dim = self.raw_action_dim * self.frameskip
        self.proprio_dim = 3
        self.compute_stats()

    def compute_stats(self):
        self.transform = lambda x: x
        self.action_mean = torch.zeros(self.action_dim)
        self.action_std = torch.ones(self.action_dim)
        self.state_mean = torch.zeros(self.proprio_dim)
        self.state_std = torch.ones(self.proprio_dim)
        self.proprio_mean = self.state_mean
        self.proprio_std = self.state_std

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 【说明】由于 train.py 中的 openloop_rollout 函数内部写死了对 action 的 rearrange 逻辑
        # 这里只需要无脑返回整条长视频的连续 raw 数据即可，不要在这里做切片或展平
        fpath = self.files[idx]
        with np.load(fpath) as data:
            imgs = data['image']
            acts = data['action']
            states = data['state']

        imgs = torch.from_numpy(imgs).float() / 255.0
        imgs = imgs.permute(0, 3, 1, 2)
        acts = torch.from_numpy(acts).float()
        states = torch.from_numpy(states).float()

        obs = {"visual": imgs, "proprio": states}

        return obs, acts, states, {}


# ============================================================
# 3. Hydra 入口函数
# ============================================================
def build_metadrive_dataset(data_path, num_hist, num_pred, frameskip):
    all_files = sorted(glob.glob(os.path.join(data_path, "*.npz")))
    assert len(all_files) > 0, f"No .npz files found in {data_path}!"

    np.random.shuffle(all_files)
    split_idx = int(len(all_files) * 0.9)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    print(f"Data Split: {len(train_files)} Train files, {len(val_files)} Valid files")

    train_dset = MetaDriveDataset(train_files, num_hist, num_pred, frameskip)
    val_dset = MetaDriveDataset(val_files, num_hist, num_pred, frameskip)

    train_traj_dset = MetaDriveTrajectoryDataset(train_files, num_hist, num_pred, frameskip)
    val_traj_dset = MetaDriveTrajectoryDataset(val_files, num_hist, num_pred, frameskip)

    datasets = {"train": train_dset, "valid": val_dset}
    traj_dsets = {"train": train_traj_dset, "valid": val_traj_dset}

    return datasets, traj_dsets