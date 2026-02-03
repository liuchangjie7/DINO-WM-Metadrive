import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset


# ============================================================
# 1. 训练专用数据集 (切片模式 - Sliced)
#    【核心修改】只返回 3 个值，满足 train.py 的需求
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
                # 只读取头信息，加快扫描速度
                with np.load(fpath) as data:
                    L = data['action'].shape[0]
                if L > self.seq_len:
                    for t in range(0, L - self.seq_len + 1):
                        self.indices.append((file_idx, t))
            except Exception as e:
                print(f"Skipping broken file {fpath}: {e}")

        np.random.shuffle(self.indices)
        print(f"[Train Dataset] Loaded {len(self.indices)} slices (Shuffled).")

        self.action_dim = 2
        self.proprio_dim = 3

        self.compute_stats()

    def compute_stats(self):
        # 定义 transform 为“什么都不做”，防止 preprocessor 报错
        self.transform = lambda x: x

        # 计算均值方差 (采样计算)
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

            self.action_mean = torch.from_numpy(all_acts.mean(axis=0)).float()
            self.action_std = torch.from_numpy(all_acts.std(axis=0)).float() + 1e-6
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

        # 采样
        imgs = all_imgs[start_t:end_t:self.frameskip]
        acts = all_acts[start_t:end_t:self.frameskip]
        states = all_states[start_t:end_t:self.frameskip]

        # 归一化
        imgs = torch.from_numpy(imgs).float() / 255.0
        imgs = imgs.permute(0, 3, 1, 2)
        acts = torch.from_numpy(acts).float()
        states = torch.from_numpy(states).float()

        obs = {"visual": imgs, "proprio": states}

        # 【核心修正】这里只返回 3 个值！train.py 只能解包 3 个！
        return obs, acts, states


# ============================================================
# 2. 验证专用数据集 (全量模式 - Full Trajectory)
#    【核心保持】返回 4 个值，满足 plan_meta.py 的需求
# ============================================================
class MetaDriveTrajectoryDataset(Dataset):
    def __init__(self, files, num_hist, num_pred, frameskip=1):
        self.files = files
        self.frameskip = frameskip
        print(f"[Valid Dataset] Loaded {len(self.files)} full trajectories.")

        self.action_dim = 2
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

        # 【核心保持】这里保持返回 4 个值，因为 plan_meta.py 需要解包 info
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

    # train.py 使用这个 (3返回值)
    train_dset = MetaDriveDataset(train_files, num_hist, num_pred, frameskip)
    val_dset = MetaDriveDataset(val_files, num_hist, num_pred, frameskip)

    # plan_meta.py 使用这个 (4返回值)
    train_traj_dset = MetaDriveTrajectoryDataset(train_files, num_hist, num_pred, frameskip)
    val_traj_dset = MetaDriveTrajectoryDataset(val_files, num_hist, num_pred, frameskip)

    datasets = {"train": train_dset, "valid": val_dset}
    traj_dsets = {"train": train_traj_dset, "valid": val_traj_dset}

    return datasets, traj_dsets