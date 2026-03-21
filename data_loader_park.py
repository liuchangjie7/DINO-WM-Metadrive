import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset

# ============================================================
# 1. 训练/验证专用数据集 (滑动窗口切片模式 + 内存加速)
# ============================================================
class ParkingDataset(Dataset):
    def __init__(self, files, num_hist, num_pred, frameskip=1):
        self.files = files
        self.num_hist = num_hist
        self.num_pred = num_pred
        self.frameskip = frameskip
        self.seq_len = (num_hist + num_pred) * frameskip

        self.raw_action_dim = 2
        self.proprio_dim = 3
        self.action_dim = self.raw_action_dim * self.frameskip

        self.indices = []
        self.cache = {}  # 🚀 新增：内存缓存字典
        
        print(f"正在将 {len(self.files)} 个文件加载到内存中，请稍等 (可极大加速训练)...")
        for file_idx, fpath in enumerate(self.files):
            try:
                with np.load(fpath) as data:
                    L = data['action'].shape[0]
                    if L > self.seq_len:
                        # 一次性读入内存
                        self.cache[file_idx] = {
                            'image': data['image'],
                            'action': data['action'],
                            'state': data['state']
                        }
                        for t in range(0, L - self.seq_len + 1):
                            self.indices.append((file_idx, t))
            except Exception as e:
                print(f"Skipping broken file {fpath}: {e}")

        np.random.shuffle(self.indices)
        print(f"[Parking Dataset] 成功加载 {len(self.indices)} 个有效切片到内存中.")
        self.compute_stats()

    def compute_stats(self):
        self.transform = lambda x: x
        all_acts = []
        all_states = []

        # 直接从缓存中计算统计量
        sample_keys = list(self.cache.keys())[:min(100, len(self.cache))]
        for k in sample_keys:
            all_acts.append(self.cache[k]['action'])
            all_states.append(self.cache[k]['state'])

        if len(all_acts) > 0:
            all_acts = np.concatenate(all_acts, axis=0)
            all_states = np.concatenate(all_states, axis=0)

            self.raw_action_mean = torch.from_numpy(all_acts.mean(axis=0)).float()
            self.raw_action_std = torch.from_numpy(all_acts.std(axis=0)).float() + 1e-6
            self.action_mean = self.raw_action_mean.repeat(self.frameskip)
            self.action_std = self.raw_action_std.repeat(self.frameskip)
            self.state_mean = torch.from_numpy(all_states.mean(axis=0)).float()
            self.state_std = torch.from_numpy(all_states.std(axis=0)).float() + 1e-6
        else:
            self.raw_action_mean = torch.zeros(self.raw_action_dim)
            self.raw_action_std = torch.ones(self.raw_action_dim)
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
        end_t = start_t + self.seq_len

        # 🚀 从内存极速读取
        data = self.cache[file_idx]
        imgs = data['image'][start_t:end_t:self.frameskip]
        states = data['state'][start_t:end_t:self.frameskip]
        acts_raw = data['action'][start_t:end_t]
        acts = acts_raw.reshape(-1, self.frameskip * self.raw_action_dim)

        # 🛠️ 修复 1：将图像映射到 [-1, 1]，解决画面全灰/发白问题！
        imgs = (torch.from_numpy(imgs).float() / 255.0) * 2.0 - 1.0
        imgs = imgs.permute(0, 3, 1, 2)

        acts = torch.from_numpy(acts).float()
        # 🛠️ 修复 2：训练集补充缺失的动作归一化！
        acts = (acts - self.action_mean) / self.action_std

        states = torch.from_numpy(states).float()
        obs = {"visual": imgs, "proprio": states}

        return obs, acts, states


# ============================================================
# 2. 验证专用数据集 (全量轨迹模式，用于 Rollout)
# ============================================================
class ParkingTrajectoryDataset(Dataset):
    def __init__(self, files, num_hist, num_pred, frameskip=1):
        self.files = files
        self.frameskip = frameskip
        self.raw_action_dim = 2
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
        fpath = self.files[idx]
        with np.load(fpath) as data:
            imgs = data['image']
            acts = data['action']
            states = data['state']

        # 🛠️ 修复 1：图像映射到 [-1, 1]
        imgs = (torch.from_numpy(imgs).float() / 255.0) * 2.0 - 1.0
        imgs = imgs.permute(0, 3, 1, 2)

        acts = torch.from_numpy(acts).float()
        if hasattr(self, 'raw_action_mean'):
            acts = (acts - self.raw_action_mean) / self.raw_action_std
        states = torch.from_numpy(states).float()

        obs = {"visual": imgs, "proprio": states}
        return obs, acts, states, {}


# ============================================================
# 3. Hydra 入口函数
# ============================================================
def build_parking_dataset(data_path, num_hist, num_pred, frameskip):
    all_files = sorted(glob.glob(os.path.join(data_path, "*.npz")))
    assert len(all_files) > 0, f"No .npz files found in {data_path}!"

    np.random.shuffle(all_files)
    split_idx = int(len(all_files) * 0.9)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    print(f"Parking Data Split: {len(train_files)} Train files, {len(val_files)} Valid files")

    train_dset = ParkingDataset(train_files, num_hist, num_pred, frameskip)
    val_dset = ParkingDataset(val_files, num_hist, num_pred, frameskip)

    train_traj_dset = ParkingTrajectoryDataset(train_files, num_hist, num_pred, frameskip)
    val_traj_dset = ParkingTrajectoryDataset(val_files, num_hist, num_pred, frameskip)

    for tdset in [train_traj_dset, val_traj_dset]:
        tdset.raw_action_mean = train_dset.raw_action_mean
        tdset.raw_action_std = train_dset.raw_action_std
        tdset.action_mean = train_dset.action_mean
        tdset.action_std = train_dset.action_std
        tdset.state_mean = train_dset.state_mean
        tdset.state_std = train_dset.state_std
        tdset.proprio_mean = train_dset.proprio_mean
        tdset.proprio_std = train_dset.proprio_std

    datasets = {"train": train_dset, "valid": val_dset}
    traj_dsets = {"train": train_traj_dset, "valid": val_traj_dset}

    return datasets, traj_dsets