import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset


# ============================================================
# 1. 训练/验证专用数据集 (滑动窗口切片模式)
#    返回 3 个值: obs, acts, states
# ============================================================
class ParkingDataset(Dataset):
    def __init__(self, files, num_hist, num_pred, frameskip=1):
        self.files = files
        self.num_hist = num_hist
        self.num_pred = num_pred
        self.frameskip = frameskip

        # 即使视频短，也需要满足 Transformer 单次输入所需的固定序列长度
        self.seq_len = (num_hist + num_pred) * frameskip

        self.indices = []
        for file_idx, fpath in enumerate(self.files):
            try:
                with np.load(fpath) as data:
                    L = data['action'].shape[0]

                # 只有当视频长度大于所需窗口时，才进行滑动窗口切片提取
                if L > self.seq_len:
                    for t in range(0, L - self.seq_len + 1):
                        self.indices.append((file_idx, t))
            except Exception as e:
                print(f"Skipping broken file {fpath}: {e}")

        np.random.shuffle(self.indices)
        print(f"[Parking Dataset] Loaded {len(self.indices)} valid slices (Shuffled).")

        # ----------------------------------------------------
        # ⚠️ 请根据你的 Parking 数据集实际维度修改以下两个值:
        # ----------------------------------------------------
        self.raw_action_dim = 2  # 比如：转向, 油门/刹车
        self.proprio_dim = 3  # 比如：速度, 航向角等

        # 拼接后的动作维度 (供 train.py 初始化 action_encoder 使用)
        self.action_dim = self.raw_action_dim * self.frameskip

        self.compute_stats()

    def compute_stats(self):
        self.transform = lambda x: x

        sample_files = self.files[:min(100, len(self.files))]
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

            # 动作的均值和方差需要 repeat 扩展到拼接后的维度
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

        # 1. 观察特征 (Images & States)：直接跳帧采样
        imgs = all_imgs[start_t:end_t:self.frameskip]
        states = all_states[start_t:end_t:self.frameskip]

        # 2. 动作特征 (Actions)：保留所有帧，并在特征维度上进行合并展平
        acts_raw = all_acts[start_t:end_t]
        acts = acts_raw.reshape(-1, self.frameskip * self.raw_action_dim)

        # 归一化与 Tensor 转换
        imgs = torch.from_numpy(imgs).float() / 255.0
        imgs = imgs.permute(0, 3, 1, 2)
        acts = torch.from_numpy(acts).float()
        states = torch.from_numpy(states).float()

        obs = {"visual": imgs, "proprio": states}

        return obs, acts, states


# ============================================================
# 2. 验证专用数据集 (全量轨迹模式，用于 Rollout)
#    返回 4 个值: obs, acts, states, info
# ============================================================
class ParkingTrajectoryDataset(Dataset):
    def __init__(self, files, num_hist, num_pred, frameskip=1):
        self.files = files
        self.frameskip = frameskip
        print(f"[Parking Trajectory] Loaded {len(self.files)} full episodes.")

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

        imgs = torch.from_numpy(imgs).float() / 255.0
        imgs = imgs.permute(0, 3, 1, 2)
        acts = torch.from_numpy(acts).float()
        states = torch.from_numpy(states).float()

        obs = {"visual": imgs, "proprio": states}

        # 注意：这里直接返回未经 reshape 的 acts，
        # 因为 train.py 中的 openloop_rollout 内部自带切片和 rearrange 逻辑
        return obs, acts, states, {}


# ============================================================
# 3. Hydra 入口函数
# ============================================================
def build_parking_dataset(data_path, num_hist, num_pred, frameskip):
    """
    入口函数：扫描 parking 文件夹，划分训练集/验证集，并返回 Dataset 实例
    """
    # 扫描指定目录下的所有 .npz 文件
    all_files = sorted(glob.glob(os.path.join(data_path, "*.npz")))
    assert len(all_files) > 0, f"No .npz files found in {data_path}!"

    np.random.shuffle(all_files)
    split_idx = int(len(all_files) * 0.9)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    print(f"Parking Data Split: {len(train_files)} Train files, {len(val_files)} Valid files")

    # 训练/验证切片数据集 (返回 3 个值)
    train_dset = ParkingDataset(train_files, num_hist, num_pred, frameskip)
    val_dset = ParkingDataset(val_files, num_hist, num_pred, frameskip)

    # 验证 Rollout 全集 (返回 4 个值)
    train_traj_dset = ParkingTrajectoryDataset(train_files, num_hist, num_pred, frameskip)
    val_traj_dset = ParkingTrajectoryDataset(val_files, num_hist, num_pred, frameskip)

    datasets = {"train": train_dset, "valid": val_dset}
    traj_dsets = {"train": train_traj_dset, "valid": val_traj_dset}

    return datasets, traj_dsets