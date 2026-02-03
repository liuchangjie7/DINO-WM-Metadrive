import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset


# ============================================================
# 1. 训练专用数据集 (切片模式 - Sliced)
#    特点：将长视频切成无数个小片段，用于训练 Transformer
# ============================================================
class MetaDriveDataset(Dataset):
    def __init__(self, files, num_hist, num_pred, frameskip=1):
        self.files = files
        self.num_hist = num_hist
        self.num_pred = num_pred
        self.frameskip = frameskip
        # 计算需要的切片长度
        self.seq_len = (num_hist + num_pred) * frameskip

        # 预计算所有合法的切片索引
        self.indices = []
        # print(f"Scanning {len(self.files)} files for training slices...")
        for file_idx, fpath in enumerate(self.files):
            try:
                # 只读取头信息，加快扫描速度
                with np.load(fpath) as data:
                    L = data['action'].shape[0]

                # 如果轨迹长度足够切片
                if L > self.seq_len:
                    # 记录所有可能的起始点 (File Index, Start Time)
                    for t in range(0, L - self.seq_len + 1):
                        self.indices.append((file_idx, t))
            except Exception as e:
                print(f"Skipping broken file {fpath}: {e}")

        # 【关键修复 1】强制打乱数据！
        # 防止模型在一个 Batch 里只看到直道，或者按顺序背诵轨迹
        np.random.shuffle(self.indices)
        print(f"[Train Dataset] Loaded {len(self.indices)} slices (Shuffled).")

        # 定义维度供 train.py 读取
        self.action_dim = 2  # Steering, Throttle
        self.proprio_dim = 3  # Velocity(2) + Angular(1)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_idx, start_t = self.indices[idx]
        fpath = self.files[file_idx]

        with np.load(fpath) as data:
            # 读取原始数据
            all_imgs = data['image']  # (T, 224, 224, 3)
            all_acts = data['action']
            all_states = data['state']

        end_t = start_t + self.seq_len

        # 【采样】根据 frameskip 跳帧读取
        imgs = all_imgs[start_t:end_t:self.frameskip]
        acts = all_acts[start_t:end_t:self.frameskip]
        states = all_states[start_t:end_t:self.frameskip]

        # --- 数据预处理 ---
        # 1. 图片归一化: (T, H, W, C) -> (T, C, H, W) float 0-1
        imgs = torch.from_numpy(imgs).float() / 255.0
        imgs = imgs.permute(0, 3, 1, 2)

        # 2. 动作和状态转 Float
        acts = torch.from_numpy(acts).float()
        states = torch.from_numpy(states).float()

        # 【关键修复 2】构造 Obs 字典
        # 必须包含 'visual' 和 'proprio'，否则模型会报错 KeyError
        obs = {
            "visual": imgs,
            "proprio": states
        }

        # 返回 train.py 期望的三元组
        return obs, acts, states


# ============================================================
# 2. 验证专用数据集 (全量模式 - Full Trajectory)
#    特点：返回整条长视频，用于 openloop_rollout 测试长程预测能力
# ============================================================
class MetaDriveTrajectoryDataset(Dataset):
    def __init__(self, files, num_hist, num_pred, frameskip=1):
        self.files = files
        self.frameskip = frameskip  # 验证时主要返回原始数据
        print(f"[Valid Dataset] Loaded {len(self.files)} full trajectories.")

        self.action_dim = 2
        self.proprio_dim = 3

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fpath = self.files[idx]

        with np.load(fpath) as data:
            # 【关键修复 3】直接返回整条轨迹，不切片！
            # 这样 openloop_rollout 就能找到足够长的数据，不会死循环
            imgs = data['image']
            acts = data['action']
            states = data['state']

        # 同样的归一化处理
        imgs = torch.from_numpy(imgs).float() / 255.0
        imgs = imgs.permute(0, 3, 1, 2)

        acts = torch.from_numpy(acts).float()
        states = torch.from_numpy(states).float()

        obs = {
            "visual": imgs,
            "proprio": states
        }

        return obs, acts, states


# ============================================================
# 3. Hydra 入口函数
# ============================================================
def build_metadrive_dataset(data_path, num_hist, num_pred, frameskip):
    """
    Hydra 会调用这个函数来初始化数据集
    """
    # 扫描所有 .npz 文件
    all_files = sorted(glob.glob(os.path.join(data_path, "*.npz")))
    assert len(all_files) > 0, f"No .npz files found in {data_path}!"

    # 划分训练集和验证集 (90% : 10%)
    # 为了保证验证集也有多样性，这里也可以打乱一下文件列表
    np.random.shuffle(all_files)

    split_idx = int(len(all_files) * 0.9)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    print(f"Data Split: {len(train_files)} Train files, {len(val_files)} Valid files")

    # 1. 实例化训练用的数据集 (Slice Mode)
    # self.datasets 用的就是这个
    train_dset = MetaDriveDataset(train_files, num_hist, num_pred, frameskip)
    val_dset = MetaDriveDataset(val_files, num_hist, num_pred, frameskip)

    # 2. 实例化验证 Rollout 用的数据集 (Trajectory Mode)
    # self.traj_dsets 用的就是这个
    train_traj_dset = MetaDriveTrajectoryDataset(train_files, num_hist, num_pred, frameskip)
    val_traj_dset = MetaDriveTrajectoryDataset(val_files, num_hist, num_pred, frameskip)

    # 返回两个字典，符合 train.py 的解包逻辑
    datasets = {"train": train_dset, "valid": val_dset}
    traj_dsets = {"train": train_traj_dset, "valid": val_traj_dset}

    return datasets, traj_dsets