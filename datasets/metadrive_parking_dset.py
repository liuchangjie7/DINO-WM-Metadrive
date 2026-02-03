import torch
import numpy as np
from pathlib import Path
from einops import rearrange
from .traj_dset import TrajDataset, TrajSlicerDataset
from typing import Optional, Callable, List


class MetaDriveParkingDataset(TrajDataset):
    def __init__(
            self,
            n_rollout: Optional[int] = None,
            transform: Optional[Callable] = None,
            data_path: str = "data/parking_dataset",
            normalize_action: bool = True,
            with_velocity: bool = True,
            specific_files: Optional[List[Path]] = None,  # <--- 修复2: 支持传入特定文件列表
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.transform = transform
        self.normalize_action = normalize_action
        self.with_velocity = with_velocity

        # 1. 确定文件列表
        if specific_files is not None:
            # 如果指定了文件列表（用于切分 Train/Val），直接使用
            self.file_paths = specific_files
        else:
            # 否则扫描文件夹
            self.file_paths = sorted(list(self.data_path.glob("episode_*.npz")))
            if n_rollout is not None:
                self.file_paths = self.file_paths[:n_rollout]

        print(f"Loading metadata for {len(self.file_paths)} trajectories...")

        # 2. 预加载 Action / State
        self.all_actions = []
        self.all_states = []
        self.seq_lengths = []
        self.proprios = []

        for fp in self.file_paths:
            with np.load(fp) as data:
                # data['action']: (T, 2)
                # data['state']: (T, 3)
                act = torch.from_numpy(data['action']).float()
                state = torch.from_numpy(data['state']).float()

                self.all_actions.append(act)
                # Proprioception: 取 state 作为本体感知
                proprio = state.clone()

                self.all_states.append(state)
                self.proprios.append(proprio)
                self.seq_lengths.append(len(act))

        if len(self.seq_lengths) == 0:
            print(f"⚠️ Warning: No trajectories loaded from {self.data_path}!")
            # 防止空数据集报错，给一些 dummy values
            self.action_dim = 2
            self.proprio_dim = 3
            self.state_dim = 3
            return

        print(f"Loaded {len(self.seq_lengths)} episodes.")

        # 3. 计算归一化统计量
        flat_actions = torch.cat(self.all_actions, dim=0)
        flat_proprios = torch.cat(self.proprios, dim=0)
        flat_states = torch.cat(self.all_states, dim=0)

        # === 🔥 修复1: 显式定义 DINO-WM 需要的维度属性 ===
        self.action_dim = flat_actions.shape[-1]
        self.proprio_dim = flat_proprios.shape[-1]
        self.state_dim = flat_states.shape[-1]
        # =================================================

        self.action_mean = flat_actions.mean(dim=0)
        self.action_std = flat_actions.std(dim=0) + 1e-6

        self.proprio_mean = flat_proprios.mean(dim=0)
        self.proprio_std = flat_proprios.std(dim=0) + 1e-6

        self.state_mean = flat_states.mean(dim=0)
        self.state_std = flat_states.std(dim=0) + 1e-6

        # 4. 执行归一化
        if self.normalize_action:
            for i in range(len(self.all_actions)):
                self.all_actions[i] = (self.all_actions[i] - self.action_mean) / self.action_std
                self.proprios[i] = (self.proprios[i] - self.proprio_mean) / self.proprio_std

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_frames(self, idx, frames):
        fp = self.file_paths[idx]
        with np.load(fp) as data:
            raw_imgs = data['image'][frames]  # (L, 224, 224, 3)

        # BGR -> RGB
        raw_imgs = raw_imgs[..., ::-1].copy()
        # Normalize 0-1
        imgs_tensor = torch.from_numpy(raw_imgs).float() / 255.0
        # (T, H, W, C) -> (T, C, H, W)
        imgs_tensor = rearrange(imgs_tensor, "t h w c -> t c h w")

        if self.transform:
            imgs_tensor = self.transform(imgs_tensor)

        act = self.all_actions[idx][frames]
        state = self.all_states[idx][frames]
        proprio = self.proprios[idx][frames]

        obs = {"visual": imgs_tensor, "proprio": proprio}

        return obs, act, state, {'shape': 'car'}

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)


# === 工厂函数：用于创建训练集和验证集切片 ===
def load_metadrive_parking_slice_train_val(
        transform,
        n_rollout=None,
        data_path="data/parking_dino_dataset",
        normalize_action=True,
        split_ratio=0.9,
        num_hist=0,
        num_pred=0,
        frameskip=0,
        with_velocity=True,
):
    # 1. 扫描所有文件
    all_files = sorted(list(Path(data_path).glob("episode_*.npz")))
    if n_rollout:
        all_files = all_files[:n_rollout]

    total_len = len(all_files)
    if total_len == 0:
        raise ValueError(f"No .npz files found in {data_path}")

    # 2. 切分文件列表
    train_len = int(total_len * split_ratio)
    # 确保至少有一个验证样本，除非总数只有1
    if train_len == total_len and total_len > 1:
        train_len -= 1

    train_files = all_files[:train_len]
    val_files = all_files[train_len:]

    # 3. 分别实例化 Dataset，传入具体的文件列表
    # 这样 __init__ 就不会去扫描整个文件夹了
    train_dset = MetaDriveParkingDataset(
        transform=transform,
        data_path=data_path,
        normalize_action=normalize_action,
        with_velocity=with_velocity,
        specific_files=train_files  # <--- 传入切分后的列表
    )

    val_dset = MetaDriveParkingDataset(
        transform=transform,
        data_path=data_path,
        normalize_action=normalize_action,
        with_velocity=with_velocity,
        specific_files=val_files  # <--- 传入切分后的列表
    )

    print(f"Split dataset: {len(train_dset)} train, {len(val_dset)} val")

    num_frames = num_hist + num_pred
    train_slices = TrajSlicerDataset(train_dset, num_frames, frameskip)
    val_slices = TrajSlicerDataset(val_dset, num_frames, frameskip)

    datasets = {}
    datasets["train"] = train_slices
    datasets["valid"] = val_slices

    traj_dset = {}
    traj_dset["train"] = train_dset
    traj_dset["valid"] = val_dset

    return datasets, traj_dset