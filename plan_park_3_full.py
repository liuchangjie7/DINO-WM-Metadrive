import os
import gym
import cv2
import math
import json
import hydra
import random
import torch
import pickle
import wandb
import logging
import warnings
import numpy as np
import submitit
from itertools import product
from pathlib import Path
from omegaconf import OmegaConf, open_dict

from env.venv import SubprocVectorEnv
from custom_resolvers import replace_slash
from preprocessor import Preprocessor
from planning.evaluator_park import PlanEvaluator
from utils import cfg_to_dict, seed

from metadrive.envs.marl_envs import MultiAgentParkingLotEnv
from metadrive.component.vehicle.vehicle_type import SVehicle

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

ALL_MODEL_KEYS = [
    "encoder",
    "predictor",
    "decoder",
    "proprio_encoder",
    "action_encoder",
]

# ========================================================
# 1. 绘图与环境配置参数 (来自 gen_dataset_parking.py)
# ========================================================
RENDER_RES = (672, 672)
REAL_RES = (224, 224)
BEV_SCALING = 24.0
MAP_CENTER = (29.0, 1.0)

CONTROL_CONFIG = {
    "MAX_FORWARD_SPEED": 3.0, "MAX_REVERSE_SPEED": -2.0, "APPROACH_DIST": 0.8,
    "APPROACH_ANGLE_TOLERANCE": 0.15, "STOP_THRESHOLD": 0.2,
}

PARKING_SLOTS = [
    {"id": 0, "pos": [34.36, 11.00], "heading": -1.57}, {"id": 1, "pos": [30.88, 11.00], "heading": -1.57},
    {"id": 2, "pos": [23.86, -8.50], "heading": 1.57}, {"id": 3, "pos": [31.00, -8.50], "heading": 1.57},
    {"id": 4, "pos": [24.17, 11.00], "heading": -1.57}, {"id": 5, "pos": [27.72, 11.00], "heading": -1.57},
    {"id": 6, "pos": [34.73, -8.50], "heading": 1.57}, {"id": 7, "pos": [27.40, -8.50], "heading": 1.57},
]

def to_px(pos):
    x = (pos[0] - MAP_CENTER[0]) * BEV_SCALING + RENDER_RES[0] / 2
    y = (pos[1] - MAP_CENTER[1]) * BEV_SCALING + RENDER_RES[1] / 2
    return (int(x), int(RENDER_RES[1] - y))

def get_transformed_pts(center, heading, local_pts):
    c, s = math.cos(heading), math.sin(heading)
    world_pts = []
    for x, y in local_pts:
        wx = center[0] + x * c - y * s
        wy = center[1] + x * s + y * c
        world_pts.append(to_px((wx, wy)))
    return np.array(world_pts, dtype=np.int32)

def draw_vehicle_on_img(img, pos, heading, is_ego=False):
    L, W = 4.6, 2.0
    if not is_ego:
        local_box = [(L / 2, W / 2), (L / 2, -W / 2), (-L / 2, -W / 2), (-L / 2, W / 2)]
        pts = get_transformed_pts(pos, heading, local_box)
        cv2.fillPoly(img, [pts], (150, 150, 150), lineType=cv2.LINE_AA)
        return

    # 1. 车尾段 (后 1/3) -> 霓虹紫/品红 (BGR: 200, 50, 180)
    tail_local = [(-L / 6, W / 2), (-L / 6, -W / 2), (-L / 2, -W / 2), (-L / 2, W / 2)]
    tail_pts = get_transformed_pts(pos, heading, tail_local)
    cv2.fillPoly(img, [tail_pts], (0, 128, 255), lineType=cv2.LINE_AA)

    # 2. 车中段 (中 1/3) -> 深石板灰 (BGR: 60, 60, 60)
    mid_local = [(L / 6, W / 2), (L / 6, -W / 2), (-L / 6, -W / 2), (-L / 6, W / 2)]
    mid_pts = get_transformed_pts(pos, heading, mid_local)
    cv2.fillPoly(img, [mid_pts], (100, 100, 100), lineType=cv2.LINE_AA)

    # 3. 车头段 (前 1/3) -> 荧光青/亮葱绿 (BGR: 150, 250, 50)
    head_local = [(L / 2, W / 2), (L / 2, -W / 2), (L / 6, -W / 2), (L / 6, W / 2)]
    head_pts = get_transformed_pts(pos, heading, head_local)
    cv2.fillPoly(img, [head_pts], (255, 200, 100), lineType=cv2.LINE_AA)

    # 4. 全车外轮廓硬描边 (亮白色，让车在任何背景下都能被抠出来)
    body_local = [(L / 2, W / 2), (L / 2, -W / 2), (-L / 2, -W / 2), (-L / 2, W / 2)]
    body_pts = get_transformed_pts(pos, heading, body_local)
    cv2.polylines(img, [body_pts], True, (255, 255, 255), 2, cv2.LINE_AA)

# ========================================================
# 2. 泊车专家策略
# ========================================================
class ParkingPilot:
    def __init__(self):
        self.stage = "APPROACH"
        self.target_slot = None
        self.setup_pose = None
        self.stop_timer = 0

    def set_target(self, slot, vehicle):
        self.target_slot = slot
        tx, ty = slot["pos"]
        th = slot["heading"]
        out_vec = np.array([math.cos(th), math.sin(th)])
        long_offset = 8.5
        curr_h = vehicle.heading_theta
        side_vec = np.array([math.cos(curr_h), math.sin(curr_h)])
        side_offset = 5.0
        setup_xy = np.array([tx, ty]) + out_vec * long_offset + side_vec * side_offset
        self.setup_pose = np.array([setup_xy[0], setup_xy[1], curr_h])
        self.stage = "APPROACH"
        self.stop_timer = 0

    def get_action(self, vehicle):
        heading_vec = np.array([math.cos(vehicle.heading_theta), math.sin(vehicle.heading_theta)])
        velocity_vec = np.array(vehicle.velocity)[:2]
        real_speed = np.dot(velocity_vec, heading_vec)
        ego_pos = vehicle.position
        ego_heading = vehicle.heading_theta

        if self.stage == "APPROACH":
            dist = np.linalg.norm(ego_pos - self.setup_pose[:2])
            if dist < CONTROL_CONFIG["APPROACH_DIST"]:
                self.stage = "STOPPING"
                return self._force_stop(real_speed)
            return self._velocity_control(ego_pos, ego_heading, real_speed, self.setup_pose[:2], self.setup_pose[2], reverse=False)

        if self.stage == "STOPPING":
            self.stop_timer += 1
            if abs(real_speed) < CONTROL_CONFIG["STOP_THRESHOLD"] or self.stop_timer > 20:
                self.stage = "REVERSE"
                return [0.0, 0.0]
            return self._force_stop(real_speed)

        if self.stage == "REVERSE":
            target_pos = np.array(self.target_slot["pos"])
            target_heading = self.target_slot["heading"]
            dist = np.linalg.norm(ego_pos - target_pos)
            head_err = abs(self._normalize_angle(ego_heading - target_heading))
            if dist < 0.15 and head_err < 0.10:
                self.stage = "FINISH"
                return [0.0, 0.0]
            return self._velocity_control(ego_pos, ego_heading, real_speed, target_pos, target_heading, reverse=True)
        return self._force_stop(real_speed)

    def _force_stop(self, real_speed):
        if real_speed > 0.05: return [0.0, -1.0]
        if real_speed < -0.05: return [0.0, 1.0]
        return [0.0, 0.0]

    def _velocity_control(self, curr_pos, curr_heading, real_speed, target_pos, target_heading, reverse=False):
        vec = target_pos - curr_pos
        dist = np.linalg.norm(vec)
        if reverse:
            aim_angle = self._normalize_angle(math.atan2(vec[1], vec[0]) + math.pi)
            alpha = np.clip((dist - 0.4) / 3.0, 0.0, 1.0)
            target_h = alpha * aim_angle + (1 - alpha) * target_heading
            angle_err = self._normalize_angle(target_h - curr_heading)
            steer = np.clip(angle_err * 3.5, -1.0, 1.0) * -1
            max_v = CONTROL_CONFIG["MAX_REVERSE_SPEED"]
            target_v = max_v if dist > 1.2 else max_v * 0.5
        else:
            final_aim_angle = math.atan2(vec[1], vec[0])
            angle_err = self._normalize_angle(final_aim_angle - curr_heading)
            steer = np.clip(angle_err * 2.5, -1.0, 1.0)
            target_v = CONTROL_CONFIG["MAX_FORWARD_SPEED"]
        throttle = np.clip((target_v - real_speed) * 2.0, -1.0, 1.0)
        return [steer, throttle]

    def _normalize_angle(self, angle):
        while angle > math.pi: angle -= 2 * math.pi
        while angle < -math.pi: angle += 2 * math.pi
        return angle

# ========================================================
# 3. 环境 Wrapper适配 (多智能体车位 + 专家策略起步)
# ========================================================
class ParkingDinoWrapper(gym.Wrapper):
    def __init__(self, env, frameskip=1, action_mean=None, action_std=None):
        super().__init__(env)
        self.frameskip = frameskip
        self.static_vehicles = [] 
        self.ego = None
        self.agent_id = None
        self.current_action = [0.0, 0.0]
        self.action_mean = action_mean if action_mean is not None else np.zeros(frameskip * 2)
        self.action_std = action_std if action_std is not None else np.ones(frameskip * 2)

    def update_env(self, env_info):
        if env_info:
            if 'seed' in env_info:
                self.current_seed = int(env_info['seed'])
            if 'steps_to_goal' in env_info:
                self.steps_to_goal = int(env_info['steps_to_goal'])
        return "OK"

    def get_obs_at_pose(self, pose_xyh):
        """生成一张车辆位于特定位置 (面包屑) 的鸟瞰图"""
        old_pos = self.ego.position
        old_heading = self.ego.heading_theta
        old_vel = self.ego.velocity
        
        # 瞬移到面包屑位置
        self.ego.set_position(pose_xyh[:2])
        self.ego.set_heading_theta(pose_xyh[2])
        self.ego.set_velocity([0, 0])
        
        obs = self._get_dino_obs()
        
        # 瞬移回原位
        self.ego.set_position(old_pos)
        self.ego.set_heading_theta(old_heading)
        self.ego.set_velocity(old_vel)
        return obs
    
    def _setup_scene(self, seed):
        """
        🚀 终极修复方案：严格清理，全局引用，确保周车稳定。
        """
        rng = np.random.RandomState(seed)
        
        # 1. 彻底清理：在 reset 之前，按 ID 逐个清除手动生成的周车
        if hasattr(self.env, "engine") and self.env.engine is not None:
            if self.static_vehicles:
                for sv in self.static_vehicles:
                    try:
                        self.env.engine.clear_objects([sv.id])
                    except:
                        pass
        self.static_vehicles = [] # 重置引用列表

        # 2. 安全重置物理环境
        self.env.reset()
        active_agents = self.env.agent_manager.active_agents
        self.agent_id = list(active_agents.keys())[0]
        self.ego = active_agents[self.agent_id]

        # 3. 严格对齐数据集起步 (8.0~14.0米) [cite: 8, 25-27, 43]
        target_slot = rng.choice(PARKING_SLOTS)
        target_id = target_slot["id"]
        target_x = target_slot["pos"][0]

        spawn_mode = ["right_side", "left_side"][rng.choice(2)]
        if spawn_mode == "left_side":
            start_x = target_x - rng.uniform(8.0, 14.0)
            start_y = rng.uniform(0.0, 2.0)
            start_h = 0.0 + rng.uniform(-0.25, 0.25)
        else:
            start_x = target_x + rng.uniform(8.0, 14.0)
            start_y = rng.uniform(0.0, 2.0)
            start_h = np.pi + rng.uniform(-0.25, 0.25)

        self.ego.set_position([start_x, start_y])
        self.ego.set_heading_theta(start_h)
        self.ego.set_velocity([0, 0])

        # 4. 生成周边车辆 (周车) - 严格对齐数据集逻辑 [cite: 51, 53]
        available_slots = [s for s in PARKING_SLOTS if s["id"] != target_id]
        num_static_cars = rng.randint(1, len(available_slots))
        occupied_indices = rng.choice(len(available_slots), num_static_cars, replace=False)
        occupied_slots = [available_slots[i] for i in occupied_indices]

        for s in occupied_slots:
            noise_x = rng.uniform(-0.2, 0.2)
            noise_y = rng.uniform(-0.2, 0.2)
            noise_h = rng.uniform(-0.1, 0.1)
            spawn_pos = [s["pos"][0] + noise_x, s["pos"][1] + noise_y]
            spawn_h = s["heading"] + noise_h
            
            # 使用全局导入的 SVehicle
            obs_car = self.env.engine.spawn_object(SVehicle, vehicle_config={
                "spawn_position_heading": (spawn_pos, spawn_h),
                "random_color": False
            })
            self.static_vehicles.append(obs_car)

        pilot = ParkingPilot()
        pilot.set_target(target_slot, self.ego)
        return pilot, target_slot

    def sample_random_init_goal_states(self, seed=None):
        pilot, target_slot = self._setup_scene(seed)
        obs_hist_visual, obs_hist_proprio = [], []
        # 收集 3 帧历史，让车动起来
        for i in range(3):
            obs = self._get_dino_obs()
            obs_hist_visual.append(obs['visual'])
            obs_hist_proprio.append(obs['proprio'])
            if i < 2:
                for _ in range(self.frameskip):
                    act = pilot.get_action(self.ego)
                    self.env.step({self.agent_id: act})
        
        t0_pose = np.array([self.ego.position[0], self.ego.position[1], self.ego.heading_theta])
        obs_init = {"visual": np.stack(obs_hist_visual), "proprio": np.stack(obs_hist_proprio)}

        setup_pose = pilot.setup_pose 
        final_pose = np.array([target_slot["pos"][0], target_slot["pos"][1], target_slot["heading"]])
        obs_subgoal = self.get_obs_at_pose(setup_pose)
        obs_goal = self.get_obs_at_pose(final_pose)
        
        return (obs_init, obs_subgoal, obs_goal), (t0_pose, setup_pose, final_pose)

    def rollout(self, seed, init_state, actions):
        """
        🚀 修复：执行物理 rollout 之前，必须先恢复周车和坐标。
        """
        pilot, _ = self._setup_scene(seed) 
        self.ego.set_position(init_state[:2])
        self.ego.set_heading_theta(init_state[2])
        self.ego.set_velocity([0, 0])

        obs_visuals, obs_proprios, obs_poses = [], [], []
        start_obs = self._get_dino_obs()
        obs_visuals.append(start_obs['visual'])
        obs_proprios.append(start_obs['proprio'])
        obs_poses.append(init_state)

        if isinstance(actions, torch.Tensor): actions = actions.cpu().numpy()
        for act in actions:
            obs, _, _, _ = self.step(act)
            obs_visuals.append(obs['visual'])
            obs_proprios.append(obs['proprio'])
            obs_poses.append(np.array([self.ego.position[0], self.ego.position[1], self.ego.heading_theta]))

        return {"visual": np.stack(obs_visuals), "proprio": np.stack(obs_proprios)}, np.stack(obs_poses)

    def eval_state(self, state, goal):
        dist = np.linalg.norm(state[:2] - goal[:2])
        is_success = dist < 1.0 
        return {"success": is_success, "distance": dist}

    def get_obs_at_pose(self, pose_xyh):
        old_pos = self.ego.position
        old_heading = self.ego.heading_theta
        self.ego.set_position(pose_xyh[:2])
        self.ego.set_heading_theta(pose_xyh[2])
        obs = self._get_dino_obs()
        self.ego.set_position(old_pos)
        self.ego.set_heading_theta(old_heading)
        return obs

    def step(self, action):
        if isinstance(action, torch.Tensor): action = action.cpu().detach().numpy().flatten()
        clipped_act = np.clip(action[:2], -1.0, 1.0)
        self.current_action = clipped_act
        try:
            _, r, d, i = self.env.step({self.agent_id: clipped_act})
        except:
            r, d, i = 0, True, {}
        return self._get_dino_obs(), r, d, i

    def _get_dino_obs(self):
        try:
            # 1. 记录真实位置、朝向和【速度】
            real_pos = self.ego.position
            real_heading = self.ego.heading_theta
            real_vel = self.ego.velocity # 👈 必须记录速度！
            
            # 2. 瞬移走
            self.ego.set_position([-1000, -1000])
            bev = self.env.render(
        mode="topdown", window=False, screen_size=RENDER_RES, 
        scaling=BEV_SCALING, camera_position=MAP_CENTER,
        draw_history=False  # <--- 必须加上这一行
    )
            img_raw = bev.get() if hasattr(bev, 'get') else bev
            img = cv2.cvtColor(np.array(img_raw).astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            # 3. 还原位置、朝向和【速度】
            self.ego.set_position(real_pos)
            self.ego.set_heading_theta(real_heading)
            self.ego.set_velocity(real_vel) # 👈 必须还原速度，否则物理动量消失

            # 4. 手动绘图 (维持不变)
            for sv in self.static_vehicles:
                draw_vehicle_on_img(img, sv.position, sv.heading_theta, is_ego=False)
            if self.ego:
                draw_vehicle_on_img(img, self.ego.position, self.ego.heading_theta, is_ego=True)
            img = cv2.resize(img, REAL_RES, interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        except Exception:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
    
        proprio = np.array([self.ego.speed if self.ego else 0, self.current_action[0], 0.0], dtype=np.float32)
        
        # 🔥 新增：把物理坐标带出去，避免调用 get_attr 报错
        raw_pose = np.array([self.ego.position[0], self.ego.position[1], self.ego.heading_theta])
        
        return {"visual": img, "proprio": proprio, "raw_pose": raw_pose}

# ========================================================
# 4. 规划与核心基础组件 (沿用 plan_meta)
# ========================================================
def planning_main_in_dir(working_dir, cfg_dict):
    os.chdir(working_dir)
    return planning_main(cfg_dict=cfg_dict)

def launch_plan_jobs(epoch, cfg_dicts, plan_output_dir):
    with submitit.helpers.clean_env():
        jobs = []
        for cfg_dict in cfg_dicts:
            subdir_name = f"{cfg_dict['planner']['name']}_goal_source={cfg_dict['goal_source']}_goal_H={cfg_dict['goal_H']}_alpha={cfg_dict['objective']['alpha']}"
            subdir_path = os.path.join(plan_output_dir, subdir_name)
            executor = submitit.AutoExecutor(folder=subdir_path, slurm_max_num_timeout=20)
            executor.update_parameters(
                **{k: v for k, v in cfg_dict["hydra"]["launcher"].items() if k != "submitit_folder"})
            cfg_dict["saved_folder"] = subdir_path
            cfg_dict["wandb_logging"] = False
            job = executor.submit(planning_main_in_dir, subdir_path, cfg_dict)
            jobs.append((epoch, subdir_name, job))
            print(f"Submitted evaluation job for checkpoint: {subdir_path}, job id: {job.job_id}")
        return jobs

def build_plan_cfg_dicts(plan_cfg_path="", ckpt_base_path="", model_name="", model_epoch="final", planner=["gd", "cem"],
                         goal_source=["dset"], goal_H=[1, 5, 10], alpha=[0, 0.1, 1]):
    config_path = os.path.dirname(plan_cfg_path)
    overrides = [
        {"planner": p, "goal_source": g_source, "goal_H": g_H, "ckpt_base_path": ckpt_base_path,
         "model_name": model_name, "model_epoch": model_epoch, "objective": {"alpha": a}}
        for p, g_source, g_H, a in product(planner, goal_source, goal_H, alpha)
    ]
    cfg = OmegaConf.load(plan_cfg_path)
    cfg_dicts = []
    for override_args in overrides:
        planner = override_args["planner"]
        planner_cfg = OmegaConf.load(os.path.join(config_path, f"planner/{planner}.yaml"))
        cfg["planner"] = OmegaConf.merge(cfg.get("planner", {}), planner_cfg)
        override_args.pop("planner")
        cfg = OmegaConf.merge(cfg, OmegaConf.create(override_args))
        cfg_dict = OmegaConf.to_container(cfg)
        cfg_dict["planner"]["horizon"] = cfg_dict["goal_H"]
        cfg_dicts.append(cfg_dict)
    return cfg_dicts

class DummyWandbRun:
    def __init__(self): self.mode = "disabled"
    def log(self, *args, **kwargs): pass
    def watch(self, *args, **kwargs): pass
    def config(self, *args, **kwargs): pass
    def finish(self): pass

def load_ckpt(snapshot_path, device):
    with snapshot_path.open("rb") as f:
        payload = torch.load(f, map_location=device, weights_only=False)
    result = {k: v.to(device) for k, v in payload.items() if k in ALL_MODEL_KEYS}
    result["epoch"] = payload["epoch"]
    return result

def load_model(model_ckpt, train_cfg, num_action_repeat, device):
    result = {}
    if model_ckpt.exists():
        result = load_ckpt(model_ckpt, device)
        print(f"Resuming from epoch {result['epoch']}: {model_ckpt}")

    if "encoder" not in result:
        result["encoder"] = hydra.utils.instantiate(train_cfg.encoder)
    if "predictor" not in result:
        raise ValueError("Predictor not found in model checkpoint")

    if train_cfg.has_decoder and "decoder" not in result:
        base_path = os.path.dirname(os.path.abspath(__file__))
        if train_cfg.env.decoder_path is not None:
            decoder_path = os.path.join(base_path, train_cfg.env.decoder_path)
            ckpt = torch.load(decoder_path)
            result["decoder"] = ckpt["decoder"] if isinstance(ckpt, dict) else torch.load(decoder_path)
        else:
            raise ValueError("Decoder path not found")
    elif not train_cfg.has_decoder:
        result["decoder"] = None

    model = hydra.utils.instantiate(
        train_cfg.model,
        encoder=result["encoder"],
        proprio_encoder=result["proprio_encoder"],
        action_encoder=result["action_encoder"],
        predictor=result["predictor"],
        decoder=result["decoder"],
        proprio_dim=train_cfg.proprio_emb_dim,
        action_dim=train_cfg.action_emb_dim,
        concat_dim=train_cfg.concat_dim,
        num_action_repeat=num_action_repeat,
        num_proprio_repeat=train_cfg.num_proprio_repeat,
    )
    model.to(device)
    return model

class PlanWorkspace:
    def __init__(
            self,
            cfg_dict: dict,
            wm: torch.nn.Module,
            dset,
            env: SubprocVectorEnv,
            env_name: str,
            frameskip: int,
            wandb_run: wandb.run,
    ):
        self.cfg_dict = cfg_dict
        self.wm = wm
        self.dset = dset
        self.env = env
        self.env_name = env_name
        self.frameskip = frameskip
        self.wandb_run = wandb_run
        self.device = next(wm.parameters()).device

        self.eval_seed = [cfg_dict["seed"] + n for n in range(cfg_dict["n_evals"])]
        print("eval_seed: ", self.eval_seed)
        self.n_evals = cfg_dict["n_evals"]
        self.goal_source = cfg_dict["goal_source"]
        self.goal_H = cfg_dict["goal_H"]
        self.action_dim = self.dset.action_dim
        self.debug_dset_init = cfg_dict["debug_dset_init"]

        objective_fn = hydra.utils.call(cfg_dict["objective"])

        self.data_preprocessor = Preprocessor(
            action_mean=self.dset.raw_action_mean, action_std=self.dset.raw_action_std,
            state_mean=self.dset.state_mean, state_std=self.dset.state_std,
            proprio_mean=self.dset.proprio_mean, proprio_std=self.dset.proprio_std,
            transform=self.dset.transform,
        )

        if self.cfg_dict["goal_source"] == "file":
            self.prepare_targets_from_file(cfg_dict["goal_file_path"])
        else:
            self.prepare_targets()

        self.evaluator = PlanEvaluator(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            state_0=self.state_0,
            state_g=self.state_g,
            env=self.env,
            wm=self.wm,
            frameskip=self.frameskip,
            seed=self.eval_seed,
            preprocessor=self.data_preprocessor,
            n_plot_samples=self.cfg_dict["n_plot_samples"],
        )

        if self.wandb_run is None or isinstance(self.wandb_run, wandb.sdk.lib.disabled.RunDisabled):
            self.wandb_run = DummyWandbRun()

        self.log_filename = "logs.json"
        self.planner = hydra.utils.instantiate(
            self.cfg_dict["planner"],
            wm=self.wm,
            env=self.env,
            action_dim=self.action_dim,
            objective_fn=objective_fn,
            preprocessor=self.data_preprocessor,
            evaluator=self.evaluator,
            wandb_run=self.wandb_run,
            log_filename=self.log_filename,
        )

        from planning.mpc_park import MPCPlanner
        if isinstance(self.planner, MPCPlanner):
            self.planner.sub_planner.horizon = cfg_dict["goal_H"]
            self.planner.n_taken_actions = cfg_dict["planner"]["n_taken_actions"]
        else:
            self.planner.horizon = cfg_dict["goal_H"]

        self.dump_targets()

    def prepare_targets(self):
        if self.goal_source == "random_state":
            obs_results, state_results = self.env.sample_random_init_goal_states(self.eval_seed)

            init_obs_list = [res[0] for res in obs_results]
            sub_obs_list  = [res[1] for res in obs_results]
            goal_obs_list = [res[2] for res in obs_results]
            
            # 提取物理坐标 (X, Y, H)
            state_0_list    = [res[0] for res in state_results]
            state_sub_list  = [res[1] for res in state_results]
            state_goal_list = [res[2] for res in state_results]

            def stack_dicts(dict_list):
                keys = dict_list[0].keys()
                return {k: np.stack([d[k] for d in dict_list]) for k in keys}

            self.obs_0 = stack_dicts(init_obs_list)
            self.obs_g_sub = stack_dicts(sub_obs_list)
            self.obs_g_final = stack_dicts(goal_obs_list)

            # 🔥 统一使用 _pose 后缀，防止混淆
            self.state_0 = np.stack(state_0_list)
            self.state_g_sub_pose = np.stack(state_sub_list)
            self.state_g_final_pose = np.stack(state_goal_list)

            for k in self.obs_g_sub.keys():
                self.obs_g_sub[k] = np.expand_dims(self.obs_g_sub[k], axis=1)
                self.obs_g_final[k] = np.expand_dims(self.obs_g_final[k], axis=1)

            # 🔥 修正报错行：将默认目标设为最终位姿
            self.state_g = self.state_g_final_pose 
            self.obs_g = self.obs_g_final
            self.gt_actions = None

        else:
            # Dataset 分支同样适配 3 帧历史逻辑
            num_hist = 3
            total_len = self.frameskip * (num_hist - 1 + self.goal_H) + 1
            observations, states, actions, env_info = self.sample_traj_segment_from_dset(traj_len=total_len)
            self.env.update_env(env_info)

            # 确定 t=0 (当前时刻) 在序列中的索引
            t0_idx = (num_hist - 1) * self.frameskip

            # 真车的初始物理状态是历史的最后一帧
            init_state = [x[t0_idx] for x in states]
            init_state = np.array(init_state)
            
            actions = torch.stack(actions)
            test_actions = actions[:, t0_idx : t0_idx + self.frameskip * self.goal_H]
            if self.goal_source == "random_action":
                test_actions = torch.randn_like(test_actions)

            wm_actions = test_actions[:, ::self.frameskip, :]
            exec_actions = self.data_preprocessor.denormalize_actions(test_actions)
            rollout_obses, rollout_states = self.env.rollout(self.eval_seed, init_state, exec_actions.numpy())

            # 提取完美的 3 帧历史喂给模型
            obs_0_dict = {}
            for k in rollout_obses.keys():
                batch_obs = []
                for b in range(self.n_evals):
                    hist_frames = []
                    for i in range(num_hist):
                        idx = i * self.frameskip
                        hist_frames.append(observations[b][k][idx])
                    batch_obs.append(np.stack(hist_frames))
                obs_0_dict[k] = np.stack(batch_obs)
                
            self.obs_0 = obs_0_dict
            self.obs_g = {key: np.expand_dims(arr[:, -1], axis=1) for key, arr in rollout_obses.items()}
            self.state_0 = init_state
            self.state_g = rollout_states[:, -1]
            self.gt_actions = wm_actions
            
            # 为了让 Dataset 分支在调用 perform_planning 时不报错，把 subgoal 直接等同于 final_goal
            self.obs_g_sub = self.obs_g
            self.obs_g_final = self.obs_g
            self.state_g_sub = self.state_g
            self.state_g_final = self.state_g

    def sample_traj_segment_from_dset(self, traj_len):
        states = []
        actions = []
        observations = []
        env_info = []
        
        valid_traj = [
            self.dset[i][0]["visual"].shape[0]
            for i in range(len(self.dset))
            if self.dset[i][0]["visual"].shape[0] >= traj_len
        ]
        if len(valid_traj) == 0:
            raise ValueError("No trajectory in the dataset is long enough.")
            
        for i in range(self.n_evals):
            max_offset = -1
            while max_offset < 0:
                traj_id = random.randint(0, len(self.dset) - 1)
                
                # 兼容返回 3 个或 4 个值的数据集
                sample_data = self.dset[traj_id]
                if len(sample_data) == 4:
                    obs, act, state, e_info = sample_data
                else:
                    obs, act, state = sample_data
                    e_info = {}
                    
                max_offset = obs["visual"].shape[0] - traj_len
            state = state.numpy()
            offset = random.randint(0, max_offset)
            obs = {key: arr[offset: offset + traj_len] for key, arr in obs.items()}
            state = state[offset: offset + traj_len]
            act = act[offset: offset + self.frameskip * self.goal_H]
            actions.append(act)
            states.append(state)
            observations.append(obs)
            env_info.append(e_info)
        return observations, states, actions, env_info

    def prepare_targets_from_file(self, file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        self.obs_0 = data["obs_0"]
        self.obs_g = data["obs_g"]
        self.state_0 = data["state_0"]
        self.state_g = data["state_g"]
        self.gt_actions = data["gt_actions"]
        self.goal_H = data["goal_H"]

    def dump_targets(self):
        with open("plan_targets.pkl", "wb") as f:
            pickle.dump({
                "obs_0": self.obs_0, "obs_g": self.obs_g, "state_0": self.state_0,
                "state_g": self.state_g, "gt_actions": self.gt_actions, "goal_H": self.goal_H,
            }, f)
        print(f"Dumped plan targets to {os.path.abspath('plan_targets.pkl')}")

# 修改 plan_park_3_full.py 中的 perform_planning
    def perform_planning(self):
        # 1. 获取初始状态 (Numpy 字典)
        cur_obs_0, _ = self.evaluator.get_init_cond()

        # 2. 定义分层阶段目标
        stages = [
            {
                "name": "阶段 1: 前进至面包屑 (刹停准备)", 
                "goal_obs": self.obs_g_sub,       # 👈 锁定面包屑图像
                "goal_pose": self.state_g_sub_pose, 
                "dist_tol": 1.5                   # 接近 1.5 米后进入倒车阶段
            },
            {
                "name": "阶段 2: 倒车入库", 
                "goal_obs": self.obs_g_final,     # 👈 锁定停车位图像
                "goal_pose": self.state_g_final_pose, 
                "dist_tol": 0.35
            }
        ]

        for st in stages:
            print(f"\n🚀 开始 {st['name']}")
            target_obs_g = st['goal_obs']
            cur_pos = self.state_0[0, :2] 

            for mpc_step in range(60): 
                dist = np.linalg.norm(cur_pos - st['goal_pose'][0, :2])
                if mpc_step % 10 == 0:
                    print(f"  [MPC] 距离目标: {dist:.2f}m")

                if dist < st['dist_tol']:
                    print(f"✅ 到达阶段目标")
                    break

                # 规划并执行
                actions, _ = self.planner.plan(obs_0=cur_obs_0, obs_g=target_obs_g)
                exec_act = self.data_preprocessor.denormalize_actions(actions[:, 0, :]).cpu().numpy()
                
                # 调用上面修复过的 step (带 frameskip)
                next_obs_batch, _, _, _ = self.env.step(exec_act)
                
                # 更新坐标和历史画面，确保 WM 知道车在动
                cur_pos = next_obs_batch['raw_pose'][0, :2] 
                for k in ['visual', 'proprio']:
                    new_frame = np.expand_dims(next_obs_batch[k], axis=1)
                    cur_obs_0[k] = np.concatenate([cur_obs_0[k][:, 1:], new_frame], axis=1)

        print("\n🏆 全程规划完成")
        return {}
        
def planning_main(cfg_dict):
    output_dir = cfg_dict["saved_folder"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if cfg_dict["wandb_logging"]:
        wandb_run = wandb.init(project=f"plan_{cfg_dict['planner']['name']}", config=cfg_dict)
        wandb.run.name = "{}".format(output_dir.split("plan_outputs/")[-1])
    else:
        wandb_run = None

    ckpt_base_path = cfg_dict["ckpt_base_path"]
    model_path = f"{ckpt_base_path}/outputs/{cfg_dict['model_name']}/"
    with open(os.path.join(model_path, "hydra.yaml"), "r") as f:
        model_cfg = OmegaConf.load(f)

    seed(cfg_dict["seed"])
    
    # 加载数据集
    datasets, traj_dsets = hydra.utils.call(
        model_cfg.env.dataset, num_hist=model_cfg.num_hist, num_pred=model_cfg.num_pred, frameskip=model_cfg.frameskip
    )
    dset = traj_dsets["valid"]

    # 🔥🔥🔥 核心修复：把真实数据集的统计量，覆盖给 Trajectory 数据集 🔥🔥🔥
    real_dset = datasets["train"] # 获取带有真实 mean/std 的数据集
    dset.action_mean = real_dset.action_mean
    dset.action_std = real_dset.action_std
    dset.state_mean = real_dset.state_mean
    dset.state_std = real_dset.state_std
    dset.proprio_mean = real_dset.proprio_mean
    dset.proprio_std = real_dset.proprio_std

    num_action_repeat = model_cfg.num_action_repeat
    model_ckpt = Path(model_path) / "checkpoints" / f"model_{cfg_dict['model_epoch']}.pth"
    model = load_model(model_ckpt, model_cfg, num_action_repeat, device=device)

    # 👇 ================= 新增：防止模型动作幻觉的 WM Wrapper ================= 👇
    # 1. 计算物理边界 [-1, 1] 在当前模型归一化空间中的合法范围
    act_mean_10d = dset.action_mean.to(device)
    act_std_10d = dset.action_std.to(device)
    norm_min = (torch.tensor([-1.0] * 10, device=device) - act_mean_10d) / act_std_10d
    norm_max = (torch.tensor([1.0] * 10, device=device) - act_mean_10d) / act_std_10d

    # 2. 编写一个极其轻量级的拦截器，伪装成原本的 model
    # 2. 编写一个极其轻量级的拦截器，伪装成原本的 model
    class ClampedWM:
        def __init__(self, original_wm, n_min, n_max):
            self.original_wm = original_wm
            self.n_min = n_min
            self.n_max = n_max
            
        def rollout(self, obs_0, act, **kwargs):
            # 将 Planner 的动作限制在物理范围内
            clipped_act = torch.max(torch.min(act, self.n_max), self.n_min)
            num_hist = obs_0['visual'].shape[1]
            
            B, _, D = clipped_act.shape
            
            # 🔥 核心修复 1：历史只需补 (num_hist - 1) 个假动作
            # 这样 clipped_act 的第 1 个动作，就会刚好落入 act_0 的位置！消灭延迟！
            dummy_hist_act = torch.zeros(B, num_hist - 1, D, device=clipped_act.device)
            full_act = torch.cat([dummy_hist_act, clipped_act], dim=1)
            
            # 丢给底层的世界模型去想象
            z_obses, z = self.original_wm.rollout(obs_0, full_act, **kwargs)
            
            sliced_z_obses = {}
            for k, v in z_obses.items():
                # 🔥 核心修复 2：严格切片，死死卡住目标长度，彻底消灭 6 and 7 的报错！
                # 目标长度 = 初始 1 帧 + Planner 预测的未来 N 帧
                target_len = clipped_act.shape[1] + 1
                sliced_z_obses[k] = v[:, num_hist - 1 : num_hist - 1 + target_len]
                
            return sliced_z_obses, z
            
        def __getattr__(self, name):
            # 其他所有方法原封不动转交还原模型
            return getattr(self.original_wm, name)

    # 3. 给原模型穿上紧身衣
    model = ClampedWM(model, norm_min, norm_max)
    # 👆 ===================================================================== 👆

    frameskip = model_cfg.frameskip

    # 提取 action_mean 和 action_std
    act_mean = dset.raw_action_mean.numpy()
    act_std = dset.raw_action_std.numpy()

    def create_wrapped_env():
        env_config = {
            "use_render": False,
            "num_agents": 1,
            "start_seed": 400,
            "allow_respawn": False,  
            "window_size": RENDER_RES,
            "out_of_road_done": False,
            "crash_vehicle_done": False,
            "vehicle_config": {"lidar": {"num_lasers": 0}, "show_navi_mark": False},
        }
        env = MultiAgentParkingLotEnv(env_config)
        # 🔥 把 mean 和 std 传给 Wrapper
        return ParkingDinoWrapper(env, frameskip=frameskip, action_mean=act_mean, action_std=act_std)

    env = SubprocVectorEnv([create_wrapped_env for _ in range(cfg_dict["n_evals"])])

    plan_workspace = PlanWorkspace(
        cfg_dict=cfg_dict, wm=model, dset=dset, env=env,
        env_name=model_cfg.env.name, frameskip=model_cfg.frameskip,
        wandb_run=wandb_run,
    )
    logs = plan_workspace.perform_planning()
    return logs

@hydra.main(config_path="conf", config_name="plan")
def main(cfg: OmegaConf):
    with open_dict(cfg):
        cfg["saved_folder"] = os.getcwd()
        log.info(f"Planning result saved dir: {cfg['saved_folder']}")
    cfg_dict = cfg_to_dict(cfg)
    cfg_dict["wandb_logging"] = True
    planning_main(cfg_dict)

if __name__ == "__main__":
    main()