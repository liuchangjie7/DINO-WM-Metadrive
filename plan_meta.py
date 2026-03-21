import os
import gym
import json
import hydra
import random
import torch
import metadrive
import pickle
import wandb
import logging
import warnings
import numpy as np
import submitit
from itertools import product
from pathlib import Path
from einops import rearrange
from omegaconf import OmegaConf, open_dict

import cv2

from env.venv import SubprocVectorEnv
from custom_resolvers import replace_slash
from preprocessor import Preprocessor
from planning.evaluator import PlanEvaluator
from utils import cfg_to_dict, seed

from gym.envs.registration import register
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.policy.idm_policy import IDMPolicy

try:
    register(
        id="metadrive",
        entry_point="metadrive.envs:MetaDriveEnv",
        kwargs={"config": {"use_render": True}},
    )
except gym.error.Error:
    pass


class MetaDriveDinoWrapper(gym.Wrapper):
    def __init__(self, env, frameskip=1):  # 接收 frameskip
        super().__init__(env)
        self.proprio_dim = 3
        self.current_seed = None
        self.steps_to_goal = 25
        self.start_offset_range = [50, 300]
        self.frameskip = frameskip  # 保存 frameskip

    def update_env(self, env_info):
        if env_info:
            if 'seed' in env_info:
                self.current_seed = int(env_info['seed'])
            if 'steps_to_goal' in env_info:
                self.steps_to_goal = int(env_info['steps_to_goal'])
        return "OK"

    def sample_random_init_goal_states(self, seed=None):
        start_seed_limit = self.env.config.get("start_seed", 0)

        if seed is None:
            seed = self.current_seed if self.current_seed is not None else start_seed_limit

        if seed < start_seed_limit:
            seed += start_seed_limit

        rng = np.random.RandomState(seed)
        start_offset = rng.randint(self.start_offset_range[0], self.start_offset_range[1])
        action_noise_seed = seed * 12345

        # Round 1: 采集
        try:
            self.env.reset(seed=seed)
        except TypeError:
            self.env.seed(seed)
            self.env.reset()

        policy = IDMPolicy(control_object=self.env.agent, random_seed=seed)
        noise_rng = np.random.RandomState(action_noise_seed)

        # 跑到起点
        for _ in range(start_offset):
            self.env.step(self._get_idm_action(policy, noise_rng))
        obs_init = self._get_dino_obs(None)

        # 跑到终点
        for _ in range(self.steps_to_goal):
            self.env.step(self._get_idm_action(policy, noise_rng))
        obs_goal = self._get_dino_obs(None)

        # Round 2: 复位
        try:
            self.env.reset(seed=seed)
        except TypeError:
            self.env.seed(seed)
            self.env.reset()

        policy = IDMPolicy(control_object=self.env.agent, random_seed=seed)
        noise_rng = np.random.RandomState(action_noise_seed)

        # 再次跑到起点
        for _ in range(start_offset):
            self.env.step(self._get_idm_action(policy, noise_rng))

        if isinstance(obs_init, dict):
            obs_init["debug_info"] = np.array([start_offset, seed])

        return obs_init, obs_goal

    # 【核心方法】实现 PlanEvaluator 需要的 rollout 接口
    def rollout(self, seed, init_state, actions):
        """
        执行给定的动作序列。为了保证与 Planning 时的起点一致，
        这里必须使用与 sample_random_init_goal_states 相同的逻辑，
        先 IDM 跑到 start_offset，再执行 actions。
        """
        start_seed_limit = self.env.config.get("start_seed", 0)
        if seed is None:
            seed = self.current_seed if self.current_seed is not None else start_seed_limit
        if seed < start_seed_limit:
            seed += start_seed_limit

        rng = np.random.RandomState(seed)
        start_offset = rng.randint(self.start_offset_range[0], self.start_offset_range[1])
        action_noise_seed = seed * 12345

        # Reset 并回放到起点
        try:
            self.env.reset(seed=seed)
        except TypeError:
            self.env.seed(seed)
            self.env.reset()

        policy = IDMPolicy(control_object=self.env.agent, random_seed=seed)
        noise_rng = np.random.RandomState(action_noise_seed)

        for _ in range(start_offset):
            self.env.step(self._get_idm_action(policy, noise_rng))

        # 执行规划出的动作
        obs_visuals = []
        obs_proprios = []

        # 【核心修复：添加初始帧 (t=0)】
        # -----------------------------------------------------------
        obs_start = self._get_dino_obs(None)
        obs_visuals.append(obs_start['visual'])
        obs_proprios.append(obs_start['proprio'])
        # -----------------------------------------------------------

        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()

        for act in actions:
            obs, reward, done, info = self.step(act)
            obs_visuals.append(obs['visual'])
            obs_proprios.append(obs['proprio'])

        visual_stack = np.stack(obs_visuals)
        proprio_stack = np.stack(obs_proprios)

        result_obs = {
            "visual": visual_stack,
            "proprio": proprio_stack
        }
        return result_obs, proprio_stack

    # 【新增核心方法】实现 eval_state 接口，解决 AttributeError
    def eval_state(self, state, goal):
        """
        计算状态评估指标。
        """
        # 计算简单的 L2 距离 (仅基于速度/转向)
        diff = state - goal
        dist = np.linalg.norm(diff)

        # 【关键修改】必须返回 bool 类型 (False) 或 int (0)，不能是 float (0.0)
        # 否则 mpc.py 中的位运算 (successes & ~self.is_success) 会报错
        is_success = False

        # 如果你有办法计算 XY 距离，可以在这里加逻辑，比如:
        # if dist < 0.1: is_success = True

        return {"success": is_success, "distance": dist}

    def _get_idm_action(self, policy, noise_rng):
        action = policy.act(agent_id="default_agent")
        if not isinstance(action, np.ndarray): action = np.array(action)
        action += noise_rng.normal(0, 0.02, size=action.shape)
        action = np.clip(action, -1.0, 1.0)
        return action

    def step(self, action):
        total_reward = 0.0
        done = False
        truncate = False
        info = {}

        # Action Repeat: 重复执行 frameskip 次
        for _ in range(self.frameskip):
            next_obs, reward, d, t, i = self.env.step(action)
            total_reward += reward
            done = d or done
            truncate = t or truncate
            info = i
            if done or truncate:
                break

        dino_obs = self._get_dino_obs(next_obs)
        return dino_obs, total_reward, done, info

    def _get_dino_obs(self, raw_obs):
        img = None
        try:
            img = self.env.render(
                mode="top_down",
                window=False,
                screen_size=(224, 224),
                scaling=4.0,
                camera_position=None,
                draw_target_vehicle_trajectory=False,
                draw_history=False
            )
        except Exception as e:
            print(f"Render Error: {e}")
            # pass

        if hasattr(img, 'get'):
            img = img.get()
        elif hasattr(img, 'cpu'):
            img = img.cpu().numpy()

        if img is None or not isinstance(img, np.ndarray):
            img = np.zeros((224, 224, 3), dtype=np.uint8)

        if img.shape[0] != 224 or img.shape[1] != 224:
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)

        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))

        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

        try:
            agent = self.env.agent
            proprio = np.array([agent.speed, agent.steering, 0.0], dtype=np.float32)
        except:
            proprio = np.zeros(self.proprio_dim, dtype=np.float32)

        return {"visual": img, "proprio": proprio}


warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

ALL_MODEL_KEYS = [
    "encoder",
    "predictor",
    "decoder",
    "proprio_encoder",
    "action_encoder",
]


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

        self.eval_seed = [cfg_dict["seed"] * n + 1 for n in range(cfg_dict["n_evals"])]
        print("eval_seed: ", self.eval_seed)
        self.n_evals = cfg_dict["n_evals"]
        self.goal_source = cfg_dict["goal_source"]
        self.goal_H = cfg_dict["goal_H"]

        # 1. 动作维度设为 2 (适配模型权重)
        self.action_dim = self.dset.action_dim

        self.debug_dset_init = cfg_dict["debug_dset_init"]

        objective_fn = hydra.utils.call(cfg_dict["objective"])

        self.data_preprocessor = Preprocessor(
            action_mean=self.dset.action_mean, action_std=self.dset.action_std,
            state_mean=self.dset.state_mean, state_std=self.dset.state_std,
            proprio_mean=self.dset.proprio_mean, proprio_std=self.dset.proprio_std,
            transform=self.dset.transform,
        )

        if self.cfg_dict["goal_source"] == "file":
            self.prepare_targets_from_file(cfg_dict["goal_file_path"])
        else:
            self.prepare_targets()

        # 2. 欺骗 Evaluator，传入 frameskip=1
        # 防止它进行动作拆分，真实的帧跳跃由 Wrapper 的 step 方法处理
        self.evaluator = PlanEvaluator(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            state_0=self.state_0,
            state_g=self.state_g,
            env=self.env,
            wm=self.wm,
            frameskip=1,
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

        from planning.mpc import MPCPlanner
        if isinstance(self.planner, MPCPlanner):
            self.planner.sub_planner.horizon = cfg_dict["goal_H"]
            self.planner.n_taken_actions = cfg_dict["planner"]["n_taken_actions"]
        else:
            self.planner.horizon = cfg_dict["goal_H"]

        self.dump_targets()

    def prepare_targets(self):
        states = []
        actions = []
        observations = []

        if self.goal_source == "random_state":
            target_phys_steps = self.frameskip * self.goal_H
            print(f"🚀 Planning Setup: IDM running {target_phys_steps} steps to generate goal.")

            observations, states, actions, env_info = self.sample_traj_segment_from_dset(traj_len=2)

            steps_update_list = [{'steps_to_goal': target_phys_steps} for _ in range(self.n_evals)]
            self.env.update_env(steps_update_list)

            init_obs_tuple, goal_obs_tuple = self.env.sample_random_init_goal_states(self.eval_seed)

            for i, obs in enumerate(init_obs_tuple):
                if "debug_info" in obs:
                    del obs["debug_info"]

            def stack_dicts(dict_iterable):
                dict_list = list(dict_iterable)
                keys = dict_list[0].keys()
                return {k: np.stack([d[k] for d in dict_list]) for k in keys}

            obs_0 = stack_dicts(init_obs_tuple)
            obs_g = stack_dicts(goal_obs_tuple)

            state_0 = obs_0['proprio']
            state_g = obs_g['proprio']

            for k in obs_0.keys():
                obs_0[k] = np.expand_dims(obs_0[k], axis=1)
                obs_g[k] = np.expand_dims(obs_g[k], axis=1)

            self.obs_0 = obs_0
            self.obs_g = obs_g
            self.state_0 = state_0
            self.state_g = state_g
            self.gt_actions = None

        else:
            observations, states, actions, env_info = self.sample_traj_segment_from_dset(
                traj_len=self.frameskip * self.goal_H + 1)
            self.env.update_env(env_info)

            init_state = [x[0] for x in states]
            init_state = np.array(init_state)
            actions = torch.stack(actions)
            if self.goal_source == "random_action":
                actions = torch.randn_like(actions)

            # 3. 数据集模式下也不堆叠动作，而是采样
            wm_actions = actions[:, ::self.frameskip, :]

            exec_actions = self.data_preprocessor.denormalize_actions(actions)
            rollout_obses, rollout_states = self.env.rollout(self.eval_seed, init_state, exec_actions.numpy())

            self.obs_0 = {key: np.expand_dims(arr[:, 0], axis=1) for key, arr in rollout_obses.items()}
            self.obs_g = {key: np.expand_dims(arr[:, -1], axis=1) for key, arr in rollout_obses.items()}
            self.state_0 = init_state
            self.state_g = rollout_states[:, -1]
            self.gt_actions = wm_actions

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
                obs, act, state, e_info = self.dset[traj_id]
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

    def perform_planning(self):
        if self.debug_dset_init:
            actions_init = self.gt_actions
        else:
            actions_init = None
        actions, action_len = self.planner.plan(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            actions=actions_init,
        )
        logs, successes, _, _ = self.evaluator.eval_actions(
            actions.detach(), action_len, save_video=True, filename="output_final"
        )
        logs = {f"final_eval/{k}": v for k, v in logs.items()}
        self.wandb_run.log(logs)
        logs_entry = {key: (value.item() if isinstance(value, (np.float32, np.int32, np.int64)) else value) for
                      key, value in logs.items()}
        with open(self.log_filename, "a") as file:
            file.write(json.dumps(logs_entry) + "\n")
        return logs


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


class DummyWandbRun:
    def __init__(self): self.mode = "disabled"

    def log(self, *args, **kwargs): pass

    def watch(self, *args, **kwargs): pass

    def config(self, *args, **kwargs): pass

    def finish(self): pass


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
    _, dset = hydra.utils.call(
        model_cfg.env.dataset, num_hist=model_cfg.num_hist, num_pred=model_cfg.num_pred, frameskip=model_cfg.frameskip
    )
    dset = dset["valid"]

    num_action_repeat = model_cfg.num_action_repeat
    model_ckpt = Path(model_path) / "checkpoints" / f"model_{cfg_dict['model_epoch']}.pth"
    model = load_model(model_ckpt, model_cfg, num_action_repeat, device=device)

    frameskip = model_cfg.frameskip

    def create_wrapped_env():
        kwargs = model_cfg.env.get("kwargs", {})
        if kwargs is None: kwargs = {}
        if "config" not in kwargs: kwargs["config"] = {}
        overrides = {
            "use_render": False, "map": "CCCCC", "traffic_density": 0.0,
            "image_on_cuda": False, "window_size": (224, 224),
            "vehicle_config": {"image_source": "main_camera"},
            "start_seed": 100, "num_scenarios": 1000000,
        }
        kwargs["config"].update(overrides)
        env = MetaDriveEnv(kwargs["config"])
        # 传递 frameskip 给 Wrapper
        return MetaDriveDinoWrapper(env, frameskip=frameskip)

    if model_cfg.env.name == "wall" or model_cfg.env.name == "deformable_env":
        from env.serial_vector_env import SerialVectorEnv
        env = SerialVectorEnv([create_wrapped_env for _ in range(cfg_dict["n_evals"])])
    else:
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