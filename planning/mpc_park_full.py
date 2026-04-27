import torch
import hydra
import copy
import numpy as np
from einops import rearrange, repeat
from utils import slice_trajdict_with_t
from .base_planner import BasePlanner


class MPCPlanner(BasePlanner):
    """
    an online planner so feedback from env is allowed
    """

    def __init__(
        self,
        max_iter,
        n_taken_actions,
        sub_planner,
        wm,
        env,  # for online exec
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        logging_prefix="mpc",
        log_filename="logs.json",
        **kwargs,
    ):
        super().__init__(
            wm,
            action_dim,
            objective_fn,
            preprocessor,
            evaluator,
            wandb_run,
            log_filename,
        )
        self.env = env
        self.max_iter = np.inf if max_iter is None else max_iter
        self.n_taken_actions = n_taken_actions
        self.logging_prefix = logging_prefix
        sub_planner["_target_"] = sub_planner["target"]
        self.sub_planner = hydra.utils.instantiate(
            sub_planner,
            wm=self.wm,
            action_dim=self.action_dim,
            objective_fn=self.objective_fn,
            preprocessor=self.preprocessor,
            evaluator=self.evaluator,  # evaluator is shared for mpc and sub_planner
            wandb_run=self.wandb_run,
            log_filename=None,
        )
        self.is_success = None
        self.action_len = None  # keep track of the step each traj reaches success
        self.iter = 0
        self.planned_actions = []

    def _apply_success_mask(self, actions):
        device = actions.device
        mask = torch.tensor(self.is_success).bool()
        actions[mask] = 0
        masked_actions = rearrange(
            actions[mask], "... (f d) -> ... f d", f=self.evaluator.frameskip
        )
        masked_actions = self.preprocessor.normalize_actions(masked_actions.cpu())
        masked_actions = rearrange(masked_actions, "... f d -> ... (f d)")
        actions[mask] = masked_actions.to(device)
        return actions

    # 将 plan 函数的定义改为如下：
    def plan(self, obs_0, obs_g, actions=None, stage_name="stage"):
        # 🚀 每次调用 plan (Stage 切换) 时，彻底重置迭代状态
        self.iter = 0
        self.planned_actions = []
        n_evals = obs_0["visual"].shape[0]
        self.is_success = np.zeros(n_evals, dtype=bool)
        self.action_len = np.full(n_evals, np.inf)
        
        init_obs_0, init_state_0 = self.evaluator.get_init_cond()
        cur_obs_0 = obs_0 # 这里的 obs_0 必须是初始的 3 帧
        memo_actions = None
        self.initial_horizon = self.sub_planner.horizon

        while not np.all(self.is_success) and self.iter < self.max_iter:
            self.sub_planner.logging_prefix = f"plan_{stage_name}_{self.iter}"
            
            # 动态缩减视距 (防止芝诺悖论)
            min_horizon = 2
            current_horizon = self.initial_horizon - self.iter
            self.sub_planner.horizon = max(min_horizon, current_horizon)

            # 1. 记录历史轨迹，供 Evaluator 绘图
            if len(self.planned_actions) > 0:
                self.evaluator.history_actions = torch.cat(self.planned_actions, dim=1)
            
            # 2. 规划 1 步动作
            actions, _ = self.sub_planner.plan(obs_0=cur_obs_0, obs_g=obs_g, actions=memo_actions)
            actions = torch.clamp(actions, -2.5, 2.5)
            taken_actions = actions.detach()[:, : self.n_taken_actions]
            self.planned_actions.append(taken_actions)

            # 3. 执行评估并拿到真实环境反馈
            action_so_far = torch.cat(self.planned_actions, dim=1)
            self.evaluator.assign_init_cond(obs_0=init_obs_0, state_0=init_state_0)
            logs, successes, e_obses, e_states = self.evaluator.eval_actions(
                action_so_far, self.action_len, filename=f"plan_{stage_name}_{self.iter}", save_video=True
            )

            # 4. 🔥 核心：滑动窗口更新 obs_0 (保持 3 帧历史) 🔥
            # e_obses 包含了从 init_obs_0 开始的所有历史图像
            # 我们截取最后 3 帧作为下一轮规划的起点
            cur_obs_0 = {k: v[:, -3:] for k, v in e_obses.items()}
            
            e_final_state = e_states[:, -1]
            self.evaluator.assign_init_cond(obs_0=cur_obs_0, state_0=e_final_state)
            
            self.iter += 1
            print(f"🚀 [{stage_name}] Iter {self.iter} | Dist: {logs.get('final_eval/distance', 999):.2f}")

        return torch.cat(self.planned_actions, dim=1), self.action_len