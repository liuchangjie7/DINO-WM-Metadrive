import math
import torch
import numpy as np
from einops import rearrange, repeat
from .base_planner import BasePlanner
from utils import move_to_device

class MPPIPlanner(BasePlanner):
    def __init__(
        self,
        horizon,
        num_samples,
        opt_steps,
        eval_every,
        temperature,        
        noise_sigma,        
        num_control_points, # 🚀 新增：贝塞尔曲线的控制点数量 (决定降维打击的强度)
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        logging_prefix="plan_0",
        log_filename="logs.json",
        **kwargs,
    ):
        super().__init__(
            wm, action_dim, objective_fn, preprocessor, evaluator, wandb_run, log_filename,
        )
        self.horizon = horizon
        self.num_samples = num_samples
        self.opt_steps = opt_steps
        self.eval_every = eval_every
        self.temperature = temperature
        self.noise_sigma = noise_sigma
        self.num_control_points = num_control_points # 默认建议设为 4 (三次贝塞尔曲线)
        self.logging_prefix = logging_prefix

    def init_mu(self, obs_0, actions=None):
        n_evals = obs_0["visual"].shape[0]
        if actions is None:
            mu = torch.zeros(n_evals, 0, self.action_dim)
        else:
            mu = actions
            
        device = mu.device
        t = mu.shape[1]
        remaining_t = self.horizon - t

        if remaining_t > 0:
            new_mu = torch.zeros(n_evals, remaining_t, self.action_dim)
            mu = torch.cat([mu, new_mu.to(device)], dim=1)
        return mu

    def _get_bezier_matrix(self, H, K, device):
        """
        🚀 创新一：预计算贝塞尔曲线基底矩阵 (Bezier Basis Matrix)
        用于将 K 个控制点瞬间映射展开到 H 个离散时间步
        """
        t = torch.linspace(0, 1, H, device=device)
        M = torch.zeros(H, K, device=device)
        n = K - 1
        for i in range(K):
            coef = math.comb(n, i)
            M[:, i] = coef * (t ** i) * ((1 - t) ** (n - i))
        return M

    def plan(self, obs_0, obs_g, actions=None):
        trans_obs_0 = move_to_device(self.preprocessor.transform_obs(obs_0), self.device)
        trans_obs_g = move_to_device(self.preprocessor.transform_obs(obs_g), self.device)
        z_obs_g = self.wm.encode_obs(trans_obs_g)

        mu = self.init_mu(obs_0, actions).to(self.device)
        n_evals = mu.shape[0]

        for i in range(self.opt_steps):
            losses = []
            
            for traj in range(n_evals):
                cur_trans_obs_0 = {
                    key: repeat(arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples)
                    for key, arr in trans_obs_0.items()
                }
                cur_z_obs_g = {
                    key: repeat(arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples)
                    for key, arr in z_obs_g.items()
                }
                
                # ==============================================================
                # 🚀 创新一：Spline-MPPI 动作空间降维打击
                # ==============================================================
                # 如果时域足够长，我们不再傻乎乎地采样 H 帧，而是只采样 K 个控制点
                if self.horizon >= self.num_control_points:
                    # 1. 采样 K 个控制点的微小扰动 (num_samples, K, action_dim)
                    cp_noise = torch.randn(self.num_samples, self.num_control_points, self.action_dim, device=self.device) * self.noise_sigma
                    # 2. 获取贝塞尔映射矩阵 M (H, K)
                    M = self._get_bezier_matrix(self.horizon, self.num_control_points, self.device)
                    # 3. 矩阵乘法：瞬间将 K 个控制点的扰动平滑展开成 H 帧的连续物理动作曲线！
                    # (Samples x K x D) @ (H x K)^T -> (Samples x H x D)
                    noise = torch.einsum('skd, hk -> shd', cp_noise, M)
                else:
                    # 如果时域极短(比如快到终点了 H=2)，直接降级为普通高斯噪声
                    noise = torch.randn(self.num_samples, self.horizon, self.action_dim, device=self.device) * self.noise_sigma
                
                # 保底策略：第 0 个样本不加噪声
                noise[0] = 0.0  
                
                # 将平滑的贝塞尔扰动加到基础均值动作上
                action = mu[traj] + noise
                
                # 世界模型推演
                with torch.no_grad():
                    i_z_obses, i_zs = self.wm.rollout(obs_0=cur_trans_obs_0, act=action)

                # 计算视觉/特征 Loss
                loss = self.objective_fn(i_z_obses, cur_z_obs_g).detach()

                # ==============================================================
                # 🚀 创新二：人类驾驶先验 (Comfort-aware Cost)
                # ==============================================================
                if self.horizon > 1:
                    action_diff = action[:, 1:, :] - action[:, :-1, :]
                    # 惩罚方向盘 (dim 0) 和 油门 (dim 1) 的抖动
                    steer_penalty = torch.sum(action_diff[:, :, 0] ** 2, dim=1)
                    throttle_penalty = torch.sum(action_diff[:, :, 1] ** 2, dim=1)
                    # 虽然贝塞尔曲线已经很平滑，但我们依然惩罚那些“画大圆弧”的夸张动作
                    smoothness_cost = 0.5 * steer_penalty + 0.2 * throttle_penalty
                    loss = loss + smoothness_cost

                # ==============================================================
                # Terminal braking cost: penalize non-zero actions at the end
                # to force the vehicle to slow down / stop near the goal.
                # NOTE: scaled down aggressively for short-horizon MPC loops
                # so the car does not crawl forward timidly.
                # ==============================================================
                terminal_penalty = 0.2 * torch.sum(action[:, -1, :] ** 2, dim=1)
                loss = loss + terminal_penalty

                # 信息论 Softmax 加权
                beta = torch.min(loss)
                exp_cost = torch.exp(- (loss - beta) / self.temperature)
                weights = exp_cost / (torch.sum(exp_cost) + 1e-8)
                
                weighted_noise = torch.sum(weights.unsqueeze(1).unsqueeze(2) * noise, dim=0)
                mu[traj] = mu[traj] + weighted_noise
                
                losses.append(beta.item())

            self.wandb_run.log({f"{self.logging_prefix}/loss": np.mean(losses), "step": i + 1})
            
            if self.evaluator is not None and i % self.eval_every == 0:
                logs, successes, _, _ = self.evaluator.eval_actions(
                    mu.detach(), filename=f"{self.logging_prefix}_output_{i+1}"
                )
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": i + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                if np.all(successes):
                    break

        return mu.detach(), np.full(n_evals, np.inf)