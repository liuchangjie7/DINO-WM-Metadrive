import os
import torch
import imageio
import numpy as np
from einops import rearrange, repeat
from utils import (
    cfg_to_dict,
    seed,
    slice_trajdict_with_t,
    aggregate_dct,
    move_to_device,
    concat_trajdict,
)
from torchvision import utils


class PlanEvaluator:  # evaluator for planning
    def __init__(
        self,
        obs_0,
        obs_g,
        state_0,
        state_g,
        env,
        wm,
        frameskip,
        seed,
        preprocessor,
        n_plot_samples,
    ):
        self.obs_0 = obs_0
        self.obs_g = obs_g
        self.state_0 = state_0
        self.state_g = state_g
        self.env = env
        self.wm = wm
        self.frameskip = frameskip
        self.seed = seed
        self.preprocessor = preprocessor
        self.n_plot_samples = n_plot_samples
        self.device = next(wm.parameters()).device

        self.plot_full = False  # plot all frames or frames after frameskip

    def assign_init_cond(self, obs_0, state_0):
        self.obs_0 = obs_0
        self.state_0 = state_0

    def assign_goal_cond(self, obs_g, state_g):
        self.obs_g = obs_g
        self.state_g = state_g

    def get_init_cond(self):
        return self.obs_0, self.state_0

    def _get_trajdict_last(self, dct, length):
        new_dct = {}
        for key, value in dct.items():
            new_dct[key] = self._get_traj_last(value, length)
        return new_dct

    def _get_traj_last(self, traj_data, length):
        last_index = np.where(length == np.inf, -1, length - 1)
        last_index = last_index.astype(int)
        if isinstance(traj_data, torch.Tensor):
            traj_data = traj_data[np.arange(traj_data.shape[0]), last_index].unsqueeze(
                1
            )
        else:
            traj_data = np.expand_dims(
                traj_data[np.arange(traj_data.shape[0]), last_index], axis=1
            )
        return traj_data

    def _mask_traj(self, data, length):
        """
        Zero out everything after specified indices for each trajectory in the tensor.
        data: tensor
        """
        result = data.clone()  # Clone to preserve the original tensor
        for i in range(data.shape[0]):
            if length[i] != np.inf:
                result[i, int(length[i]) :] = 0
        return result

    def eval_actions(
        self, actions, action_len=None, filename="output", save_video=False
    ):
        """
        actions: detached torch tensors on cuda
        Returns
            metrics, and feedback from env
        """
        n_evals = actions.shape[0]
        # 👇 Debug 1: 检查输入的 action_len
        # print(f"\n[Debug] 开始评估。n_evals: {n_evals}, 传入的 action_len: {action_len}")
        if action_len is None:
            action_len = np.full(n_evals, np.inf)
            # print("[Debug] action_len 为空，已填充为 inf")
        # rollout in wm
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(self.obs_0), self.device
        )
        trans_obs_g = move_to_device(
            self.preprocessor.transform_obs(self.obs_g), self.device
        )
        with torch.no_grad():
            i_z_obses, _ = self.wm.rollout(
                obs_0=trans_obs_0,
                act=actions,
            )
        i_final_z_obs = self._get_trajdict_last(i_z_obses, action_len + 1)

        # rollout in env
        exec_actions = rearrange(
            actions.cpu(), "b t (f d) -> b (t f) d", f=self.frameskip
        )
        exec_actions = self.preprocessor.denormalize_actions(exec_actions).numpy()

        # 👇 ====== 核心修复：历史回放与画面对齐 ====== 👇
        if hasattr(self, "history_actions") and self.history_actions is not None:
            # 如果有历史动作，就把它反归一化并和当前的假想动作拼起来
            history_exec = rearrange(
                self.history_actions.cpu(), "b t (f d) -> b (t f) d", f=self.frameskip
            )
            history_exec = self.preprocessor.denormalize_actions(history_exec).numpy()
            full_exec_actions = np.concatenate([history_exec, exec_actions], axis=1)
            hist_len = history_exec.shape[1]
        else:
            full_exec_actions = exec_actions
            hist_len = 0

        # 让环境从大爆炸起点开始，老老实实跑完“历史 + 未来”所有动作
        e_obses, e_states = self.env.rollout(self.seed, self.state_0, full_exec_actions)
        
        # 跑完之后，就像剪辑视频一样，把历史部分的画面和状态“一刀切掉”！
        # 这样剩下的部分就刚好和世界模型的起点完美重合了。
        if hist_len > 0:
            e_obses = slice_trajdict_with_t(e_obses, start_idx=hist_len)
            e_states = e_states[:, hist_len:]
            
        e_visuals = e_obses["visual"]
        # 👆 ========================================= 👆
        
        e_final_obs = self._get_trajdict_last(e_obses, action_len * self.frameskip + 1)
        e_final_state = self._get_traj_last(e_states, action_len * self.frameskip + 1)[
            :, 0
        ]  # reduce dim back

        # compute eval metrics
        logs, successes = self._compute_rollout_metrics(
            e_state=e_final_state,
            e_obs=e_final_obs,
            i_z_obs=i_final_z_obs,
        )

        # plot trajs
        if self.wm.decoder is not None:
            i_visuals = self.wm.decode_obs(i_z_obses)[0]["visual"]
            # 👇 Debug 2: 看看 mask 之前和之后的形状
            # print(f"[Debug] Mask 前 i_visuals 形状: {i_visuals.shape}")
            i_visuals = self._mask_traj(i_visuals, action_len + 1)
            
            e_visuals = self.preprocessor.transform_obs_visual(e_visuals)
            # print(f"[Debug] Mask 前 e_visuals 形状: {e_visuals.shape}")
            e_visuals = self._mask_traj(e_visuals, action_len * self.frameskip + 1)
            self._plot_rollout_compare(
                e_visuals=e_visuals,
                i_visuals=i_visuals,
                successes=successes,
                save_video=save_video,
                filename=filename,
            )

        return logs, successes, e_obses, e_states

    def _compute_rollout_metrics(self, e_state, e_obs, i_z_obs):
        """
        Args
            e_state
            e_obs
            i_z_obs
        Return
            logs
            successes
        """
        eval_results = self.env.eval_state(self.state_g, e_state)
        successes = eval_results['success']

        logs = {
            f"success_rate" if key == "success" else f"mean_{key}": np.mean(value) if key != "success" else np.mean(value.astype(float))
            for key, value in eval_results.items()
        }

        print("Success rate: ", logs['success_rate'])
        # print(eval_results)

        visual_dists = np.linalg.norm(e_obs["visual"] - self.obs_g["visual"], axis=1)
        mean_visual_dist = np.mean(visual_dists)
        proprio_dists = np.linalg.norm(e_obs["proprio"] - self.obs_g["proprio"], axis=1)
        mean_proprio_dist = np.mean(proprio_dists)

        e_obs = move_to_device(self.preprocessor.transform_obs(e_obs), self.device)
        e_z_obs = self.wm.encode_obs(e_obs)
        div_visual_emb = torch.norm(e_z_obs["visual"] - i_z_obs["visual"]).item()
        div_proprio_emb = torch.norm(e_z_obs["proprio"] - i_z_obs["proprio"]).item()

        logs.update({
            "mean_visual_dist": mean_visual_dist,
            "mean_proprio_dist": mean_proprio_dist,
            "mean_div_visual_emb": div_visual_emb,
            "mean_div_proprio_emb": div_proprio_emb,
        })

        return logs, successes

    def _plot_rollout_compare(
        self, e_visuals, i_visuals, successes, save_video=False, filename=""
    ):
        """
        i_visuals may have less frames than e_visuals due to frameskip, so pad accordingly
        e_visuals: (b, t, h, w, c)
        i_visuals: (b, t, h, w, c)
        goal: (b, h, w, c)
        """
        e_visuals = e_visuals[: self.n_plot_samples]
        i_visuals = i_visuals[: self.n_plot_samples]
        goal_visual = self.obs_g["visual"][: self.n_plot_samples]
        goal_visual = self.preprocessor.transform_obs_visual(goal_visual)

        i_visuals = i_visuals.unsqueeze(2)
        i_visuals = torch.cat(
            [i_visuals] + [i_visuals] * (self.frameskip - 1),
            dim=2,
        )  # pad i_visuals (due to frameskip)
        i_visuals = rearrange(i_visuals, "b t n c h w -> b (t n) c h w")
        i_visuals = i_visuals[:, : i_visuals.shape[1] - (self.frameskip - 1)]

        correction = 0.3  # to distinguish env visuals and imagined visuals

        if save_video:
            # 创建一个 debug 目录存放图片
            debug_img_dir = f"debug_frames_{filename}"
            os.makedirs(debug_img_dir, exist_ok=True)

            for idx in range(e_visuals.shape[0]):
                success_tag = "success" if successes[idx] else "failure"
                video_name = f"{filename}_{idx}_{success_tag}.mp4"
                video_writer = imageio.get_writer(video_name, format='FFMPEG', fps=10, macro_block_size=None)

                print(f"[Debug] 正在为样本 {idx} 写入视频，预期总帧数: {e_visuals.shape[1]}")

                for i in range(e_visuals.shape[1]):
                    e_obs = e_visuals[idx, i, ...].cpu()
                    i_obs = i_visuals[idx, i, ...].cpu()
                    
                    e_obs_cat = torch.cat([e_obs, goal_visual[idx, 0].cpu() - correction], dim=2)
                    i_obs_cat = torch.cat([i_obs, goal_visual[idx, 0].cpu() - correction], dim=2)
                    frame_tensor = torch.cat([e_obs_cat - correction, i_obs_cat], dim=1)
                    
                    frame_np = rearrange(frame_tensor, "c h w -> h w c").detach().numpy()
                    frame_u8 = (((np.clip(frame_np, -1, 1) + 1.0) / 2.0) * 255.0).astype(np.uint8)
                    
                    # 👇 Debug 3: 同时保存为图片文件
                    if idx == 0: # 只存第一个样本的 debug 图，防止磁盘爆满
                        img_path = os.path.join(debug_img_dir, f"sample_{idx}_frame_{i:03d}.jpg")
                        imageio.imwrite(img_path, frame_u8)
                    
                    video_writer.append_data(frame_u8)
                
                print(f"[Debug] 样本 {idx} 视频写入完成并关闭。")
                video_writer.close()

        # pad i_visuals or subsample e_visuals
        if not self.plot_full:
            e_visuals = e_visuals[:, :: self.frameskip]
            i_visuals = i_visuals[:, :: self.frameskip]

        n_columns = e_visuals.shape[1]
        assert (
            i_visuals.shape[1] == n_columns
        ), f"Rollout lengths do not match, {e_visuals.shape[1]} and {i_visuals.shape[1]}"

        # add a goal column
        e_visuals = torch.cat([e_visuals.cpu(), goal_visual - correction], dim=1)
        i_visuals = torch.cat([i_visuals.cpu(), goal_visual - correction], dim=1)
        rollout = torch.cat([e_visuals.cpu() - correction, i_visuals.cpu()], dim=1)
        n_columns += 1

        imgs_for_plotting = rearrange(rollout, "b h c w1 w2 -> (b h) c w1 w2")
        imgs_for_plotting = (
            imgs_for_plotting * 2 - 1
            if imgs_for_plotting.min() >= 0
            else imgs_for_plotting
        )
        utils.save_image(
            imgs_for_plotting,
            f"{filename}.png",
            nrow=n_columns,  # nrow is the number of columns
            normalize=True,
            value_range=(-1, 1),
        )
