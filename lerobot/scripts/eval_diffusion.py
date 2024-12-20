#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evaluate a policy on an environment by running rollouts and computing metrics.

Usage examples:

You want to evaluate a model from the hub (eg: https://huggingface.co/lerobot/diffusion_pusht)
for 10 episodes.

```
python lerobot/scripts/eval.py -p lerobot/diffusion_pusht eval.n_episodes=10
```

OR, you want to evaluate a model checkpoint from the LeRobot training script for 10 episodes.

```
python lerobot/scripts/eval.py \
    -p outputs/train/diffusion_pusht/checkpoints/005000/pretrained_model \
    eval.n_episodes=10
```

Note that in both examples, the repo/folder should contain at least `config.json`, `config.yaml` and
`model.safetensors`.

Note the formatting for providing the number of episodes. Generally, you may provide any number of arguments
with `qualified.parameter.name=value`. In this case, the parameter eval.n_episodes appears as `n_episodes`
nested under `eval` in the `config.yaml` found at
https://huggingface.co/lerobot/diffusion_pusht/tree/main.
"""

import argparse
import json
import logging
import threading
import time
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime as dt
from pathlib import Path
from typing import Callable

import einops
import gymnasium as gym
import numpy as np
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import RepositoryNotFoundError
from huggingface_hub.utils._validators import HFValidationError
from torch import Tensor, nn
from tqdm import trange

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.envs.factory import make_env
from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.logger import log_output_dir
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.policy_protocol import Policy
from lerobot.common.policies.utils import get_device_from_parameters, get_dtype_from_parameters
from lerobot.common.utils.io_utils import write_video
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_hydra_config,
    init_logging,
    inside_slurm,
    set_global_seed,
)



def rollout(
    env: gym.vector.VectorEnv,
    policy: Policy,
    seeds: list[int] | None = None,
    return_observations: bool = False,
    render_callback: Callable[[gym.vector.VectorEnv], None] | None = None,
) -> dict:
    """Run a batched policy rollout once through a batch of environments.

    Note that all environments in the batch are run until the last environment is done. This means some
    data will probably need to be discarded (for environments that aren't the first one to be done).

    The return dictionary contains:
        (optional) "observation": A a dictionary of (batch, sequence + 1, *) tensors mapped to observation
            keys. NOTE the that this has an extra sequence element relative to the other keys in the
            dictionary. This is because an extra observation is included for after the environment is
            terminated or truncated.
        "action": A (batch, sequence, action_dim) tensor of actions applied based on the observations (not
            including the last observations).
        "reward": A (batch, sequence) tensor of rewards received for applying the actions.
        "success": A (batch, sequence) tensor of success conditions (the only time this can be True is upon
            environment termination/truncation).
        "done": A (batch, sequence) tensor of **cumulative** done conditions. For any given batch element,
            the first True is followed by True's all the way till the end. This can be used for masking
            extraneous elements from the sequences above.

    Args:
        env: The batch of environments.
        policy: The policy. Must be a PyTorch nn module.
        seeds: The environments are seeded once at the start of the rollout. If provided, this argument
            specifies the seeds for each of the environments.
        return_observations: Whether to include all observations in the returned rollout data. Observations
            are returned optionally because they typically take more memory to cache. Defaults to False.
        render_callback: Optional rendering callback to be used after the environments are reset, and after
            every step.
    Returns:
        The dictionary described above.
    """
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."
    device = get_device_from_parameters(policy)

    # Reset the policy and environments.
    policy.reset()

    observation, info = env.reset(seed=seeds)
    if render_callback is not None:
        render_callback(env)

    all_observations = []
    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []
    # Modified: keep track of the samples generated during the diffusion
    all_samples = []

    step = 0
    # Keep track of which environments are done.
    done = np.array([False] * env.num_envs)
    max_steps = env.call("_max_episode_steps")[0]
    progbar = trange(
        max_steps,
        desc=f"Running rollout with at most {max_steps} steps",
        disable=inside_slurm(),  # we dont want progress bar when we use slurm, since it clutters the logs
        leave=False,
    )
    while not np.all(done):
        # Numpy array to tensor and changing dictionary keys to LeRobot policy format.
        observation = preprocess_observation(observation)
        if return_observations:
            all_observations.append(deepcopy(observation))

        observation = {key: observation[key].to(device, non_blocking=True) for key in observation}

        with torch.inference_mode():
            action = policy.select_action(observation)

        # Convert to CPU / numpy.
        action = action.to("cpu").numpy()
        assert action.ndim == 2, "Action dimensions should be (batch, action_dim)"

        # Apply the next action.
        observation, reward, terminated, truncated, info = env.step(action)
        if render_callback is not None:
            render_callback(env)

        # VectorEnv stores is_success in `info["final_info"][env_index]["is_success"]`. "final_info" isn't
        # available of none of the envs finished.
        if "final_info" in info:
            successes = [info["is_success"] if info is not None else False for info in info["final_info"]]
        else:
            successes = [False] * env.num_envs

        # Keep track of which environments are done so far.
        done = terminated | truncated | done

        all_actions.append(torch.from_numpy(action))
        all_rewards.append(torch.from_numpy(reward))
        all_dones.append(torch.from_numpy(done))
        all_successes.append(torch.tensor(successes))

        # Modified: get the generated samples and unnormalize them
        samples = torch.tensor(policy.diffusion.samples)
        samples = policy.unnormalize_outputs({"action": samples})["action"]
        all_samples.append(samples.to("cpu"))

        step += 1
        running_success_rate = (
            einops.reduce(torch.stack(all_successes, dim=1), "b n -> b", "any").numpy().mean()
        )
        progbar.set_postfix({"running_success_rate": f"{running_success_rate.item() * 100:.1f}%"})
        progbar.update()

    # Track the final observation.
    if return_observations:
        observation = preprocess_observation(observation)
        all_observations.append(deepcopy(observation))

    # Stack the sequence along the first dimension so that we have (batch, sequence, *) tensors.
    ret = {
        "action": torch.stack(all_actions, dim=1),
        "reward": torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done": torch.stack(all_dones, dim=1),
        # Modified
        "diffusion_samples": torch.stack(all_samples, dim=1),
    }
    if return_observations:
        stacked_observations = {}
        for key in all_observations[0]:
            stacked_observations[key] = torch.stack([obs[key] for obs in all_observations], dim=1)
        ret["observation"] = stacked_observations

    return ret


def eval_policy(
    env: gym.vector.VectorEnv,
    policy: torch.nn.Module,
    n_episodes: int,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
) -> dict:
    """
    Args:
        env: The batch of environments.
        policy: The policy.
        n_episodes: The number of episodes to evaluate.
        max_episodes_rendered: Maximum number of episodes to render into videos.
        videos_dir: Where to save rendered videos.
        return_episode_data: Whether to return episode data for online training. Incorporates the data into
            the "episodes" key of the returned dictionary.
        start_seed: The first seed to use for the first individual rollout. For all subsequent rollouts the
            seed is incremented by 1. If not provided, the environments are not manually seeded.
    Returns:
        Dictionary with metrics and data regarding the rollouts.
    """
    if max_episodes_rendered > 0 and not videos_dir:
        raise ValueError("If max_episodes_rendered > 0, videos_dir must be provided.")

    assert isinstance(policy, Policy)
    start = time.time()
    policy.eval()

    # Determine how many batched rollouts we need to get n_episodes. Note that if n_episodes is not evenly
    # divisible by env.num_envs we end up discarding some data in the last batch.
    n_batches = n_episodes // env.num_envs + int((n_episodes % env.num_envs) != 0)

    # Keep track of some metrics.
    sum_rewards = []
    max_rewards = []
    all_successes = []
    all_seeds = []
    threads = []  # for video saving threads
    n_episodes_rendered = 0  # for saving the correct number of videos

    # Callback for visualization.
    def render_frame(env: gym.vector.VectorEnv):
        # noqa: B023
        if n_episodes_rendered >= max_episodes_rendered:
            return
        n_to_render_now = min(max_episodes_rendered - n_episodes_rendered, env.num_envs)
        if isinstance(env, gym.vector.SyncVectorEnv):
            ep_frames.append(np.stack([env.envs[i].render() for i in range(n_to_render_now)]))  # noqa: B023
        elif isinstance(env, gym.vector.AsyncVectorEnv):
            # Here we must render all frames and discard any we don't need.
            ep_frames.append(np.stack(env.call("render")[:n_to_render_now]))

    if max_episodes_rendered > 0:
        video_paths: list[str] = []

    if return_episode_data:
        episode_data: dict | None = None

    # we dont want progress bar when we use slurm, since it clutters the logs
    progbar = trange(n_batches, desc="Stepping through eval batches", disable=inside_slurm())
    for batch_ix in progbar:
        # Cache frames for rendering videos. Each item will be (b, h, w, c), and the list indexes the rollout
        # step.
        if max_episodes_rendered > 0:
            ep_frames: list[np.ndarray] = []

        if start_seed is None:
            seeds = None
        else:
            seeds = range(
                start_seed + (batch_ix * env.num_envs), start_seed + ((batch_ix + 1) * env.num_envs)
            )
        rollout_data = rollout(
            env,
            policy,
            seeds=list(seeds) if seeds else None,
            return_observations=return_episode_data,
            render_callback=render_frame if max_episodes_rendered > 0 else None,
        )

        # Figure out where in each rollout sequence the first done condition was encountered (results after
        # this won't be included).
        n_steps = rollout_data["done"].shape[1]
        # Note: this relies on a property of argmax: that it returns the first occurrence as a tiebreaker.
        done_indices = torch.argmax(rollout_data["done"].to(int), dim=1)

        # Make a mask with shape (batch, n_steps) to mask out rollout data after the first done
        # (batch-element-wise). Note the `done_indices + 1` to make sure to keep the data from the done step.
        mask = (torch.arange(n_steps) <= einops.repeat(done_indices + 1, "b -> b s", s=n_steps)).int()
        # Extend metrics.
        batch_sum_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "sum")
        sum_rewards.extend(batch_sum_rewards.tolist())
        batch_max_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "max")
        max_rewards.extend(batch_max_rewards.tolist())
        batch_successes = einops.reduce((rollout_data["success"] * mask), "b n -> b", "any")
        all_successes.extend(batch_successes.tolist())
        if seeds:
            all_seeds.extend(seeds)
        else:
            all_seeds.append(None)

        # FIXME: episode_data is either None or it doesn't exist
        if return_episode_data:
            this_episode_data = _compile_episode_data(
                rollout_data,
                done_indices,
                start_episode_index=batch_ix * env.num_envs,
                start_data_index=(0 if episode_data is None else (episode_data["index"][-1].item() + 1)),
                fps=env.unwrapped.metadata["render_fps"],
            )
            if episode_data is None:
                episode_data = this_episode_data
            else:
                # Some sanity checks to make sure we are correctly compiling the data.
                assert episode_data["episode_index"][-1] + 1 == this_episode_data["episode_index"][0]
                assert episode_data["index"][-1] + 1 == this_episode_data["index"][0]
                # Concatenate the episode data.
                episode_data = {k: torch.cat([episode_data[k], this_episode_data[k]]) for k in episode_data}

        # Maybe render video for visualization.
        if max_episodes_rendered > 0 and len(ep_frames) > 0:
            batch_stacked_frames = np.stack(ep_frames, axis=1)  # (b, t, *)
            for stacked_frames, done_index in zip(
                batch_stacked_frames, done_indices.flatten().tolist(), strict=False
            ):
                if n_episodes_rendered >= max_episodes_rendered:
                    break

                videos_dir.mkdir(parents=True, exist_ok=True)
                video_path = videos_dir / f"eval_episode_{n_episodes_rendered}.mp4"
                video_paths.append(str(video_path))
                thread = threading.Thread(
                    target=write_video,
                    args=(
                        str(video_path),
                        stacked_frames[: done_index + 1],  # + 1 to capture the last observation
                        env.unwrapped.metadata["render_fps"],
                    ),
                )
                thread.start()
                threads.append(thread)

                # Nathan: added to save as gif
                gif_path = videos_dir / f"gif/eval_episode_{n_episodes_rendered}.gif"
                gif_path.parent.mkdir(parents=True, exist_ok=True)
                _frames = stacked_frames[: done_index + 1]
                # skip frames
                _frames = _frames[::2]
                # adjust fps
                fps = env.unwrapped.metadata["render_fps"] // 2
                thread = threading.Thread(
                    target=write_video,
                    args=(
                        str(gif_path),
                        _frames,  # + 1 to capture the last observation
                        fps,
                    ),
                )
                thread.start()
                threads.append(thread)

                n_episodes_rendered += 1

        progbar.set_postfix(
            {"running_success_rate": f"{np.mean(all_successes[:n_episodes]).item() * 100:.1f}%"}
        )

    # Wait till all video rendering threads are done.
    for thread in threads:
        thread.join()

    # Compile eval info.
    info = {
        "per_episode": [
            {
                "episode_ix": i,
                "sum_reward": sum_reward,
                "max_reward": max_reward,
                "success": success,
                "seed": seed,
            }
            for i, (sum_reward, max_reward, success, seed) in enumerate(
                zip(
                    sum_rewards[:n_episodes],
                    max_rewards[:n_episodes],
                    all_successes[:n_episodes],
                    all_seeds[:n_episodes],
                    strict=True,
                )
            )
        ],
        "aggregated": {
            "avg_sum_reward": float(np.nanmean(sum_rewards[:n_episodes])),
            "avg_max_reward": float(np.nanmean(max_rewards[:n_episodes])),
            "pc_success": float(np.nanmean(all_successes[:n_episodes]) * 100),
            "eval_s": time.time() - start,
            "eval_ep_s": (time.time() - start) / n_episodes,
        },
    }

    if return_episode_data:
        info["episodes"] = episode_data

    if max_episodes_rendered > 0:
        info["video_paths"] = video_paths

    return info


def _compile_episode_data(
    rollout_data: dict, done_indices: Tensor, start_episode_index: int, start_data_index: int, fps: float
) -> dict:
    """Convenience function for `eval_policy(return_episode_data=True)`

    Compiles all the rollout data into a Hugging Face dataset.

    Similar logic is implemented when datasets are pushed to hub (see: `push_to_hub`).
    """
    ep_dicts = []
    total_frames = 0
    for ep_ix in range(rollout_data["action"].shape[0]):
        # + 2 to include the first done frame and the last observation frame.
        num_frames = done_indices[ep_ix].item() + 2
        total_frames += num_frames

        # Here we do `num_frames - 1` as we don't want to include the last observation frame just yet.
        ep_dict = {
            "action": rollout_data["action"][ep_ix, : num_frames - 1],
            "episode_index": torch.tensor([start_episode_index + ep_ix] * (num_frames - 1)),
            "frame_index": torch.arange(0, num_frames - 1, 1),
            "timestamp": torch.arange(0, num_frames - 1, 1) / fps,
            "next.done": rollout_data["done"][ep_ix, : num_frames - 1],
            "next.success": rollout_data["success"][ep_ix, : num_frames - 1],
            "next.reward": rollout_data["reward"][ep_ix, : num_frames - 1].type(torch.float32),
        }

        # For the last observation frame, all other keys will just be copy padded.
        for k in ep_dict:
            ep_dict[k] = torch.cat([ep_dict[k], ep_dict[k][-1:]])

        for key in rollout_data["observation"]:
            ep_dict[key] = rollout_data["observation"][key][ep_ix, :num_frames]

        ep_dicts.append(ep_dict)

    data_dict = {}
    for key in ep_dicts[0]:
        data_dict[key] = torch.cat([x[key] for x in ep_dicts])

    data_dict["index"] = torch.arange(start_data_index, start_data_index + total_frames, 1)

    # Modified: add diffusion samples
    data_dict["diffusion_samples"] = rollout_data["diffusion_samples"]

    return data_dict


def conditional_sample(
    self, batch_size: int, global_cond: Tensor | None = None, generator: torch.Generator | None = None
) -> Tensor:
    """
    Modified version of DiffusionModel.conditional_sample that saves the samples generated during the difussion. Useful for visualizing the diffusion process.
    """
    device = get_device_from_parameters(self)
    dtype = get_dtype_from_parameters(self)

    # Sample prior.
    sample = torch.randn(
        size=(batch_size, self.config.horizon, self.config.output_shapes["action"][0]),
        dtype=dtype,
        device=device,
        generator=generator,
    )

    self.noise_scheduler.set_timesteps(self.num_inference_steps)

    samples = []  # Modified to save the samples (x_t-1)
    original_samples = []  # Modified to save the predicted original samples (x_0)

    for t in self.noise_scheduler.timesteps:
        # Predict model output.
        model_output = self.unet(
            sample,
            torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device),
            global_cond=global_cond,
        )
        # Compute previous image: x_t -> x_t-1
        out = self.noise_scheduler.step(model_output, t, sample, generator=generator)
        sample = out.prev_sample

        # Modified
        if hasattr(self, "guidance_callback"):
            sample = self.guidance_callback(sample, t)

        samples.append(sample.clone().detach())
        original_samples.append(out.pred_original_sample.clone().detach())

    # Modified: save samples as attribute (so that don't change the interface)
    samples = torch.stack(samples)  # diffusion_steps x batch_size x horizon x action_dim
    # transpose to have batch_size as first dim (to be consistent with the other rollout_data)
    self.samples = samples.transpose(0, 1)  # batch_size x diffusion_steps x horizon x action_dim
    self.original_samples = torch.stack(original_samples).transpose(0, 1)

    return sample



def analyze_episodes(episodes, out_dir, max_episodes_rendered=20, plot_endpoint_barchart=False):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    import matplotlib.cm as cm

    fig_dir = Path(out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    # episodes.keys(): 'action', 'episode_index', 'frame_index', 'timestamp', 'next.done', 'next.success', 'next.reward', 'observation.image', 'observation.state', 'index'
    episode_index = episodes["episode_index"]
    n_episodes_rendered_so_far = 0

    for episode_idx in episode_index.unique():
        if n_episodes_rendered_so_far >= max_episodes_rendered:
            break
        
        _fig_dir = fig_dir / ("episode_" + str(episode_idx.item()))
        _fig_dir.mkdir(parents=True, exist_ok=True)

        action = episodes["action"][episode_index == episode_idx]
        obs_state = episodes["observation.state"][episode_index == episode_idx]
        t = episodes["timestamp"][episode_index == episode_idx]

        plt.figure()
        plt.plot(t, action[:, 0])
        plt.plot(t, action[:, 1])
        plt.xlabel("Time (s)")
        plt.ylabel("Action")
        plt.savefig(_fig_dir / "action.png")

        plt.figure()
        plt.plot(t, obs_state[:, 0])
        plt.plot(t, obs_state[:, 1])
        plt.xlabel("Time (s)")
        plt.ylabel("Position")
        plt.savefig(Path(_fig_dir) / "observation_state.png")

        # animate diffusion samples
        # diffusion_samples tensor has different shape than the other tensors because didn't concat in the _compile_episode_data
        # shape: n_episodes x n_rollouts x n_diffusion_steps x horizon x action_dim
        samples = episodes["diffusion_samples"][episode_idx]
        # animate the diffusion for the first rollout (from the initial position of the agent)
        samples = samples[0]
        # samples = samples[80:]

        num_points = samples.shape[1]
        color_indices = np.linspace(0, 1, num_points)
        colors = cm.viridis(color_indices)  # Replace 'viridis' with your desired colormap

        fig = plt.figure()
        plt.axis('equal')
        plt.axis('off')
        scatter = plt.scatter(samples[0, :, 0], samples[0, :, 1], c=colors)

        # origin
        plt.scatter([0], [0], marker="o", edgecolor='black', facecolor='none')

        # target positions
        target_positions = np.array([
            [0.5, 0.0],
            [-0.5, 0.0],
            [0.0, 0.5],
            [0.0, -0.5],
        ])

        for pos in target_positions:
            plt.scatter(pos[0], pos[1], marker='o', edgecolor='black', facecolor='none')


        def update(frame_number):
            scatter.set_offsets(samples[frame_number, :, :2])
            return scatter,
        
        ani = FuncAnimation(fig, update, frames=len(samples), blit=True)
        writer = PillowWriter(fps=30)
        ani.save(_fig_dir / "diffusion_samples.gif", writer=writer)

        n_episodes_rendered_so_far += 1


    if plot_endpoint_barchart:
        # Calculate distance to each of the four endpoints:
        endpoints = np.array([[-0.5, 0], # left
                                [0.5, 0], # right
                                [0, 0.5], # top
                                [0, -0.5]]) # bottom
        closest_endpoint_count = [0] * 5
        
        for episode_idx in episode_index.unique():
            obs_state = episodes["observation.state"][episode_index == episode_idx]
            last_state = obs_state[-1].detach().numpy()
            dists = np.linalg.norm(last_state - endpoints, axis=-1)
            closest_index = np.argmin(dists)
            if dists[closest_index] >= 0.2:
                closest_index = 4 # means it is close to none of them
            closest_endpoint_count[closest_index] += 1
        
        plt.figure()
        plt.bar(["left", "right", "top", "bottom", "none"], closest_endpoint_count)
        plt.xlabel("Closest endpoint")
        plt.ylabel("Count")
        plt.savefig(fig_dir / "endpoint_frequency.png")


    # dim reduction on the diffusion samples
    samples = episodes["diffusion_samples"]  # shape: n_episodes x n_rollouts x n_diffusion_steps x horizon x action_dim
    n_episodes = samples.shape[0]
    n_diffusion_steps = samples.shape[2]

    samples = samples[:, 0]  # only consider the first rollout
    # samples = samples[:, 80:]
    samples = samples.reshape(-1, samples.shape[-2]*samples.shape[-1])  # last dim = horizon x action_dim

    colors = cm.viridis(np.linspace(0, 1, n_diffusion_steps))

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca_samples = pca.fit_transform(samples)
    pca_samples = pca_samples.reshape(n_episodes, n_diffusion_steps, 2)

    plt.figure(figsize=(4, 4), dpi=300)
    for episode_idx in range(n_episodes):
        plt.scatter(pca_samples[episode_idx, :, 0], pca_samples[episode_idx, :, 1], c=colors, marker=".")
    plt.axis('off')
    plt.axis('equal')
    plt.savefig(fig_dir / "diffusion_samples_pca.png")

    from sklearn.manifold import MDS
    mds = MDS(n_components=2)
    mds_samples = mds.fit_transform(samples)
    mds_samples = mds_samples.reshape(n_episodes, n_diffusion_steps, 2)

    plt.figure(figsize=(4, 4), dpi=300)
    for episode_idx in range(n_episodes):
        plt.scatter(mds_samples[episode_idx, :, 0], mds_samples[episode_idx, :, 1], c=colors, marker=".")
    plt.axis("off")
    plt.axis("equal")
    plt.savefig(fig_dir / "diffusion_samples_mds.png")



class TargetDistanceGuidance:
    def __init__(self, batch_size, normalize=None, T=25, lambd=30, reach_threshold=0.2, device="cpu"):
        # TODO: normalize the targets
        # sequence of target to reach in order
        self.targets = torch.tensor([
            [0, 1.0],
            [0, 0],
            [0, -1.0],
            [0, 0],
            [1.0, 0],
            [0, 0],
            [-1.0, 0],
            [0, 0],
            [0, 1.0],
            [0, 0],
            [0, -1.0],
            [0, 0]
        ]).to(device)  # num_targets x 2
        self.current_target_idx = torch.zeros(batch_size, dtype=torch.int).to(device)  # batch_size
        self.T = T
        self.lambd = lambd
        self.reach_threshold = reach_threshold

    def __call__(self, model, sample, t):
        target = self.targets[self.current_target_idx]  # batch_size x 2

        if t >= self.T:
            with torch.inference_mode(False):
                _sample = sample.clone().detach().requires_grad_(True)
                distance = torch.norm(_sample - target[:, None], p=2, dim=-1)
                distance = distance.mean()
                gradient = torch.autograd.grad(distance, _sample)[0]

            sample = sample - self.lambd * gradient
        
        if t == 0:
            # check if target is reached
            distance = torch.norm(sample - target[:, None], p=2, dim=-1)  # batch_size x horizon
            # TODO: check the max distance to target?
            is_reached = torch.mean(distance, dim=1) < self.reach_threshold  # batch_size

            # increment the target index if the target is reached
            self.current_target_idx = is_reached * (self.current_target_idx + 1) + ~is_reached * self.current_target_idx

        return sample


# def target_distance_guidance(model, sample, t):
#     if t < 50:
#         return sample

#     target = torch.tensor([0, 1.0], device=sample.device)
#     lambd = 60

#     with torch.inference_mode(False):
#         _sample = sample.clone().detach().requires_grad_(True)

#         distance = torch.norm(_sample - target, p=2, dim=-1)
#         distance = distance.mean()
#         gradient = torch.autograd.grad(distance, _sample)[0]

#     sample = sample - lambd * gradient
#     return sample


def main(
    pretrained_policy_path: Path | None = None,
    hydra_cfg_path: str | None = None,
    out_dir: str | None = None,
    config_overrides: list[str] | None = None,
):
    assert (pretrained_policy_path is None) ^ (hydra_cfg_path is None)
    if pretrained_policy_path is not None:
        hydra_cfg = init_hydra_config(str(pretrained_policy_path / "config.yaml"), config_overrides)
    else:
        hydra_cfg = init_hydra_config(hydra_cfg_path, config_overrides)

    if hydra_cfg.eval.batch_size > hydra_cfg.eval.n_episodes:
        raise ValueError(
            "The eval batch size is greater than the number of eval episodes "
            f"({hydra_cfg.eval.batch_size} > {hydra_cfg.eval.n_episodes}). As a result, {hydra_cfg.eval.batch_size} "
            f"eval environments will be instantiated, but only {hydra_cfg.eval.n_episodes} will be used. "
            "This might significantly slow down evaluation. To fix this, you should update your command "
            f"to increase the number of episodes to match the batch size (e.g. `eval.n_episodes={hydra_cfg.eval.batch_size}`), "
            f"or lower the batch size (e.g. `eval.batch_size={hydra_cfg.eval.n_episodes}`)."
        )

    if out_dir is None:
        out_dir = f"outputs/eval/{dt.now().strftime('%Y-%m-%d/%H-%M-%S')}_{hydra_cfg.env.name}_{hydra_cfg.policy.name}"

    # Check device is available
    device = get_safe_torch_device(hydra_cfg.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_global_seed(hydra_cfg.seed)

    log_output_dir(out_dir)

    logging.info("Making environment.")
    env = make_env(hydra_cfg)

    logging.info("Making policy.")
    if hydra_cfg_path is None:
        policy = make_policy(hydra_cfg=hydra_cfg, pretrained_policy_name_or_path=str(pretrained_policy_path))
    else:
        # Note: We need the dataset stats to pass to the policy's normalization modules.
        policy = make_policy(hydra_cfg=hydra_cfg, dataset_stats=make_dataset(hydra_cfg).stats)

    assert isinstance(policy, nn.Module)
    policy.eval()

    # modfiy the policy to save the samples generated during the diffusion
    policy.diffusion.conditional_sample = conditional_sample.__get__(policy.diffusion)

    # add guidance callback
    guidance = TargetDistanceGuidance(batch_size=hydra_cfg.eval.batch_size, device=device)
    def _guidance_callback(model, sample, t, guidance=guidance):
        return guidance(model, sample, t)

    policy.diffusion.guidance_callback = _guidance_callback.__get__(policy.diffusion)
    # policy.diffusion.guidance_callback = target_distance_guidance.__get__(policy.diffusion)

    max_episodes_rendered = 20

    episodes_dir = Path(out_dir) / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    if not (episodes_dir / "episodes.pt").exists():
        with torch.no_grad(), torch.autocast(device_type=device.type) if hydra_cfg.use_amp else nullcontext():
            info = eval_policy(
                env,
                policy,
                hydra_cfg.eval.n_episodes,
                max_episodes_rendered=max_episodes_rendered,
                videos_dir=Path(out_dir) / "videos",
                start_seed=hydra_cfg.seed,
                return_episode_data=True
            )
        episodes = info["episodes"]
        torch.save(episodes, episodes_dir / "episodes.pt")

        # Save info (without episodes)
        with open(Path(out_dir) / "eval_info.json", "w") as f:
            _info = {k: v for k, v in info.items() if k != "episodes"}
            json.dump(_info, f, indent=2)

    else:
        # load
        episodes = torch.load(episodes_dir / "episodes.pt")

    env.close()

    plot_endpoint_barchart = 'straight_lines' in hydra_cfg.env['task']
    analyze_episodes(episodes, out_dir, max_episodes_rendered=max_episodes_rendered, plot_endpoint_barchart=plot_endpoint_barchart)

    logging.info("End of eval")


def get_pretrained_policy_path(pretrained_policy_name_or_path, revision=None):
    try:
        pretrained_policy_path = Path(snapshot_download(pretrained_policy_name_or_path, revision=revision))
    except (HFValidationError, RepositoryNotFoundError) as e:
        if isinstance(e, HFValidationError):
            error_message = (
                "The provided pretrained_policy_name_or_path is not a valid Hugging Face Hub repo ID."
            )
        else:
            error_message = (
                "The provided pretrained_policy_name_or_path was not found on the Hugging Face Hub."
            )

        logging.warning(f"{error_message} Treating it as a local directory.")
        pretrained_policy_path = Path(pretrained_policy_name_or_path)
    if not pretrained_policy_path.is_dir() or not pretrained_policy_path.exists():
        raise ValueError(
            "The provided pretrained_policy_name_or_path is not a valid/existing Hugging Face Hub "
            "repo ID, nor is it an existing local directory."
        )
    return pretrained_policy_path


if __name__ == "__main__":
    init_logging()

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-p",
        "--pretrained-policy-name-or-path",
        help=(
            "Either the repo ID of a model hosted on the Hub or a path to a directory containing weights "
            "saved using `Policy.save_pretrained`. If not provided, the policy is initialized from scratch "
            "(useful for debugging). This argument is mutually exclusive with `--config`."
        ),
    )
    group.add_argument(
        "--config",
        help=(
            "Path to a yaml config you want to use for initializing a policy from scratch (useful for "
            "debugging). This argument is mutually exclusive with `--pretrained-policy-name-or-path` (`-p`)."
        ),
    )
    parser.add_argument("--revision", help="Optionally provide the Hugging Face Hub revision ID.")
    parser.add_argument(
        "--out-dir",
        help=(
            "Where to save the evaluation outputs. If not provided, outputs are saved in "
            "outputs/eval/{timestamp}_{env_name}_{policy_name}"
        ),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )
    args = parser.parse_args()

    if args.pretrained_policy_name_or_path is None:
        main(hydra_cfg_path=args.config, out_dir=args.out_dir, config_overrides=args.overrides)
    else:
        pretrained_policy_path = get_pretrained_policy_path(
            args.pretrained_policy_name_or_path, revision=args.revision
        )

        main(
            pretrained_policy_path=pretrained_policy_path,
            out_dir=args.out_dir,
            config_overrides=args.overrides,
        )
