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
import logging

import torch
from omegaconf import ListConfig, OmegaConf

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, MultiLeRobotDataset
from lerobot.common.datasets.transforms import get_image_transforms


def resolve_delta_timestamps(cfg):
    """Resolves delta_timestamps config key (in-place) by using `eval`.

    Doesn't do anything if delta_timestamps is not specified or has already been resolve (as evidenced by
    the data type of its values).
    """
    delta_timestamps = cfg.training.get("delta_timestamps")
    if delta_timestamps is not None:
        for key in delta_timestamps:
            if isinstance(delta_timestamps[key], str):
                # TODO(rcadene, alexander-soare): remove `eval` to avoid exploit
                cfg.training.delta_timestamps[key] = eval(delta_timestamps[key])


def make_dataset(cfg, split: str = "train") -> LeRobotDataset | MultiLeRobotDataset:
    """
    Args:
        cfg: A Hydra config as per the LeRobot config scheme.
        split: Select the data subset used to create an instance of LeRobotDataset.
            All datasets hosted on [lerobot](https://huggingface.co/lerobot) contain only one subset: "train".
            Thus, by default, `split="train"` selects all the available data. `split` aims to work like the
            slicer in the hugging face datasets:
            https://huggingface.co/docs/datasets/v2.19.0/loading#slice-splits
            As of now, it only supports `split="train[:n]"` to load the first n frames of the dataset or
            `split="train[n:]"` to load the last n frames. For instance `split="train[:1000]"`.
    Returns:
        The LeRobotDataset.
    """
    if not isinstance(cfg.dataset_repo_id, (str, ListConfig)):
        raise ValueError(
            "Expected cfg.dataset_repo_id to be either a single string to load one dataset or a list of "
            "strings to load multiple datasets."
        )

    # A soft check to warn if the environment matches the dataset. Don't check if we are using a real world env (dora).
    if cfg.env.name != "dora":
        if isinstance(cfg.dataset_repo_id, str):
            dataset_repo_ids = [cfg.dataset_repo_id]  # single dataset
        else:
            dataset_repo_ids = cfg.dataset_repo_id  # multiple datasets

        for dataset_repo_id in dataset_repo_ids:
            if cfg.env.name not in dataset_repo_id:
                logging.warning(
                    f"There might be a mismatch between your training dataset ({dataset_repo_id=}) and your "
                    f"environment ({cfg.env.name=})."
                )

    resolve_delta_timestamps(cfg)


    # TODO: temp
    if cfg.env.name in ["foraging", "maze"]:
        from diffusion_memory.exp.foraging.generate_data import data_dict_to_hf
        from lerobot.common.datasets.compute_stats import compute_stats

        fps = cfg.fps

        if cfg.policy.name == "diffusion":
            horizon = cfg.policy.horizon
            n_obs_steps = cfg.policy.n_obs_steps
            delta_timestamps = {}
            delta_timestamps["observation.image"] = [i / fps for i in range(1 - n_obs_steps, 1)]
            delta_timestamps["observation.state"] = [i / fps for i in range(1 - n_obs_steps, 1)]
            delta_timestamps["action"] =[i / fps for i in range(1 - n_obs_steps, 1 - n_obs_steps + horizon)]
        else:
            raise NotImplementedError

        
        if cfg.dataset.get("online", False):
            assert cfg.env.name == "maze"

            from torchvision import transforms
            from lerobot.common.datasets.lerobot_dataset import LeRobotOnlineDataset
            from efficient_routing import make
            from efficient_routing.exp.maze.generate_data import optimal_path

            env_id = cfg.env.task
            obs_size = (cfg.env.image_size, cfg.env.image_size)
            episode_length = cfg.dataset.episode_length
            # not really the number of samples since online dataset is infinite
            num_samples = cfg.dataset.total_episodes

            env = make(f"env/{env_id}", params=None, obs_size=obs_size)

            def _generate_sample(idx):
                # TODO: cases where there is no path
                path = None
                while path is None:
                    state = env.reset()
                    path = optimal_path(state.grid, state.start_pos, state.goal_pos, obs_size, episode_length+1, path_upsample=4)
                
                img = env.render(state)  # image is constont over time
                states = path[:-1]  # state is current position
                actions = path[1:]  # action is next position

                obs_img = transforms.ToTensor()(img)
                # expand along time dimension
                obs_img = obs_img[None].expand(n_obs_steps, -1, -1, -1)  # (n_obs_steps, C, H, W)
                obs_state = torch.Tensor(states[:n_obs_steps])
                tgt_action = torch.Tensor(actions)

                assert obs_img.shape == (2, 3, 96, 96)
                assert obs_state.shape == (2, 2)
                assert tgt_action.shape == (104, 2)

                return {
                    "observation.image": obs_img,
                    "observation.state": obs_state,
                    "action": tgt_action,
                    "action_is_pad": torch.full(tgt_action.shape, False),  # same shape as tgt action
                    "index": idx
                }

            from datasets.features import Features, Image, Sequence, Value

            features = {}
            features["observation.image"] = Image()
            features["observation.state"] = Sequence(length=2, feature=Value(dtype="float32"))
            features["action"] = Sequence(length=2, feature=Value(dtype="float32"))
            # features["episode_index"] = Value(dtype="int64", id=None)
            # features["frame_index"] = Value(dtype="int64", id=None)
            # features["timestamp"] = Value(dtype="float32", id=None)
            # features["next.reward"] = Value(dtype="float32", id=None)
            # features["next.done"] = Value(dtype="bool", id=None)
            # features["next.success"] = Value(dtype="bool", id=None)
            features["index"] = Value(dtype="int64", id=None)
            features = Features(features)


            # TODO: can't use online dataset to compute stats because of assert in compute_stats (line 136)
            # temp_dataset = LeRobotOnlineDataset(
            #     generate_sample_fn=_generate_sample,
            #     features=features,
            #     num_samples=num_samples,
            # )
            # # TODO: requires dataset.features!!
            # stats = compute_stats(temp_dataset, 32, 1)

            from efficient_routing.exp.maze.generate_data import generate_maze_data
            data_dict, episode_data_index = generate_maze_data(
                env_id=cfg.env.task,
                agent_id="astar",
                total_episodes=32,
                episode_length=cfg.dataset.episode_length,
                obs_size=(cfg.env.image_size, cfg.env.image_size),
                fps=fps
            )
            hf_dataset = data_dict_to_hf(data_dict)
            info = {"fps": fps}
            temp_dataset = LeRobotDataset.from_preloaded(
                hf_dataset=hf_dataset,
                episode_data_index=episode_data_index
            )
            stats = compute_stats(temp_dataset, 32, 8)


            dataset = LeRobotOnlineDataset(
                generate_sample_fn=_generate_sample,
                features=features,
                num_samples=num_samples,
                delta_timestamps=delta_timestamps,
                stats=stats
            )

            return dataset

        # TODO: caching
        print("Generating data...")

        if cfg.env.name == "foraging":
            from diffusion_memory.exp.foraging.generate_data import generate_data
            data_dict, episode_data_index = generate_data(
                env_id=cfg.env.task,
                agent_id=cfg.env.task,
                total_episodes=cfg.dataset.total_episodes,
                episode_length=cfg.dataset.episode_length,
                fps=fps
            )

        elif cfg.env.name == "maze":
            from efficient_routing.exp.maze.generate_data import generate_maze_data
            data_dict, episode_data_index = generate_maze_data(
                env_id=cfg.env.task,
                agent_id="astar",
                total_episodes=cfg.dataset.total_episodes,
                episode_length=cfg.dataset.episode_length,
                obs_size=(cfg.env.image_size, cfg.env.image_size),
                fps=fps
            )

        print("Data generated. Convert to HF dataset.")

        hf_dataset = data_dict_to_hf(data_dict)
        info = {"fps": fps}

        print("Done. Convert to LeRobotDataset.")
        # create dataset without delta_timesteps to compute stats (compute_stats function doesn't expect an additional dim for time)
        temp_dataset = LeRobotDataset.from_preloaded(
            hf_dataset=hf_dataset,
            episode_data_index=episode_data_index
        )
        # breakpoint()
        stats = compute_stats(temp_dataset, 32, 8)

        dataset = LeRobotDataset.from_preloaded(
            hf_dataset=hf_dataset,
            episode_data_index=episode_data_index,
            delta_timestamps=delta_timestamps,
            info=info,
            stats=stats
        )
        
        return dataset

    # TODO(rcadene): add data augmentations
    image_transforms = None
    if cfg.training.image_transforms.enable:
        cfg_tf = cfg.training.image_transforms
        image_transforms = get_image_transforms(
            brightness_weight=cfg_tf.brightness.weight,
            brightness_min_max=cfg_tf.brightness.min_max,
            contrast_weight=cfg_tf.contrast.weight,
            contrast_min_max=cfg_tf.contrast.min_max,
            saturation_weight=cfg_tf.saturation.weight,
            saturation_min_max=cfg_tf.saturation.min_max,
            hue_weight=cfg_tf.hue.weight,
            hue_min_max=cfg_tf.hue.min_max,
            sharpness_weight=cfg_tf.sharpness.weight,
            sharpness_min_max=cfg_tf.sharpness.min_max,
            max_num_transforms=cfg_tf.max_num_transforms,
            random_order=cfg_tf.random_order,
        )

    if isinstance(cfg.dataset_repo_id, str):
        dataset = LeRobotDataset(
            cfg.dataset_repo_id,
            split=split,
            delta_timestamps=cfg.training.get("delta_timestamps"),
            image_transforms=image_transforms,
            video_backend=cfg.video_backend,
        )
    else:
        dataset = MultiLeRobotDataset(
            cfg.dataset_repo_id,
            split=split,
            delta_timestamps=cfg.training.get("delta_timestamps"),
            image_transforms=image_transforms,
            video_backend=cfg.video_backend,
        )

    if cfg.get("override_dataset_stats"):
        for key, stats_dict in cfg.override_dataset_stats.items():
            for stats_type, listconfig in stats_dict.items():
                # example of stats_type: min, max, mean, std
                stats = OmegaConf.to_container(listconfig, resolve=True)
                dataset.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset
