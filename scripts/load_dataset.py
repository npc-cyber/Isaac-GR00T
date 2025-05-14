# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script is a replication of the notebook `getting_started/load_dataset.ipynb`
"""

import argparse
import json
import pathlib
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import time

from gr00t.data.dataset import (
    LE_ROBOT_MODALITY_FILENAME,
    LeRobotSingleDataset,
    ModalityConfig,
    LeRobotMixtureDataset,
)
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.utils.misc import any_describe


def get_modality_keys(dataset_path: pathlib.Path) -> dict[str, list[str]]:
    """
    Get the modality keys from the dataset path.
    Returns a dictionary with modality types as keys and their corresponding modality keys as values,
    maintaining the order: video, state, action, annotation
    """
    modality_path = dataset_path / LE_ROBOT_MODALITY_FILENAME
    with open(modality_path, "r") as f:
        modality_meta = json.load(f)

    # Initialize dictionary with ordered keys
    modality_dict = {}
    for key in modality_meta.keys():
        modality_dict[key] = []
        for modality in modality_meta[key]:
            modality_dict[key].append(f"{key}.{modality}")
    return modality_dict

def plot_state_action_space(state_dict: dict[str, np.ndarray], action_dict: dict[str, np.ndarray]):
    """
    Plot the state and action space side by side.
    
    state_dict: dict[str, np.ndarray] with key: [Time, Dimension]
    action_dict: dict[str, np.ndarray] with key: [Time, Dimension]
    """
    # Calculate total number of plots needed
    total_plots = len(state_dict) + len(action_dict)
    
    # Create a figure with two columns - left for states, right for actions
    fig = plt.figure(figsize=(16, 2 * max(len(state_dict), len(action_dict))))
    
    # Create GridSpec to organize the layout
    gs = fig.add_gridspec(max(len(state_dict), len(action_dict)), 1)
    
    # Plot state and action data together
    # e.g. action.left_arm, state.left_arm, action.right_arm, state.right_arm, etc.
    shared_keys = ["left_arm", "right_arm", "left_hand", "right_hand"]
    
    # Color palette for different dimensions
    colors = plt.cm.tab10.colors
    
    for i, key in enumerate(shared_keys):
        state_key = f"state.{key}"
        action_key = f"action.{key}"

        # plot state and action data on the same plot
        # and label the lines with the key
        ax = fig.add_subplot(gs[i, 0])
        
        # Get the data
        state_data = state_dict[state_key]
        action_data = action_dict[action_key]
        
        # Plot each dimension with a different color
        for dim in range(state_data.shape[1]):
            # State with dashed line
            ax.plot(state_data[:, dim], '--', color=colors[dim % len(colors)], 
                   linewidth=1.5, label=f"state dim {dim}")
            
            # Action with solid line (same color as corresponding state dimension)
            ax.plot(action_data[:, dim], '-', color=colors[dim % len(colors)], 
                   linewidth=2, label=f"action dim {dim}")
        
        ax.set_title(f"{key}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle=':', alpha=0.7)
        
        # Create a more organized legend
        handles, labels = ax.get_legend_handles_labels()
        # Sort the legend so state and action for each dimension are grouped
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    # for i, (key, state_data) in enumerate(state_dict.items()):
    #     ax = fig.add_subplot(gs[i, 0])
    #     ax.plot(state_data)
    #     ax.set_title(f"State: {key}")
    #     ax.set_xlabel("Time")
    #     ax.set_ylabel("Value")
    #     ax.grid(True)
    
    # # Plot action data on the right column
    # for i, (key, action_data) in enumerate(action_dict.items()):
    #     ax = fig.add_subplot(gs[i, 1])
    #     ax.plot(action_data)
    #     ax.set_title(f"Action: {key}")
    #     ax.set_xlabel("Time")
    #     ax.set_ylabel("Value")
    #     ax.grid(True)
    plt.tight_layout()
    # plt.savefig("output_figure1.png")  # Save the plot to a file
    # plt.show()

def plot_image(image: np.ndarray):
    """
    Plot the image.
    """
    # matplotlib show the image
    plt.imshow(image)
    plt.axis('off')
    plt.pause(0.05)  # Non-blocking show
    plt.clf()  # Clear the figure for the next frame


def load_dataset(dataset_path: str, embodiment_tag: str, video_backend: str = "decord", steps: int = 220):
    # 1. get modality keys
    dataset_path = pathlib.Path(dataset_path)
    modality_keys_dict = get_modality_keys(dataset_path)
    video_modality_keys = modality_keys_dict["video"]
    language_modality_keys = modality_keys_dict["annotation"]
    state_modality_keys = modality_keys_dict["state"]
    action_modality_keys = modality_keys_dict["action"]

    pprint(f"Valid modality_keys for debugging:: {modality_keys_dict} \n")

    print(f"state_modality_keys: {state_modality_keys}")
    print(f"action_modality_keys: {action_modality_keys}")
    
    # remove dummy_tensor from state_modality_keys
    state_modality_keys = [key for key in state_modality_keys if key != "state.dummy_tensor"]

    # 2. modality configs
    modality_configs = {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=video_modality_keys,  # we will include all video modalities
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=state_modality_keys,
        ),
        "action": ModalityConfig(
            delta_indices=[0],
            modality_keys=action_modality_keys,
        ),
    }

    # 3. language modality config (if exists)
    if language_modality_keys:
        modality_configs["language"] = ModalityConfig(
            delta_indices=[0],
            modality_keys=language_modality_keys,
        )

    # 4. gr00t embodiment tag
    embodiment_tag: EmbodimentTag = EmbodimentTag(embodiment_tag)

    # 5. load dataset
    dataset = LeRobotSingleDataset(
        dataset_path,
        modality_configs,
        embodiment_tag=embodiment_tag,
        video_backend=video_backend,
    )

    dataset2 = LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_configs,
        embodiment_tag=embodiment_tag,
        video_backend=video_backend,
    )

    #   mixture_kwargs:
    #     mode: train
    #     balance_dataset_weights: true
    #     seed: 42
    #     metadata_config:
    #       merge: true
    #       percentile_mixing_method: min_max
    dataset = LeRobotMixtureDataset(
        data_mixture=[(dataset, 1.0), (dataset2, 1.0)],
        mode="train",
        balance_dataset_weights=True,
        balance_trajectory_weights=True,
        seed=42,
        metadata_config={
            "merge": True,
            "percentile_mixing_method": "min_max",
        },
    )

    print("\n" * 2)
    print("=" * 100)
    print(f"{' Humanoid Dataset ':=^100}")
    print("=" * 100)

    # print the 7th data point
    # resp = dataset[7]
    resp = dataset[0]
    any_describe(resp)
    print(resp.keys())

    print("=" * 50)
    for key, value in resp.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {value}")

    # 6. plot the first 100 images
    images_list = []
    video_key = video_modality_keys[0]  # we will use the first video modality

    state_dict = {
        key: [] for key in state_modality_keys
    }
    action_dict = {
        key: [] for key in action_modality_keys
    }

    SKIP_FRAMES = 3

    for i in range(steps):
        resp = dataset[i]
        if i % SKIP_FRAMES == 0:
            img = resp[video_key][0]

            # cv2 show the image
            # plot_image(img)
            print(f"Image {i}")
            images_list.append(img.copy())

        for state_key in state_modality_keys:
            state_dict[state_key].append(resp[state_key][0])
        for action_key in action_modality_keys:
            action_dict[action_key].append(resp[action_key][0])
        time.sleep(0.05)

    # convert lists of [np[D]] T size to np(T, D)
    for state_key in state_modality_keys:
        state_dict[state_key] = np.array(state_dict[state_key])
    for action_key in action_modality_keys:
        action_dict[action_key] = np.array(action_dict[action_key])

    print(state_dict[state_modality_keys[0]].shape)
    print(action_dict[action_modality_keys[0]].shape)

    if args.plot_state_action:
        plot_state_action_space(state_dict, action_dict)
        print("Plotted state and action space")

    fig, axs = plt.subplots(4, 5, figsize=(20, 10))
    for i, ax in enumerate(axs.flat):
        ax.imshow(images_list[i])
        ax.axis("off")
        ax.set_title(f"Image {i*SKIP_FRAMES}")
    plt.tight_layout()  # adjust the subplots to fit into the figure area.
    # plt.savefig("output_figure2.png")  # Save the plot to a file
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load Robot Dataset")
    parser.add_argument(
        "--data_path",
        type=str,
        default="demo_data/robot_sim.PickNPlace",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--embodiment_tag",
        type=str,
        default="gr1",
        help="Full list of embodiment tags can be found in gr00t.data.schema.EmbodimentTag",
    )
    parser.add_argument(
        "--video_backend",
        type=str,
        default="decord",
        choices=["decord", "torchvision_av"],
        help="Backend to use for video loading, use torchvision_av for av encoded videos",
    )
    parser.add_argument(
        "--plot_state_action",
        action="store_true",
        help="Plot the state and action space",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=150,
        help="Number of steps to plot",
    )
    # parser.add_argument(
    #     "--plot_rerun",
    #     action="store_true",
    #     help="Plot the dataset using rerun",
    # )
    args = parser.parse_args()
    load_dataset(args.data_path, args.embodiment_tag, args.video_backend, args.steps)
