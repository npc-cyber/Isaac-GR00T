import os
import torch
import gr00t
import numpy as np
import matplotlib.pyplot as plt
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.model.policy import Gr00tPolicy
from gr00t.experiment.data_config import DATA_CONFIG_MAP

from ms.util import *

# Paths and configurations
MODEL_PATH = "nvidia/GR00T-N1-2B"
REPO_PATH = os.path.dirname(os.path.dirname(gr00t.__file__))
DATASET_PATH = os.path.join(REPO_PATH, "demo_data/panda.PickNPlace")
EMBODIMENT_TAG = "new_embodiment"
device = "cuda" if torch.cuda.is_available() else "cpu"

# # Load data config and policy
data_config = DATA_CONFIG_MAP["PandaRobot"]
modality_config = data_config.modality_config()
modality_transform = data_config.transform()

policy = Gr00tPolicy(
    model_path=MODEL_PATH,
    embodiment_tag=EMBODIMENT_TAG,
    modality_config=modality_config,
    modality_transform=modality_transform,
    device=device,
)

print(policy.model)

# Create the dataset
modality_config = policy.modality_config

print(modality_config.keys())

# for key, value in modality_config.items():
#     if isinstance(value, np.ndarray):
#         print(key, value.shape)
#     else:
#         print(key, value)

# # Create the dataset
# dataset = LeRobotSingleDataset(
#     dataset_path=DATASET_PATH,
#     modality_configs=modality_config,
#     video_backend="decord",
#     video_backend_kwargs=None,
#     transforms=None,  # We'll handle transforms separately through the policy
#     embodiment_tag=EMBODIMENT_TAG,
# )

# # Plot joint angles and images
# traj_id = 0
# max_steps = 150
# sample_images = 6

# state_joints_across_time = []
# gt_action_joints_across_time = []
# images = []

# for step_count in range(max_steps):
#     data_point = dataset.get_step_data(traj_id, step_count)
#     state_joints = data_point["state.right_arm"][0]
#     gt_action_joints = data_point["action.right_arm"][0]

#     state_joints_across_time.append(state_joints)
#     gt_action_joints_across_time.append(gt_action_joints)

#     if step_count % (max_steps // sample_images) == 0:
#         image = data_point["video.ego_view"][0]
#         images.append(image)

# # Convert to numpy arrays
# state_joints_across_time = np.array(state_joints_across_time)
# gt_action_joints_across_time = np.array(gt_action_joints_across_time)

# # Plot joint angles
# fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(8, 2 * 7))
# for i, ax in enumerate(axes):
#     ax.plot(state_joints_across_time[:, i], label="state joints")
#     ax.plot(gt_action_joints_across_time[:, i], label="gt action joints")
#     ax.set_title(f"Joint {i}")
#     ax.legend()
# plt.tight_layout()
# plt.show()

# # Plot images
# fig, axes = plt.subplots(nrows=1, ncols=sample_images, figsize=(16, 4))
# for i, ax in enumerate(axes):
#     ax.imshow(images[i])
#     ax.axis("off")
# plt.show()

# # Get predicted action
# step_data = dataset[0]
# predicted_action = policy.get_action(step_data)
# for key, value in predicted_action.items():
#     print(key, value.shape)
