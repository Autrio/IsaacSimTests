# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv
 

def tray_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
) -> torch.Tensor:
    """The position of the tray in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    tray: RigidObject = env.scene[tray_cfg.name]
    tray_pos_w = tray.data.root_pos_w[:, :3]
    tray_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], tray_pos_w
    )
    return tray_pos_b

def ball_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """The position of the ball in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    ball_pos_w = ball.data.root_pos_w[:, :3]
    ball_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], ball_pos_w
    )
    return ball_pos_b

