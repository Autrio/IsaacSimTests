import argparse

from omni.isaac.lab.app import AppLauncher #type: ignore

parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import omni.isaac.lab.sim as sim_utils #type: ignore
from omni.isaac.lab.assets import AssetBaseCfg #type: ignore
from omni.isaac.lab.assets import RigidObjectCfg, RigidObjectCollectionCfg #type: ignore
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg #type: ignore
from omni.isaac.lab.managers import SceneEntityCfg #type: ignore
from omni.isaac.lab.markers import VisualizationMarkers #type: ignore
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG #type: ignore
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg #type: ignore
from omni.isaac.lab.utils import configclass #type: ignore
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR #type: ignore
from omni.isaac.lab.utils.math import subtract_frame_transforms #type: ignore
from scipy.spatial.transform import Rotation as R #type: ignore

from icecream import ic

from franka import FRANKA_PANDA_HIGH_PD_CFG  # isort:skip

import pygame
pygame.init()

screen = pygame.display.set_mode((300, 300))
pygame.display.set_caption("Keyboard Input for Simulation")

@configclass
class TableTopSceneCfg(InteractiveSceneCfg):

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    tray = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/tray",
            spawn=sim_utils.CuboidCfg(
                    size=(0.3,0.3,0.01),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=10, solver_velocity_iteration_count=10
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0, 0.201), rot=(1, 0, 0, 0)),
        )

    base = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/base",
            spawn=sim_utils.CuboidCfg(
                    size=(0.1,0.1,0.2),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=10, solver_velocity_iteration_count=10
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0, 0.1), rot=(1, 0, 0, 0)),
        )
    
    ball = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/ball",
            spawn=sim_utils.SphereCfg(
                    radius=0.01,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=10, solver_velocity_iteration_count=10
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0, 0.21), rot=(1, 0, 0, 0)),
        )

    robot1 = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot1")
    robot2 = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot2")

    robot1.init_state.pos = (0.0, 0.6, 0.0)
    robot2.init_state.pos = (0.0, -0.6, 0.0)

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):

    robot1 = scene["robot1"]
    robot2 = scene["robot2"]

    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller1 = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)
    diff_ik_controller2 = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    ee_marker1 = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current1"))
    goal_marker1 = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal1"))

    ee_marker2 = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current2"))
    goal_marker2 = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal2"))

    ee_goals1 = [
        [0.5, 0.5, 0.5, 0.0, 1.0, 0.0, 0.0],
        [0.5000, 0.5000, 0.2500, 0.7071, 0.7071, 0.0000, 0.0000],
        [0.5, 0.21, 0.25, 0.707, 0.707, 0.0, 0.0],    
    ]
    ee_goals2 = [
        [0.5, -0.7, 0.6, 0.707, 0.707, 0.0, 0.0],
        [0.5, -0.6, 0.7, 1, 0, 0, 0],
        [0.5, -0.5, 0.5, 0.0, 1.0, 0.0, 0.0],
    ]
    ee_goals1 = torch.tensor(ee_goals1, device=sim.device)
    ee_goals2 = torch.tensor(ee_goals2, device=sim.device)

    current_goal_idx = 0
    ik_commands1 = torch.zeros(scene.num_envs, diff_ik_controller1.action_dim, device=robot1.device)
    ik_commands1[:] = ee_goals1[current_goal_idx]

    ik_commands2 = torch.zeros(scene.num_envs, diff_ik_controller2.action_dim, device=robot2.device)
    ik_commands2[:] = ee_goals2[current_goal_idx]

    robot1_entity_cfg = SceneEntityCfg("robot1", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    robot2_entity_cfg = SceneEntityCfg("robot2", joint_names=["panda_joint.*"], body_names=["panda_hand"])

    gripper1_entity_cfg = SceneEntityCfg("robot1", joint_names=["panda_finger_joint.*"], body_names=["panda_.*finger"])
    gripper2_entity_cfg = SceneEntityCfg("robot2", joint_names=["panda_finger_joint.*"], body_names=["panda_.*finger"])

    robot1_entity_cfg.resolve(scene)
    robot2_entity_cfg.resolve(scene)
    gripper1_entity_cfg.resolve(scene)
    gripper2_entity_cfg.resolve(scene)

    ic(robot1.joint_names)
    ic(robot1_entity_cfg.joint_ids)
    ic(gripper1_entity_cfg.joint_ids)
    ic(robot2.joint_names)
    ic(robot2_entity_cfg.joint_ids)
    ic(gripper2_entity_cfg.joint_ids)

    if robot1.is_fixed_base:
        ee_jacobi_idx1 = robot1_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx1 = robot1_entity_cfg.body_ids[0]

    if robot2.is_fixed_base:
        ee_jacobi_idx2 = robot2_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx2 = robot2_entity_cfg.body_ids[0]

    sim_dt = sim.get_physics_dt()
    count = 0

    joint_pos1 = robot1.data.default_joint_pos.clone()
    joint_vel1 = robot1.data.default_joint_vel.clone()
    robot1.write_joint_state_to_sim(joint_pos1, joint_vel1)
    robot1.reset()

    joint_pos2 = robot2.data.default_joint_pos.clone()
    joint_vel2 = robot2.data.default_joint_vel.clone()
    robot2.write_joint_state_to_sim(joint_pos2, joint_vel2)
    robot2.reset()

    joint_pos_des1 = joint_pos1[:, robot1_entity_cfg.joint_ids].clone()
    joint_pos_des2 = joint_pos2[:, robot2_entity_cfg.joint_ids].clone()

    RB1_GPR_ID = gripper1_entity_cfg.joint_ids
    RB2_GPR_ID = gripper2_entity_cfg.joint_ids
    
    rot_deg = [0,0,0]
    grasp = False
    running = True
    while simulation_app.is_running() and running:
        event = pygame.event.poll()
        if event is not pygame.NOEVENT:
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    joint_pos1 = robot1.data.default_joint_pos.clone()
                    joint_vel1 = robot1.data.default_joint_vel.clone()
                    robot1.write_joint_state_to_sim(joint_pos1, joint_vel1)
                    robot1.reset()

                    joint_pos2 = robot2.data.default_joint_pos.clone()
                    joint_vel2 = robot2.data.default_joint_vel.clone()
                    robot2.write_joint_state_to_sim(joint_pos2, joint_vel2)
                    robot2.reset()

                if event.key == pygame.K_c:
                    RB_ID = 2

                if event.key == pygame.K_w:
                    ik_commands1[:,0] -= 0.01
                if event.key == pygame.K_s:
                    ik_commands1[:,0] += 0.01
                if event.key == pygame.K_a:
                    ik_commands1[:,1] -= 0.01
                if event.key == pygame.K_d:
                    ik_commands1[:,1] += 0.01
                if event.key == pygame.K_e:
                    ik_commands1[:,2] -= 0.01
                if event.key == pygame.K_q:
                    ik_commands1[:,2] += 0.01

                if event.key == pygame.K_UP:
                    rot_deg[0] += 10
                if event.key == pygame.K_DOWN:
                    rot_deg[0] -= 10
                if event.key == pygame.K_LEFT:
                    rot_deg[1] += 10
                if event.key == pygame.K_RIGHT:
                    rot_deg[1] -= 10
                if event.key == pygame.K_PAGEUP:
                    rot_deg[2] += 10
                if event.key == pygame.K_PAGEDOWN:
                    rot_deg[2] -= 10
                
                quat = R.from_euler('xyz', rot_deg, degrees=True).as_quat()
                ik_commands1[:,3:7] = torch.tensor(quat)

                if event.key == pygame.K_ESCAPE:
                    exit(0)
                
                if event.key == pygame.K_0:
                    ik_commands1[:] = ee_goals1[0]
                    ik_commands2[:] = ee_goals2[0]
                    quat = ik_commands1[0,3:7].cpu().numpy()
                    rot_deg = R.from_quat(quat).as_euler('xyz', degrees=True)
                    
                if event.key == pygame.K_1:
                    ik_commands1[:] = ee_goals1[1]
                    ik_commands2[:] = ee_goals2[1]
                    quat = ik_commands1[0,3:7].cpu().numpy()
                    rot_deg = R.from_quat(quat).as_euler('xyz', degrees=True)

                if event.key == pygame.K_2:
                    ik_commands1[:] = ee_goals1[2]
                    ik_commands2[:] = ee_goals2[2]
                    quat = ik_commands1[0,3:7].cpu().numpy()
                    rot_deg = R.from_quat(quat).as_euler('xyz', degrees=True)

                if event.key == pygame.K_SPACE:
                    grasp = not grasp
                
                ic(ik_commands1)
                ic(ik_commands2)
                ic(rot_deg)    

                joint_pos_des1 = joint_pos1[:, robot1_entity_cfg.joint_ids].clone()
                joint_pos_des2 = joint_pos2[:, robot2_entity_cfg.joint_ids].clone()
                diff_ik_controller1.reset()
                diff_ik_controller1.set_command(ik_commands1)

                diff_ik_controller2.reset()
                diff_ik_controller2.set_command(ik_commands2)

            if grasp:
                robot1.set_joint_position_target(torch.tensor([0.0,0.0],device=sim.device), joint_ids=RB1_GPR_ID)
                robot2.set_joint_position_target(torch.tensor([0.0,0.0],device=sim.device), joint_ids=RB2_GPR_ID)
            else:
                robot1.set_joint_position_target(torch.tensor([0.04,0.04],device=sim.device), joint_ids=RB1_GPR_ID)
                robot2.set_joint_position_target(torch.tensor([0.04,0.04],device=sim.device), joint_ids=RB2_GPR_ID)
            
        jacobian1 = robot1.root_physx_view.get_jacobians()[:, ee_jacobi_idx1, :, robot1_entity_cfg.joint_ids]
        ee_pose_w1 = robot1.data.body_state_w[:, robot1_entity_cfg.body_ids[0], 0:7]
        root_pose_w1 = robot1.data.root_state_w[:, 0:7]
        root_pose_w1[:, 0:3] -= torch.tensor((0.0, 0.6, 0.0), device=sim.device)
        joint_pos1 = robot1.data.joint_pos[:, robot1_entity_cfg.joint_ids]
        ee_pos_b1, ee_quat_b1 = subtract_frame_transforms(
            root_pose_w1[:, 0:3], root_pose_w1[:, 3:7], ee_pose_w1[:, 0:3], ee_pose_w1[:, 3:7]
        )
        joint_pos_des1 = diff_ik_controller1.compute(ee_pos_b1, ee_quat_b1, jacobian1, joint_pos1)

        jacobian2 = robot2.root_physx_view.get_jacobians()[:, ee_jacobi_idx2, :, robot2_entity_cfg.joint_ids]
        ee_pose_w2 = robot2.data.body_state_w[:, robot2_entity_cfg.body_ids[0], 0:7]
        root_pose_w2 = robot2.data.root_state_w[:, 0:7]
        root_pose_w2[:, 0:3] -= torch.tensor((0.0, -0.6, 0.0), device=sim.device)
        joint_pos2 = robot2.data.joint_pos[:, robot2_entity_cfg.joint_ids]
        ee_pos_b2, ee_quat_b2 = subtract_frame_transforms(
            root_pose_w2[:, 0:3], root_pose_w2[:, 3:7], ee_pose_w2[:, 0:3], ee_pose_w2[:, 3:7]
        )
        joint_pos_des2 = diff_ik_controller2.compute(ee_pos_b2, ee_quat_b2, jacobian2, joint_pos2)

        robot1.set_joint_position_target(joint_pos_des1, joint_ids=robot1_entity_cfg.joint_ids)
        robot2.set_joint_position_target(joint_pos_des2, joint_ids=robot2_entity_cfg.joint_ids)
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)

        ee_pose_w1 = robot1.data.body_state_w[:, robot1_entity_cfg.body_ids[0], 0:7]
        ee_pose_w2 = robot2.data.body_state_w[:, robot2_entity_cfg.body_ids[0], 0:7]
        ee_marker1.visualize(ee_pose_w1[:, 0:3], ee_pose_w1[:, 3:7])
        goal_marker1.visualize(ik_commands1[:, 0:3] + scene.env_origins, ik_commands1[:, 3:7])

        ee_marker2.visualize(ee_pose_w2[:, 0:3], ee_pose_w2[:, 3:7])
        goal_marker2.visualize(ik_commands2[:, 0:3] + scene.env_origins, ik_commands2[:, 3:7])

def main():

    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    print("+"*100)
    print(f'Simulation Running on Device: {sim.device}')
    print("+"*100)

    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)

if __name__ == "__main__":
    main()
    simulation_app.close()
