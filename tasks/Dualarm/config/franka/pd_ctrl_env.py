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
import time

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
import matplotlib.pyplot as plt

from icecream import ic
import numpy as np

from franka import FRANKA_PANDA_HIGH_PD_CFG  # isort:skip

pos_errors = []
vel_errors = []


class PD:
    def __init__(self, kp, kd, scene : InteractiveScene , sim : sim_utils.SimulationContext):
        self.kp = -kp
        self.kd = -kd
        self.scene = scene
        self.sim = sim
        self.ball = scene["ball"]
        self.tray = scene["tray"]

    def control(self):
        self.ball_pos = self.ball.data.root_state_w[:,0:3]
        self.tray_pos = self.tray.data.root_state_w[:,0:3]

        self.pos_error = -(self.tray_pos[:] + torch.tensor([0,0,0.01],device=self.sim.device)) + self.ball_pos
        self.vel_error = torch.zeros_like(self.ball.data.root_state_w[:,7:10]) - self.ball.data.root_state_w[:,7:10]
        self.tray_angle = torch.zeros_like(self.pos_error)
        self.tray_angle[0:2] = self.kp * self.pos_error[0:2] + self.kd * self.vel_error[0:2] #* pd control in x and y direction

        # Plot position error
        pos_errors.append(np.linalg.norm(self.pos_error.cpu().detach().numpy()))
        vel_errors.append(np.linalg.norm(self.vel_error.cpu().detach().numpy()))
        
        return self.tray_angle[0] 
       



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
                    size=(0.2,0.2,0.01),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction = 0.5,dynamic_friction=0.5),
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
                    radius=0.005,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=10, solver_velocity_iteration_count=10
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5 , 0.04, 0.21), rot=(1, 0, 0, 0)),
        )

    robot1 = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot1")
    robot2 = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot2")

    robot1.init_state.pos = (0.0, 0.5, 0.0)
    robot2.init_state.pos = (0.0, -0.5, 0.0)


import torch




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

    test_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/test_marker"))

    ee_goals1 = [
        [0.5, 0.5, 0.5, 0.0, 1.0, 0.0, 0.0],
        [0.5000, 0.5000, 0.2500,7.0711e-01, 7.0711e-01, 4.3298e-17,
                           4.3298e-17],
        [0.5, 0.17, 0.225, 7.0711e-01, 7.0711e-01, 4.3298e-17,
                           4.3298e-17],
        [0.5, 0.17, 0.225, 7.0711e-01, 7.0711e-01, 4.3298e-17,
                           4.3298e-17],
        [0.5, 0.17, 0.35, 7.0711e-01, 7.0711e-01, 4.3298e-17,
                           4.3298e-17]    
    ]
    ee_goals2 = [
        [0.5, -0.7, 0.6, 0.707, 0.707, 0.0, 0.0],
        [0.5000, -0.5000,  0.2500,  0.7044, -0.7044,  0.0616,  0.0616],
        [0.5000, -0.23,  0.2,  0.7044, -0.7044,  0.0616,  0.0616],
        [0.5000, -0.23,  0.2,  0.7044, -0.7044,  0.0616,  0.0616],
        [0.5000, -0.23, 0.25,  0.7044, -0.7044,  0.0616,  0.0616]
    ]

    #! diabling second arm
    ee_goals2 = [
        [0.5, -0.7, 0.6, 0.707, 0.707, 0.0, 0.0],
        # [0.5000, -0.5000,  0.2500,  0.7044, -0.7044,  0.0616,  0.0616],
        [0.5, -0.7, 0.6, 0.707, 0.707, 0.0, 0.0],
        # [0.5000, -0.23,  0.2,  0.7044, -0.7044,  0.0616,  0.0616],
        [0.5, -0.7, 0.6, 0.707, 0.707, 0.0, 0.0],
        # [0.5000, -0.23,  0.2,  0.7044, -0.7044,  0.0616,  0.0616],
        [0.5, -0.7, 0.6, 0.707, 0.707, 0.0, 0.0],
        # [0.5000, -0.23, 0.3500,  0.7044, -0.7044,  0.0616,  0.0616]
        [0.5, -0.7, 0.6, 0.707, 0.707, 0.0, 0.0],
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
    RB_1 = True

    pd = PD(0.15,0.02,scene,sim)

    while simulation_app.is_running() and running:

        angle = pd.control()
        angle[0] *= -1
        ic(angle)
        
        if count % 150 == 0:
            if current_goal_idx == 5:
                plt.plot(pos_errors)
                plt.plot(vel_errors)
                plt.show()
                return
            ik_commands1[:] = ee_goals1[current_goal_idx]
            ik_commands2[:] = ee_goals2[current_goal_idx]

            diff_ik_controller1.reset()
            diff_ik_controller1.set_command(ik_commands1)

            diff_ik_controller2.reset()
            diff_ik_controller2.set_command(ik_commands2)
            if current_goal_idx == 3:
                grasp = True
            if grasp:
                robot1.set_joint_position_target(torch.tensor([0.0,0.0],device=sim.device), joint_ids=RB1_GPR_ID)
                robot2.set_joint_position_target(torch.tensor([0.0,0.0],device=sim.device), joint_ids=RB2_GPR_ID)
            else:
                robot1.set_joint_position_target(torch.tensor([0.04,0.04],device=sim.device), joint_ids=RB1_GPR_ID)
                robot2.set_joint_position_target(torch.tensor([0.04,0.04],device=sim.device), joint_ids=RB2_GPR_ID)
            ic(current_goal_idx)
            current_goal_idx += 1

        grasped = (robot1.data.joint_pos[:, RB1_GPR_ID] <= 0.01).all()

        # if grasped :
        #     cur_obj_pose = scene["tray"].data.root_state_w[:,0:7]
        #     obj_to_ee = object_to_ee_transform(cur_obj_pose, robot1.data.body_state_w[:, robot1_entity_cfg.body_ids[0], 0:7])

        #     des_obj_pos = scene["tray"].data.root_state_w[:,0:3]
        #     des_obj_quat = torch.zeros_like(scene["tray"].data.root_state_w[:,3:7])
        #     des_obj_quat[:] = torch.tensor(R.from_euler('xyz', angle.cpu().numpy(), degrees=False).as_quat(),device=sim.device)

        #     des_obj_rot = quaternion_to_rotation_matrix(des_obj_quat)
        #     des_ee_pos = des_obj_pos + torch.bmm(des_obj_rot, obj_to_ee[:,0:3].unsqueeze(-1)).squeeze(-1)
        #     des_ee_quat = quaternion_multiply(des_obj_quat, obj_to_ee[:,3:7])
        #     des_ee_pose = torch.hstack([des_ee_pos, des_ee_quat])
        #     ik_commands1[:] = des_ee_pose
        #     diff_ik_controller1.reset()
        #     diff_ik_controller1.set_command(ik_commands1)
        #     time.sleep(0.5)

        # if grasped:
        cur_obj_pose = scene["tray"].data.root_state_w[:, 0:7]
        des_quat = torch.tensor(R.from_euler('xyz', angle.cpu().numpy(), degrees=False).as_quat(), device=sim.device).repeat(cur_obj_pose.shape[0], 1)
        
        w0, x0, y0, z0 = des_quat[:, 0], des_quat[:, 1], des_quat[:, 2], des_quat[:, 3]
        w1, x1, y1, z1 = cur_obj_pose[:, 3], cur_obj_pose[:, 4], cur_obj_pose[:, 5], cur_obj_pose[:, 6]

        # Quaternion multiplication (batch-wise)
        rot_quat = torch.stack([
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
        ], dim=1)
        

        offset = torch.tensor([0.0, 0.18, 0.0], device=sim.device, dtype=cur_obj_pose.dtype).unsqueeze(0).repeat(cur_obj_pose.shape[0],1,1)

        # Convert rotation matrix to match dtype
        # rot = torch.tensor(
        #     np.linalg.inv(R.from_quat(cur_obj_pose[:, 3:7]).as_matrix()), 
        #     device=sim.device, 
        #     dtype=cur_obj_pose.dtype
        # )
        rot = torch.tensor(
            np.linalg.inv(R.from_quat(rot_quat).as_matrix()),
            device=sim.device,
            dtype=cur_obj_pose.dtype
        )

        # ic(rot.shape)  #? n x 3 x 3
        # ic(offset.shape) #? n x 1 x 3

        # exit()

        des_pos = cur_obj_pose[:, 0:3] - torch.bmm(offset,rot)

        # Get the original quaternion (batch)
        # des_quat = cur_obj_pose[:, 3:7]

        # Define 180-degree rotation about Y-axis quaternion (batch size, 4)
        y_180_quat = torch.tensor([np.sqrt(2)/2, np.sqrt(2)/2, 0.0, 0.0], device=sim.device, dtype=cur_obj_pose.dtype).repeat(des_quat.shape[0], 1)

        # Quaternion multiplication (batch-wise)
        w2, x2, y2, z2 = rot_quat[:, 0], rot_quat[:, 1], rot_quat[:, 2], rot_quat[:, 3]
        w3, x3, y3, z3 = y_180_quat[:, 0], y_180_quat[:, 1], y_180_quat[:, 2], y_180_quat[:, 3]

        des_quat_new = torch.stack([
            w2 * w3 - x2 * x3 - y2 * y3 - z2 * z3,
            w2 * x3 + x2 * w3 + y2 * z3 - z2 * y3,
            w2 * y3 - x2 * z3 + y2 * w3 + z2 * x3,
            w2 * z3 + x2 * y3 - y2 * x3 + z2 * w3
        ], dim=1)

        # Stack new position and orientation
        # ic(des_pos.shape, des_quat_new.shape)
        # exit()
        # des_pose = torch.hstack([des_pos, des_quat_new])

        # Visualize the new pose
        test_marker.visualize(des_pos.squeeze(0), des_quat)

                
        jacobian1 = robot1.root_physx_view.get_jacobians()[:, ee_jacobi_idx1, :, robot1_entity_cfg.joint_ids]
        ee_pose_w1 = robot1.data.body_state_w[:, robot1_entity_cfg.body_ids[0], 0:7]
        root_pose_w1 = robot1.data.root_state_w[:, 0:7]
        root_pose_w1[:, 0:3] -= torch.tensor((0.0, 0.5, -0.01), device=sim.device)
        joint_pos1 = robot1.data.joint_pos[:, robot1_entity_cfg.joint_ids]
        ee_pos_b1, ee_quat_b1 = subtract_frame_transforms(
            root_pose_w1[:, 0:3], root_pose_w1[:, 3:7], ee_pose_w1[:, 0:3], ee_pose_w1[:, 3:7]
        )
        joint_pos_des1 = diff_ik_controller1.compute(ee_pos_b1, ee_quat_b1, jacobian1, joint_pos1)

        jacobian2 = robot2.root_physx_view.get_jacobians()[:, ee_jacobi_idx2, :, robot2_entity_cfg.joint_ids]
        ee_pose_w2 = robot2.data.body_state_w[:, robot2_entity_cfg.body_ids[0], 0:7]
        root_pose_w2 = robot2.data.root_state_w[:, 0:7]
        root_pose_w2[:, 0:3] -= torch.tensor((0.0, -0.5, -0.01), device=sim.device)
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

        error1 = torch.norm(ee_pose_w1[:, 0:3] - ik_commands1[:, 0:3])
        error2 = torch.norm(ee_pose_w2[:, 0:3] - ik_commands2[:, 0:3])

        ic(error1, error2)

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

    

    return 0

if __name__ == "__main__":
    main()
    simulation_app.close()
