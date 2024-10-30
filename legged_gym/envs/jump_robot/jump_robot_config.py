from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np
from scipy.spatial.transform import Rotation as Rot

D2R = np.pi / 180.0

class JumpRobotCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 48  # 
        num_actions = 12

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False

    class commands(LeggedRobotCfg.commands):
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-2.0, 2.0]


    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 1.3]  # Initial position x, y, z [m]
        # initPos_euler = [0.0, -0.0 * D2R, 0.0]
        # trans = Rot.from_euler('xyz', initPos_euler)
        # initPos_quate = trans.as_quat()
        # rot = [initPos_quate[0], initPos_quate[1], initPos_quate[2], initPos_quate[3]]
        default_joint_angles = {  # Target angles [rad] when action = 0.0
            'hip_x_left': 10.0 * D2R,
            'hip_y_left': 20.0 * D2R, #'hip_left_y': 70.0 * D2R,
            'knee_left': -40.0 * D2R, #'knee_left': -100.0 * D2R,
            'ankle_y_left': 20.0 * D2R, #'ankle_left_y': 46.0 * D2R,
            'ankle_x_left': -10.0 * D2R,
            'shoulder_left': 90.0 * D2R,

            'hip_x_right': 10.0 * D2R,
            'hip_y_right': 20.0 * D2R, #'hip_right_y': 70.0 * D2R,
            'knee_right': -40.0 * D2R, #'knee_right': -100.0 * D2R,
            'ankle_y_right': 20.0 * D2R, #'ankle_right_y': 46.0 * D2R,
            'ankle_x_right': -10.0 * D2R,
            'shoulder_right': 90.0 * D2R,
        }

    class control(LeggedRobotCfg.control):
        stiffness = {
            'hip_x': 150.0, 'hip_y': 150.0, 'knee': 150.0,
            'ankle_y': 50.0, 'ankle_x': 10.0, 'shoulder': 10.0
        }  # [N*m/rad]
        damping = {
            'hip_x': 3.0, 'hip_y': 3.0, 'knee': 3.0,
            'ankle_y': 1.0, 'ankle_x': 0.5, 'shoulder': 0.5
        }  # [N*m*s/rad]
        action_scale = 0.5  # Scale for actions
        decimation = 4  # Number of control action updates per policy update

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/jump_robot/urdf/jump_robot_qhx.urdf'
        name = "jump_robot_qhx"
        foot_name = 'foot'
        terminate_after_contacts_on = ['trunk']
        flip_visual_attachments = False
        self_collisions = 0  # 1 to disable, 0 to enable (bitwise filter)

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 800.
        only_positive_rewards = False

        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -200.
            tracking_ang_vel = 1.0
            torques = -5.e-6
            dof_acc = -2.e-7
            lin_vel_z = 0.01
            feet_air_time = 5.
            dof_pos_limits = -1.
            dof_vel = -0.0
            ang_vel_xy = -0.0
            feet_contact_forces = -0.
            # qhx
            double_fly = 0.3
            double_no_fly = 0.1
            footPosture = -8.e-4
    
    # class viewer(LeggedRobotCfg.viewer):
        # pos = [3, 0, 1]

class JumpRobotCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        run_name = 'jump_robot_training'
        experiment_name = 'jump_robot_experiment'
        max_iterations = 1000

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.02
        learning_rate = 3e-4
        gamma = 0.99
