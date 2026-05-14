import os 
import sys
import torch
import numpy as np

from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim, SingleRigidPrim
from isaacsim.core.api.robots import Robot
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles, quat_to_rot_matrix, rot_matrix_to_quat
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.universal_robots import KinematicsSolver
from isaacsim.robot_motion.motion_generation.lula.motion_policies import RmpFlow, RmpFlowSmoothed
from isaacsim.robot_motion.motion_generation.articulation_motion_policy import ArticulationMotionPolicy

sys.path.append(os.getcwd())
from Env_Config.Utils_Project.Set_Drive import set_drive
from Env_Config.Utils_Project.Transforms import quat_diff_rad, Rotation, get_pose_relat, get_pose_world
from Env_Config.Utils_Project.Code_Tools import float_truncate, dense_trajectory_points_generation

class DexRight_Ur10e(Robot):
    def __init__(self, world:World, translation:np.ndarray, orientation:np.ndarray):
        # define world
        self.world = world
        # define DexLeft name
        self._name = "DexRight"
        # define DexLeft prim
        self._prim_path = "/World/DexRight"
        # get DexLeft usd file path
        self.asset_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../Assets/Robots/assembly/ur10e_shadow/ur10e_shadow_right_hand_glb.usd")
        # define DexLeft positon
        self.translation = translation
        # define DexLeft orientation
        self.orientation = euler_angles_to_quat(orientation,degrees=True)
        
        # add DexLeft USD to stage
        add_reference_to_stage(self.asset_file, self._prim_path)
        # initialize DexLeft Robot according to USD file loaded in stage
        super().__init__(
            prim_path=self._prim_path,
            name=self._name,
            translation=self.translation,
            orientation=self.orientation,
            articulation_controller = None
        )
        # add DexLeft to the scene
        self.world.scene.add(self)

        # check whether pick point is reachable or not
        self.pre_distance = 0
        self.distance_nochange_epoch = 0
        
        # vis cube (debug tools)
        # self.vis_cube = VisualCuboid(
        #     prim_path = "/World/vis_cube_right",
        #     color=np.array([1.0, 0.0, 0.0]),
        #     name = "vis_cube_right", 
        #     position = [0.0, 0.0, 2.0],
        #     size = 0.01,
        #     visible = True,
        # )
        # self.world.scene.add(
        #     self.vis_cube
        # )
        
    def initialize(self, physics_sim_view):
        # initialize robot
        super().initialize(physics_sim_view)
        # reset default status
        self.set_default_state(position=self.translation, orientation=self.orientation)
        self.set_joints_default_state(
            [
                1.57, -1.3, 2.5, -1.25, 1.57, 0.0,
                0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                ]
        )
        # get arm_dof names and arm_dof indices
        self.arm_dof_names = [
            "shoulder_pan_joint", 
            "shoulder_lift_joint", 
            "elbow_joint",
            "wrist_1_joint", 
            "wrist_2_joint", 
            "wrist_3_joint"
        ]
        self.arm_dof_indices = [self.get_dof_index(dof_name) for dof_name in self.arm_dof_names]
        # get hand_dof names and hand_dof indices
        self.hand_dof_names = [
            "WRJ2", "WRJ1", 
            "FFJ4", "FFJ3", "FFJ2", "FFJ1", 
            "MFJ4", "MFJ3", "MFJ2", "MFJ1", 
            "RFJ4", "RFJ3", "RFJ2", "RFJ1", 
            "LFJ5", "LFJ4", "LFJ3", "LFJ2", "LFJ1", 
            "THJ5", "THJ4", "THJ3", "THJ2", "THJ1"
        ]
        self.hand_dof_indices = [self.get_dof_index(dof_name) for dof_name in self.hand_dof_names]

        # kinematics control
        self.ki_solver = KinematicsSolver(self, end_effector_frame_name="ee_link")
        self.end_effector = SingleRigidPrim(self._prim_path + "/ee_link", "end_effector")
        self.end_effector.initialize(physics_sim_view)
        
        self.disable_gravity()
        
        # RMPFlow control
        # self.rmp_config = load_supported_motion_policy_config("UR10e", "RMPflow")
        # self.rmpflow = RmpFlow(**self.rmp_config)
        # self.rmpflow.set_robot_base_pose(self.translation, self.orientation)
        # self.articulation_rmpflow = ArticulationMotionPolicy(self, self.rmpflow, default_physics_dt = 1 / 60.0)
        # self._articulation_controller=self.get_articulation_controller()
        
    def close(self):
        '''
        close hand, custom definition of close pos.
        '''
        close_hand_pose = ArticulationAction(
            joint_positions=np.array([ 
                    7.93413107e-02,-1.50470986e-02 ,2.67736827e-01 ,7.11539682e-01,
                    9.51350867e-01 ,8.78066358e-01,-8.53160175e-02 ,8.53321711e-01,
                    7.87673219e-01 ,7.87673219e-01 ,3.40252007e-02 ,6.76240842e-01,
                    1.12421269e+00 ,8.73928860e-01 ,2.19334288e-01 ,1.76221645e-04,
                    4.14303315e-01 ,1.31339149e+00 ,7.58440614e-01 ,9.76555113e-01,
                    1.10293828e+00 ,5.89046458e-02,-1.27955038e-02 ,5.75099223e-02]),
            joint_indices=np.array(self.hand_dof_indices)
        )
        self.apply_action(close_hand_pose)
        for i in range(30):
            self.world.step(render=True)
        
    def open(self):
        '''
        open hand, custom definition of open pos.
        '''
        open_hand_pose = ArticulationAction(
            joint_positions=np.array([
            -0.06150788, -0.07899461,
            -0.34906599, 0.02261488 , 0.01851636,0.01851636,
            -0.08347217, 0.         , 0.00755569,0.00755569 ,
            -0.09509672, 0.00797776, 0.0236096 ,0.0236096   ,
            0.06855482, -0.34906599, 0.04799218 ,0.03465942, 0.03446003 ,
            -0.06890292, 0.63903833, -0.15110777,-0.08261109  ,0.02802108
            ]),
            joint_indices=np.array(self.hand_dof_indices)
        )
        self.apply_action(open_hand_pose)
        for i in range(30):
            self.world.step(render=True)

    def pinch(self):
        '''
        pinch hand, custom definition of pinch pos.
        '''
        pinch_hand_pose = ArticulationAction(
            joint_positions=np.array([
            -1.35955052e-01,  5.75466704e-02, -3.49065989e-01,  5.68935978e-01,
            1.15562234e+00,  7.44316154e-01, -1.70137126e-01,  6.15123597e-02,
            7.19645386e-02,  7.19645386e-02,  3.24903752e-02,  6.18193961e-19,
            3.44893551e-02,  3.44893551e-02,  4.06464774e-20, -3.49065989e-01,
            8.79347072e-02,  5.14625171e-02,  5.14625171e-02,  3.32827328e-01,
            5.36995551e-01,  7.89638528e-02,  3.2958011e-01,  2.48388920e-01,
            ]),
            joint_indices=np.array(self.hand_dof_indices)
        )
        self.apply_action(pinch_hand_pose)
        for i in range(30):
            self.world.step(render=True)

    def cradle(self):
        '''
        cradle hand, custom definition of cradle pos.
        '''
        cradle_hand_pose = ArticulationAction(
            joint_positions=np.array([
                -0.16060828 , 0.09542108, -0.32950141 , 0.00419018 , 0.06623378 , 0.06622696,
                -0.09273153 , 0.0146647  , 0.04232529 , 0.04232529, -0.0497151  , 0.01105581,
                0.07081381 , 0.07081381 , 0.06527984, -0.34906599 , 0.04872768 , 0.05029842,
                0.05029842 , 0.53537881 , 0.8075118  , 0.19626232 , 0.58037802 , 0.0140745,
            ]),
            joint_indices=np.array(self.hand_dof_indices)
        )
        self.apply_action(cradle_hand_pose)
        for i in range(30):
            self.world.step(render=True)

    def smooth(self):
        '''
        smooth hand, custom definition of smooth pos.
        '''
        smooth_hand_pose = ArticulationAction(
            joint_positions=np.array([
                    -6.67141718e-02,  1.66426553e-01, -2.43158132e-01,  2.33355593e-01,
                    6.73720170e-01,  6.44790593e-02, -1.62202123e-01,  5.12784203e-01,
                    1.65908404e-01,  8.80870383e-02,  1.94555304e-01,  3.39687364e-01,
                    4.71966296e-01,  6.82939946e-21,  1.26723228e-01,  9.82577176e-02,
                    3.40769729e-01,  1.61393453e-01,  5.65633490e-02,  6.14344438e-01,
                    5.08354918e-01, -3.19813694e-02,  5.31338993e-02,  7.06677286e-02,
            ]),
            joint_indices=np.array(self.hand_dof_indices)
        )
        self.apply_action(smooth_hand_pose)
        for i in range(30):
            self.world.step(render=True)

    def get_cur_ee_pos(self):
        '''
        get current end_effector_position and end_effector orientation
        '''
        world_position, world_orientation = self.end_effector.get_world_pose()
        return world_position, world_orientation

    # //////////////////////////////////////////////////////////////
    # /                                                            /
    # /********         Inverse Kinematics Control         ********/
    # /                                                            /
    # //////////////////////////////////////////////////////////////
            
    def step_action(self, target_pos:np.ndarray, target_ori:np.ndarray=None, angular_type:str="quat"):
        '''
        get next action using inverse kinematics.
        if get action successfully, apply aciton.
        '''
        # if 'euler' type, change euler to quartenion
        if angular_type == "euler" and target_ori is not None:
            target_ori = euler_angles_to_quat(target_ori)
        target_pos = target_pos + Rotation(target_ori, np.array([-0.37, -0.025, -0.025]))
        print(target_pos, target_ori)
        target_ori = quat_to_rot_matrix(target_ori)
        # get current world pose
        base_pose, base_ori = self.get_world_pose()
        base_ori = quat_to_rot_matrix(base_ori)
        pose_in_local, ori_in_local = get_pose_relat(target_pos, target_ori, base_pose, base_ori)
        ori_in_local = rot_matrix_to_quat(ori_in_local)
        # get action
        action, succ = self.ki_solver.compute_inverse_kinematics(pose_in_local, ori_in_local, position_tolerance=0.06)
        # if get action successfully, apply action
        if succ:
            self._articulation_controller.apply_action(action)
            current_pos, _ = self.end_effector.get_world_pose()
            pre = np.linalg.norm(target_pos-current_pos)
            while True:
                self.world.step(render=True)
                current_pos, _ = self.end_effector.get_world_pose()
                cur = np.linalg.norm(target_pos-current_pos)
                if float_truncate(cur) == float_truncate(pre):
                    break
                pre = cur
            print("finish moving!")
        return succ
    
    def dense_step_action(self, target_pos:np.ndarray, target_ori:np.ndarray=None, angular_type:str="quat", dense_sample_scale:float=0.01):
        '''
        get next action using inverse kinematics.
        if get action successfully, apply aciton.
        use dense trajectory planning to make the robot move smoothly and synchronously.
        '''
        assert angular_type in ["euler", "quat"], "angular_type must be 'euler' or 'quat'"
        # if 'euler' type, change euler to quartenion
        if angular_type == "euler" and target_ori is not None:
            target_ori = euler_angles_to_quat(target_ori)
        # transfer to the wrist pos and rot matrix
        target_pos = target_pos + Rotation(target_ori, np.array([-0.37, -0.025, -0.025]))
        target_ori = quat_to_rot_matrix(target_ori)
        # get current robot base pose
        base_pose, base_ori = self.get_world_pose()
        base_ori = quat_to_rot_matrix(base_ori)   
        # get current end effector pose
        current_ee_pos, _ = self.get_cur_ee_pos()         
        # dense sample trajectory
        dense_sample_num = int(np.linalg.norm(target_pos - current_ee_pos) // dense_sample_scale)
        # print("dense_sample_num : ", dense_sample_num)
        interp_pos = dense_trajectory_points_generation(
            start_pos=current_ee_pos, 
            end_pos=target_pos,
            num_points=dense_sample_num,
        )
        for i in range(len(interp_pos)):
            # transfer target pose to robot base coordinate
            pose_in_local, ori_in_local = get_pose_relat(interp_pos[i], target_ori, base_pose, base_ori)
            ori_in_local = rot_matrix_to_quat(ori_in_local)
            # get action
            action, succ = self.ki_solver.compute_inverse_kinematics(
                pose_in_local, 
                ori_in_local,
                position_tolerance=0.06
            )
            if succ:
                self._articulation_controller.apply_action(action)
                self.world.step(render=True)
                # print(f"sample points {i} finished!")

        if succ:
            print("\033[32mfinish moving!\033[0m")
        else:
            print("\033[31mleft hand failed to move completely!\033[0m")
        
        return succ
        
    # //////////////////////////////////////////////////////////////
    # /                                                            /
    # /********        RMPFlow Control (Deprecated)        ********/
    # /                                                            /
    # //////////////////////////////////////////////////////////////
            
    def add_obstacle(self, obstacle):
        '''
        add obstacle to franka motion
        make franka avoid potential collision smartly
        '''
        print("add obstacle : ", self.rmpflow.add_obstacle(obstacle, False))
        for i in range(10):
            self.world.step(render=True)
        return
    
    def check_end_effector_arrive(self, target_position=None)->bool:
        '''
        check whether end_effector has arrived at the target position
        if arrived, return True; else return False.
        '''
        # get current position and calculate the distance between current position and target position
        ee_position, ee_orientation = self.get_cur_ee_pos()
        current_position, current_orientation = ee_position + Rotation(ee_orientation, np.array([0.025, -0.055, 0.35])), ee_orientation
        distance = torch.dist(torch.tensor(current_position, dtype=float), torch.tensor(target_position, dtype=float))
        # print(distance.item())
        self.vis_cube.set_world_pose(current_position, current_orientation)
        # get distance_nochange_epoch according to distance_gap
        distance_gap = distance - self.pre_distance
        self.pre_distance = distance
        if abs(distance_gap) < 5e-6:
            self.distance_nochange_epoch += 1
            # print(self.distance_nochange_epoch)
        # return result
        if distance < 0.02 and abs(distance_gap) < 5e-4:
            self.distance_nochange_epoch = 0
            return True
        else:
            return False
        
    def RMPflow_Move(self, position, orientation=None):
        '''
        Use RMPflow_controller to move the Ur10e.
        '''
        self.world.step(render=True)
        # obtain ee_orientation
        if orientation is not None:
            ee_orientation = euler_angles_to_quat(orientation, degrees=True)
        # obtain ee_position
        ee_position = position + Rotation(ee_orientation, np.array([-0.025, 0.055, -0.35]))
        # set end effector target
        self.rmpflow.set_end_effector_target(
            target_position=ee_position, 
            target_orientation=ee_orientation,
        )
        # update obstacle position 
        self.rmpflow.update_world()
        # get target articulation action
        actions = self.articulation_rmpflow.get_next_articulation_action()
        # apply actions
        self._articulation_controller.apply_action(actions)
               