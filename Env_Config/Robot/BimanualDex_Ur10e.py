import os
import sys
import numpy as np
import torch
from termcolor import cprint

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles, quat_to_rot_matrix, rot_matrix_to_quat
from isaacsim.core.prims import SingleXFormPrim

sys.path.append(os.getcwd())
from Env_Config.Robot.DexLeft_Ur10e import DexLeft_Ur10e
from Env_Config.Robot.DexRight_Ur10e import DexRight_Ur10e
from Env_Config.Utils_Project.Transforms import quat_diff_rad, Rotation, get_pose_relat, get_pose_world
from Env_Config.Utils_Project.Code_Tools import float_truncate, dense_trajectory_points_generation


class Bimanual_Ur10e:
    def __init__(self, world:World, dexleft_pos, dexleft_ori, dexright_pos, dexright_ori):
        self.world = world
        self.dexleft = DexLeft_Ur10e(world, dexleft_pos, dexleft_ori)
        self.dexright = DexRight_Ur10e(world, dexright_pos, dexright_ori)
        self.dexleft_gripper_state = None
        self.dexright_gripper_state = None
        
        # vis cube (debug tools)
        # self.vis_cube_left = VisualCuboid(
        #     prim_path = "/World/vis_cube_left",
        #     color=np.array([1.0, 0.0, 0.0]),
        #     name = "vis_cube_left", 
        #     position = [0.0, 0.0, 2.0],
        #     size = 0.01,
        #     visible = True,
        # )
        # self.vis_cube_right = VisualCuboid(
        #     prim_path = "/World/vis_cube_right",
        #     color=np.array([1.0, 0.0, 0.0]),
        #     name = "vis_cube_right", 
        #     position = [0.0, 0.0, 2.0],
        #     size = 0.01,
        #     visible = True,
        # )
        # self.world.scene.add(
        #     self.vis_cube_left, 
        # )
        # self.world.scene.add(
        #     self.vis_cube_right, 
        # )
        
    # //////////////////////////////////////////////////////////////
    # /                                                            /
    # /********              Hand Pose Control             ********/
    # /                                                            /
    # //////////////////////////////////////////////////////////////
    
    def set_both_hand_state(self, left_hand_state:str="None", right_hand_state:str="None"):
        # set hand pose according to given hand state ('open' / 'close')
        if left_hand_state == "close":
            left_hand_pose = ArticulationAction(
                joint_positions=np.array([ 
                        7.93413107e-02,-1.50470986e-02 ,2.67736827e-01 ,7.11539682e-01,
                        9.51350867e-01 ,8.78066358e-01,-8.53160175e-02 ,8.53321711e-01,
                        7.87673219e-01 ,7.87673219e-01 ,3.40252007e-02 ,6.76240842e-01,
                        1.12421269e+00 ,8.73928860e-01 ,2.19334288e-01 ,1.76221645e-04,
                        4.14303315e-01 ,1.31339149e+00 ,7.58440614e-01 ,9.76555113e-01,
                        1.10293828e+00 ,5.89046458e-02,-1.27955038e-02 ,5.75099223e-02]),
                joint_indices=np.array(self.dexleft.hand_dof_indices)
            )
            self.dexleft_gripper_state = 0
        elif left_hand_state == "open":
            left_hand_pose = ArticulationAction(
                joint_positions=np.array([
                -0.06150788, -0.07899461,
                -0.34906599, 0.02261488 , 0.01851636,0.01851636,
                -0.08347217, 0.         , 0.00755569,0.00755569 ,
                -0.09509672, 0.00797776, 0.0236096 ,0.0236096   ,
                0.06855482, -0.34906599, 0.04799218 ,0.03465942, 0.03446003 ,
                -0.06890292, 0.63903833, -0.15110777,-0.08261109  ,0.02802108
                ]),
                joint_indices=np.array(self.dexleft.hand_dof_indices)
            )
            self.dexleft_gripper_state = 1
        elif left_hand_state == "pinch":
            left_hand_pose = ArticulationAction(
                joint_positions=np.array([
                    -1.35955052e-01,  5.75466704e-02, -3.49065989e-01,  5.68935978e-01,
                    1.15562234e+00,  7.44316154e-01, -1.70137126e-01,  6.15123597e-02,
                    7.19645386e-02,  7.19645386e-02,  3.24903752e-02,  6.18193961e-19,
                    3.44893551e-02,  3.44893551e-02,  4.06464774e-20, -3.49065989e-01,
                    8.79347072e-02,  5.14625171e-02,  5.14625171e-02,  3.32827328e-01,
                    5.36995551e-01,  7.89638528e-02,  3.2958011e-01,  2.48388920e-01,
                ]),
                joint_indices=np.array(self.dexleft.hand_dof_indices)
            )
            self.dexleft_gripper_state = 2
        elif left_hand_state == "cradle":
            left_hand_pose = ArticulationAction(
                joint_positions=np.array([
                    -0.16060828 , 0.09542108, -0.32950141 , 0.00419018 , 0.06623378 , 0.06622696,
                    -0.09273153 , 0.0146647  , 0.04232529 , 0.04232529, -0.0497151  , 0.01105581,
                    0.07081381 , 0.07081381 , 0.06527984, -0.34906599 , 0.04872768 , 0.05029842,
                    0.05029842 , 0.53537881 , 0.8075118  , 0.19626232 , 0.58037802 , 0.0140745,
                ]),
                joint_indices=np.array(self.dexleft.hand_dof_indices)
            )
            self.dexleft_gripper_state = 3
        elif left_hand_state == "smooth":
            left_hand_pose = ArticulationAction(
                joint_positions=np.array(
                    [-6.67141718e-02,  1.66426553e-01, -2.43158132e-01,  2.33355593e-01,
                    6.73720170e-01,  6.44790593e-02, -1.62202123e-01,  5.12784203e-01,
                    1.65908404e-01,  8.80870383e-02,  1.94555304e-01,  3.39687364e-01,
                    4.71966296e-01,  6.82939946e-21,  1.26723228e-01,  9.82577176e-02,
                    3.40769729e-01,  1.61393453e-01,  5.65633490e-02,  6.14344438e-01,
                    5.08354918e-01, -3.19813694e-02,  5.31338993e-02,  7.06677286e-02,]
                ),
                joint_indices=np.array(self.dexleft.hand_dof_indices)
            )
            self.dexleft_gripper_state = 4

        
        if right_hand_state == "close":
            right_hand_pose = ArticulationAction(
                joint_positions=np.array([ 
                        7.93413107e-02,-1.50470986e-02 ,2.67736827e-01 ,7.11539682e-01,
                        9.51350867e-01 ,8.78066358e-01,-8.53160175e-02 ,8.53321711e-01,
                        7.87673219e-01 ,7.87673219e-01 ,3.40252007e-02 ,6.76240842e-01,
                        1.12421269e+00 ,8.73928860e-01 ,2.19334288e-01 ,1.76221645e-04,
                        4.14303315e-01 ,1.31339149e+00 ,7.58440614e-01 ,9.76555113e-01,
                        1.10293828e+00 ,5.89046458e-02,-1.27955038e-02 ,5.75099223e-02]),
                joint_indices=np.array(self.dexright.hand_dof_indices)
            )
            self.dexright_gripper_state = 0
        elif right_hand_state == "open":
            right_hand_pose = ArticulationAction(
                joint_positions=np.array([
                -0.06150788, -0.07899461,
                -0.34906599, 0.02261488 , 0.01851636,0.01851636,
                -0.08347217, 0.         , 0.00755569,0.00755569 ,
                -0.09509672, 0.00797776, 0.0236096 ,0.0236096   ,
                0.06855482, -0.34906599, 0.04799218 ,0.03465942, 0.03446003 ,
                -0.06890292, 0.63903833, -0.15110777,-0.08261109  ,0.02802108
                ]),
                joint_indices=np.array(self.dexright.hand_dof_indices)
            )
            self.dexright_gripper_state = 1
        elif right_hand_state == "pinch":
            right_hand_pose = ArticulationAction(
                joint_positions=np.array([
                    -1.35955052e-01,  5.75466704e-02, -3.49065989e-01,  5.68935978e-01,
                    1.15562234e+00,  7.44316154e-01, -1.70137126e-01,  6.15123597e-02,
                    7.19645386e-02,  7.19645386e-02,  3.24903752e-02,  6.18193961e-19,
                    3.44893551e-02,  3.44893551e-02,  4.06464774e-20, -3.49065989e-01,
                    8.79347072e-02,  5.14625171e-02,  5.14625171e-02,  3.32827328e-01,
                    5.36995551e-01,  7.89638528e-02,  3.2958011e-01,  2.48388920e-01,
                ]),
                joint_indices=np.array(self.dexright.hand_dof_indices)
            )
            self.dexright_gripper_state = 2
        elif right_hand_state == "cradle":
            right_hand_pose = ArticulationAction(
                joint_positions=np.array([
                    -0.16060828 , 0.09542108, -0.32950141 , 0.00419018 , 0.06623378 , 0.06622696,
                    -0.09273153 , 0.0146647  , 0.04232529 , 0.04232529, -0.0497151  , 0.01105581,
                    0.07081381 , 0.07081381 , 0.06527984, -0.34906599 , 0.04872768 , 0.05029842,
                    0.05029842 , 0.53537881 , 0.8075118  , 0.19626232 , 0.58037802 , 0.0140745,
                ]),
                joint_indices=np.array(self.dexright.hand_dof_indices)
            )
            self.dexright_gripper_state = 3
        elif right_hand_state == "smooth":
            right_hand_pose = ArticulationAction(
                joint_positions=np.array([
                    -6.67141718e-02,  1.66426553e-01, -2.43158132e-01,  2.33355593e-01,
                    6.73720170e-01,  6.44790593e-02, -1.62202123e-01,  5.12784203e-01,
                    1.65908404e-01,  8.80870383e-02,  1.94555304e-01,  3.39687364e-01,
                    4.71966296e-01,  6.82939946e-21,  1.26723228e-01,  9.82577176e-02,
                    3.40769729e-01,  1.61393453e-01,  5.65633490e-02,  6.14344438e-01,
                    5.08354918e-01, -3.19813694e-02,  5.31338993e-02,  7.06677286e-02,
                    ]
                ),
                joint_indices=np.array(self.dexright.hand_dof_indices)
            )
            self.dexright_gripper_state = 4

        # apply action
        if left_hand_state != "None":
            self.dexleft.apply_action(left_hand_pose)
        if right_hand_state != "None":
            self.dexright.apply_action(right_hand_pose)
        
        # wait action to be done
        for i in range(20):
            self.world.step(render=True)
       
    # //////////////////////////////////////////////////////////////
    # /                                                            /
    # /********         Inverse Kinematics Control         ********/
    # /                                                            /
    # //////////////////////////////////////////////////////////////      
            
    def dense_move_both_ik(self, left_pos, left_ori, right_pos, right_ori, angular_type="quat",degree=True, dense_sample_scale:int=0.01):
        '''
        Move DexLeft and DexRight simultaneously once and use dense trajectory to guaranteer smoothness.
        '''
        # debug
        # self.vis_cube_left.set_world_pose(left_pos)
        # self.vis_cube_right.set_world_pose(right_pos)
        assert angular_type in ["quat", "euler"]
        if angular_type == "euler" and left_ori is not None:
            if degree:
                left_ori = euler_angles_to_quat(left_ori,degrees=True)
            else:
                left_ori = euler_angles_to_quat(left_ori)
        if angular_type == "euler" and right_ori is not None:
            if degree:
                right_ori = euler_angles_to_quat(right_ori,degrees=True)
            else:
                right_ori = euler_angles_to_quat(right_ori)
        
        ee_left_pos = left_pos + Rotation(left_ori, np.array([-0.37, -0.025, 0.025]))
        ee_right_pos = right_pos + Rotation(right_ori, np.array([-0.37, -0.025, -0.025]))
        
        ee_left_ori = quat_to_rot_matrix(left_ori)
        ee_right_ori = quat_to_rot_matrix(right_ori)
        
        left_base_pose, left_base_ori = self.dexleft.get_world_pose()
        right_base_pose, right_base_ori = self.dexright.get_world_pose()

        left_base_ori = quat_to_rot_matrix(left_base_ori)
        right_base_ori = quat_to_rot_matrix(right_base_ori)
        
        current_ee_left_pos, current_ee_left_ori = self.dexleft.end_effector.get_world_pose()
        current_ee_right_pos, current_ee_right_ori = self.dexright.end_effector.get_world_pose()
        
        dense_sample_num = int(max(np.linalg.norm(current_ee_left_pos - ee_left_pos), np.linalg.norm(current_ee_right_pos - ee_right_pos)) // dense_sample_scale)

        # print("dense_sample_num", dense_sample_num)
        
        left_interp_pos = dense_trajectory_points_generation(
            start_pos=current_ee_left_pos, 
            end_pos=ee_left_pos,
            num_points=dense_sample_num,
        )
        
        right_interp_pos = dense_trajectory_points_generation(
            start_pos=current_ee_right_pos, 
            end_pos=ee_right_pos,
            num_points=dense_sample_num,
        )
        

        for i in range(dense_sample_num):
            left_pose_in_local, left_ori_in_local = get_pose_relat(left_interp_pos[i], ee_left_ori, left_base_pose, left_base_ori)
            left_ori_in_local = rot_matrix_to_quat(left_ori_in_local)
            
            right_pose_in_local, right_ori_in_local = get_pose_relat(right_interp_pos[i], ee_right_ori, right_base_pose, right_base_ori)
            right_ori_in_local = rot_matrix_to_quat(right_ori_in_local)
            
            left_action, left_succ = self.dexleft.ki_solver.compute_inverse_kinematics(
                left_pose_in_local, 
                left_ori_in_local,
                position_tolerance=0.06
            )
            right_action, right_succ = self.dexright.ki_solver.compute_inverse_kinematics(
                right_pose_in_local, 
                right_ori_in_local,
                position_tolerance=0.06
            )
            
            if left_succ and right_succ:            
                self.dexleft.apply_action(left_action)
                self.dexright.apply_action(right_action)
                self.world.step(render=True)
        
        if left_succ and right_succ:
            print("\033[32mfinish moving!\033[0m")
        else:
            if not left_succ and not right_succ:
                print("\033[31mboth hand failed to move completely!\033[0m")
            elif not left_succ:
                print("\033[31mleft hand failed to move completely!\033[0m")
            elif not right_succ:
                print("\033[31mright hand failed to move completely!\033[0m")
        
        return left_succ and right_succ
    
    def dense_move_both_ik_val(self, left_pos, left_ori, right_pos, right_ori, angular_type="quat",degree=True, dense_sample_scale:int=0.01):
        '''
        For TriDFA validation
        '''
        # debug
        # self.vis_cube_left.set_world_pose(left_pos)
        # self.vis_cube_right.set_world_pose(right_pos)
        assert angular_type in ["quat", "euler"]
        if angular_type == "euler" and left_ori is not None:
            if degree:
                left_ori = euler_angles_to_quat(left_ori,degrees=True)
            else:
                left_ori = euler_angles_to_quat(left_ori)
        if angular_type == "euler" and right_ori is not None:
            if degree:
                right_ori = euler_angles_to_quat(right_ori,degrees=True)
            else:
                right_ori = euler_angles_to_quat(right_ori)
        
        ee_left_pos = left_pos
        ee_right_pos = right_pos
        
        ee_left_ori = quat_to_rot_matrix(left_ori)
        ee_right_ori = quat_to_rot_matrix(right_ori)
        
        left_base_pose, left_base_ori = self.dexleft.get_world_pose()
        right_base_pose, right_base_ori = self.dexright.get_world_pose()

        left_base_ori = quat_to_rot_matrix(left_base_ori)
        right_base_ori = quat_to_rot_matrix(right_base_ori)
        
        current_ee_left_pos, current_ee_left_ori = self.dexleft.end_effector.get_world_pose()
        current_ee_right_pos, current_ee_right_ori = self.dexright.end_effector.get_world_pose()
        
        dense_sample_num = int(max(np.linalg.norm(current_ee_left_pos - ee_left_pos), np.linalg.norm(current_ee_right_pos - ee_right_pos)) // dense_sample_scale)

        # print("dense_sample_num", dense_sample_num)
        
        left_interp_pos = dense_trajectory_points_generation(
            start_pos=current_ee_left_pos, 
            end_pos=ee_left_pos,
            num_points=dense_sample_num,
        )
        
        right_interp_pos = dense_trajectory_points_generation(
            start_pos=current_ee_right_pos, 
            end_pos=ee_right_pos,
            num_points=dense_sample_num,
        )
        

        for i in range(dense_sample_num):
            left_pose_in_local, left_ori_in_local = get_pose_relat(left_interp_pos[i], ee_left_ori, left_base_pose, left_base_ori)
            left_ori_in_local = rot_matrix_to_quat(left_ori_in_local)
            
            right_pose_in_local, right_ori_in_local = get_pose_relat(right_interp_pos[i], ee_right_ori, right_base_pose, right_base_ori)
            right_ori_in_local = rot_matrix_to_quat(right_ori_in_local)
            
            left_action, left_succ = self.dexleft.ki_solver.compute_inverse_kinematics(
                left_pose_in_local, 
                left_ori_in_local,
                position_tolerance=0.06
            )
            right_action, right_succ = self.dexright.ki_solver.compute_inverse_kinematics(
                right_pose_in_local, 
                right_ori_in_local,
                position_tolerance=0.06
            )
            
            if left_succ and right_succ:            
                self.dexleft.apply_action(left_action)
                self.dexright.apply_action(right_action)
                self.world.step(render=True)
        
        if left_succ and right_succ:
            print("\033[32mfinish moving!\033[0m")
        else:
            if not left_succ and not right_succ:
                print("\033[31mboth hand failed to move completely!\033[0m")
            elif not left_succ:
                print("\033[31mleft hand failed to move completely!\033[0m")
            elif not right_succ:
                print("\033[31mright hand failed to move completely!\033[0m")
        
        return left_succ and right_succ
    
    def move_both_with_blocks(self, left_pos, left_ori, right_pos, right_ori,
                            angular_type="quat",degree=None, dense_sample_scale:int=0.01,
                            attach=None, indices=None):
        '''
        Move DexLeft and DexRight simultaneously once and use dense trajectory to guaranteer smoothness.
        '''
        assert angular_type in ["quat", "euler"]
        if angular_type == "euler" and left_ori is not None:
            if degree:
                left_ori = euler_angles_to_quat(left_ori,degrees=True)
            else:
                left_ori = euler_angles_to_quat(left_ori)
        if angular_type == "euler" and right_ori is not None:
            if degree:
                right_ori = euler_angles_to_quat(right_ori,degrees=True)
            else:
                right_ori = euler_angles_to_quat(right_ori)
            
        left_base_pose, _ = self.dexleft.get_world_pose()
        right_base_pose, _ = self.dexright.get_world_pose()
        
        ee_left_pos, ee_left_ori = left_pos + Rotation(left_ori, np.array([-0.35, -0.055, 0.025])), left_ori
        ee_right_pos, ee_right_ori = right_pos + Rotation(right_ori, np.array([-0.35, -0.055, -0.025])), right_ori
        
        current_ee_left_pos, current_ee_left_ori = self.dexleft.end_effector.get_world_pose()
        current_ee_right_pos, current_ee_right_ori = self.dexright.end_effector.get_world_pose()
        
        dense_sample_num = int(max(np.linalg.norm(current_ee_left_pos - ee_left_pos), np.linalg.norm(current_ee_right_pos - ee_right_pos)) // dense_sample_scale)

        # print("dense_sample_num", dense_sample_num)
        
        left_interp_pos = dense_trajectory_points_generation(
            start_pos=current_ee_left_pos, 
            end_pos=ee_left_pos,
            num_points=dense_sample_num,
        )
        
        # print(left_interp_pos)

        right_interp_pos = dense_trajectory_points_generation(
            start_pos=current_ee_right_pos, 
            end_pos=ee_right_pos,
            num_points=dense_sample_num,
        )
        
        # print(right_interp_pos)

        left_finger = SingleXFormPrim("/World/DexLeft/fftip")
        right_finger = SingleXFormPrim("/World/DexRight/fftip")


        for i in range(dense_sample_num):
            left_action, left_succ = self.dexleft.ki_solver.compute_inverse_kinematics(
                left_interp_pos[i] - left_base_pose, 
                ee_left_ori,
                position_tolerance=0.06
            )
            right_action, right_succ = self.dexright.ki_solver.compute_inverse_kinematics(
                right_interp_pos[i] - right_base_pose, 
                ee_right_ori,
                position_tolerance=0.06
            )
            
            if left_succ and right_succ:            
                self.dexleft.apply_action(left_action)
                self.dexright.apply_action(right_action)
                self.world.step(render=True)
                # print(f"sample points {i} finished!")

            if attach is not None and indices is not None:
                block1_pos = left_finger.get_world_pose()[0]
                block0_pos = right_finger.get_world_pose()[0]
                block_pos=[]
                block_pos.append(block0_pos)
                block_pos.append(block1_pos)
                gripper_ori=[]
                gripper_ori.append([1.0, 0.0, 0.0, 0.0])
                gripper_ori.append([1.0, 0.0, 0.0, 0.0])
                for i in indices:
                    attach.block_list[i].set_world_pose(block_pos[i],gripper_ori[i])
                self.world.step(render=True)


        if left_succ and right_succ:
            print("\033[32mfinish moving!\033[0m")
        else:
            if not left_succ and not right_succ:
                print("\033[31mboth hand failed to move completely!\033[0m")
            elif not left_succ:
                print("\033[31mleft hand failed to move completely!\033[0m")
            elif not right_succ:
                print("\033[31mright hand failed to move completely!\033[0m")
        
        return left_succ and right_succ
            
    # //////////////////////////////////////////////////////////////
    # /                                                            /
    # /********        RMPFlow Control (Deprecated)        ********/
    # /                                                            /
    # //////////////////////////////////////////////////////////////
            
    def move_both_rmpflow(self, left_pos:np.ndarray, left_ori:np.ndarray, right_pos:np.ndarray, right_ori:np.ndarray):
        '''
        Move DexLeft and DexRight simultaneously using RMPFlow.
        '''
        while True:
            # get arrive flag
            arrive_left = self.dexleft.check_end_effector_arrive(left_pos)
            arrive_right = self.dexright.check_end_effector_arrive(right_pos)
            # if dexleft not arrive, move it
            if not arrive_left:
                self.dexleft.RMPflow_Move(position=left_pos, orientation=left_ori)
                # if stuck, return False
                if self.dexleft.distance_nochange_epoch >= 100:
                    print("dexleft fail")
                    self.dexright.distance_nochange_epoch = 0
                    self.dexleft.distance_nochange_epoch = 0
                    return False
            # if dexright not arrive, move it
            if not arrive_right:
                self.dexright.RMPflow_Move(position=right_pos, orientation=right_ori)
                # if stuck, return False
                if self.dexright.distance_nochange_epoch >= 100:
                    print("dexright fail")
                    self.dexright.distance_nochange_epoch = 0
                    self.dexleft.distance_nochange_epoch = 0
                    return False
            # if all arrive, clear epoch record and return True
            if arrive_left and arrive_right:
                print("success!")
                self.dexright.distance_nochange_epoch = 0
                self.dexleft.distance_nochange_epoch = 0
                return True