import time
from pathlib import Path

import mink
import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd
from loop_rate_limiters import RateLimiter
from scipy.spatial.transform import Rotation as R

_HERE = Path(__file__).parent.parent.parent.parent
_XML = _HERE / "robot" / "openloong" /"scene_mink.xml"


# FIXME
# bvh file path
file_name = "goose-step184_Skeleton0.bvh"

bvh_file_path = _HERE / "datas" / "mocap" / "goose-step" / file_name 
# save path
csv_file = _HERE / "datas" / "retargeted" / "goose-step" / f"{file_name[:-4]}.csv"

# 上下身比例
up_scal = 1.15
leg_scal = 1.05


class HierarchyParser(object):

    def __init__(self, bvh_file_path):
        self.lines = self.get_hierarchy_lines(bvh_file_path)
        self.line_number = 0

        self.root_position_channels = []
        self.joint_rotation_channels = []

        self.joint_names = []
        self.joint_parents = []
        self.joint_offsets = []

    def get_hierarchy_lines(self, bvh_file_path):
        hierarchy_lines = []
        for line in open(bvh_file_path, 'r'):
            line = line.strip()
            if line.startswith('MOTION'):
                break
            else:
                hierarchy_lines.append(line)

        return hierarchy_lines
    
    def parse_offset(self, line):
        return [float(x) for x in line.split()[1:]]
    
    def parse_channels(self, line):
        return [x for x in line.split()[2:]]

    def parse_root(self, parent=-1):
        self.joint_parents.append(parent)

        self.joint_names.append(self.lines[self.line_number].split()[1])
        self.line_number += 2

        if self.lines[self.line_number].startswith('OFFSET'):
            self.joint_offsets.append(self.parse_offset(self.lines[self.line_number]))
        else:
            print('cannot find root offset')
        self.line_number += 1

        if self.lines[self.line_number].startswith('CHANNELS'):
            channels = self.parse_channels(self.lines[self.line_number])
            if self.lines[self.line_number].split()[1] == '3':
                self.joint_rotation_channels.append((channels[0], channels[1], channels[2]))
            elif self.lines[self.line_number].split()[1] == '6':
                self.root_position_channels.append((channels[0], channels[1], channels[2]))
                self.joint_rotation_channels.append((channels[3], channels[4], channels[5]))
        else:
            print('cannot find root channels')
        self.line_number += 1

        while self.lines[self.line_number].startswith('JOINT'):
            self.parse_joint(0)
        self.line_number += 1

    def parse_joint(self, parent):
        self.joint_parents.append(parent)

        index = len(self.joint_names)
        self.joint_names.append(self.lines[self.line_number].split()[1])
        self.line_number += 2

        if self.lines[self.line_number].startswith('OFFSET'):
            self.joint_offsets.append(self.parse_offset(self.lines[self.line_number]))
        else:
            print('cannot find joint offset')
        self.line_number += 1

        if self.lines[self.line_number].startswith('CHANNELS'):
            channels = self.parse_channels(self.lines[self.line_number])
            if self.lines[self.line_number].split()[1] == '3':
                self.joint_rotation_channels.append((channels[0], channels[1], channels[2]))
        else:
            print('cannot find joint channels')
        self.line_number += 1

        while self.lines[self.line_number].startswith('JOINT') or \
                self.lines[self.line_number].startswith('End'):
            if self.lines[self.line_number].startswith('JOINT'):
                self.parse_joint(index)
            elif self.lines[self.line_number].startswith('End'):
                self.parse_end(index)
        self.line_number += 1

    def parse_end(self, parent):
        self.joint_parents.append(parent)

        self.joint_names.append(self.joint_names[parent] + '_end')
        self.line_number += 2

        if self.lines[self.line_number].startswith('OFFSET'):
            self.joint_offsets.append(self.parse_offset(self.lines[self.line_number]))
        else:
            print('cannot find joint offset')
        self.line_number += 2

    def analyze(self):
        if not self.lines[self.line_number].startswith('HIERARCHY'):
            print('cannot find hierarchy')
        self.line_number += 1

        if self.lines[self.line_number].startswith('ROOT'):
            self.parse_root()
    
        return self.joint_names, self.joint_parents, self.joint_offsets


def forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    m = len(joint_name)
    joint_positions = np.zeros((m, 3), dtype=np.float64)
    joint_orientations = np.zeros((m, 4), dtype=np.float64)
    channels = motion_data[frame_id]
    rotations = np.zeros((m, 3), dtype=np.float64)
    cnt = 1

    for i in range(m):
        if '_end' not in joint_name[i]:
            for j in range(3):
                rotations[i][j] = channels[cnt * 3 + j]
            cnt += 1
    for i in range(m):
        parent = joint_parent[i]
        if parent == -1:
            for j in range(3):
                joint_positions[0][j] = channels[j]
            joint_orientations[0] = R.from_euler('YXZ', [rotations[0][0], rotations[0][1], rotations[0][2]], degrees=True).as_quat()
            
        else:
            if '_end' in joint_name[i]:
                joint_orientations[i] = np.array([0, 0, 0, 1])                
                joint_positions[i] = joint_positions[parent] + R.from_quat(joint_orientations[parent]).as_matrix() @ joint_offset[i]
            else:
                rotation = R.from_euler('YXZ', [rotations[i][0], rotations[i][1], rotations[i][2]], degrees=True)                
                joint_orientations[i] = (R.from_quat(joint_orientations[parent]) * rotation).as_quat()
                joint_positions[i] = joint_positions[parent] + R.from_quat(joint_orientations[parent]).as_matrix() @ joint_offset[i]


    return joint_positions, joint_orientations


def load_motion_data(bvh_file_path):
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data

def convert_bvh_to_mujoco(bvh_coords):
    # 创建一个新的数组，存储转换后的坐标
    mujoco_coords = np.zeros_like(bvh_coords)
    mujoco_coords1 = np.zeros_like(bvh_coords)
    # 转换坐标系
    mujoco_coords[:, 0] = -bvh_coords[:, 0]  
    mujoco_coords[:, 1] = bvh_coords[:, 2]  
    mujoco_coords[:, 2] = bvh_coords[:, 1] 

    mujoco_coords1[:, 0] = mujoco_coords[:, 1]  
    mujoco_coords1[:, 1] = -mujoco_coords[:, 0]  
    mujoco_coords1[:, 2] = mujoco_coords[:, 2] 

    return mujoco_coords1

def convert_quaternion(bvh_quaternions):

    """ 转换关节姿态四元数从BVH到MuJoCo """
    mujoco_quaternions = np.zeros_like(bvh_quaternions)
    mujoco_quaternions1 = np.zeros_like(bvh_quaternions)
    mujoco_quaternions[:, 0] = bvh_quaternions[:, 0]  # w
    mujoco_quaternions[:, 1] = -bvh_quaternions[:, 1]  # x
    mujoco_quaternions[:, 2] = bvh_quaternions[:, 3]  # z
    mujoco_quaternions[:, 3] = bvh_quaternions[:, 2]  # -y

    mujoco_quaternions1[:, 0] = mujoco_quaternions[:, 0]  # w
    mujoco_quaternions1[:, 1] = mujoco_quaternions[:, 2]  # x
    mujoco_quaternions1[:, 2] = -mujoco_quaternions[:, 1]  # z
    mujoco_quaternions1[:, 3] = mujoco_quaternions[:, 3]  # -y

    return mujoco_quaternions1






if __name__ == "__main__":
    # viewer = SimpleViewer()
    parser = HierarchyParser(bvh_file_path)
    joint_names, joint_parents, joint_offsets = parser.analyze()
    motion_data = load_motion_data(bvh_file_path)
    frame_num = motion_data.shape[0]
    print(f'data length is {frame_num}, processing!')
    current_frame = 0

    joint_positions = []
    joint_orientations = []
    R_extra = np.array([[0, 1, 0],
                [-1, 0, 0],
                [0, 0, 1]]) 

    for current_frame in range(frame_num):                     
        joint_position, joint_orientation = forward_kinematics(joint_names, \
                    joint_parents, joint_offsets, motion_data, current_frame)  
        joint_positions.append(joint_position)
        joint_orientations.append(joint_orientation[:,[3,0,1,2]])
      
    
    joint_positions = np.array(joint_positions)
    joint_orientations = np.array(joint_orientations)
    joint_positions *= 0.01
    


    # print(joint_positions[:,0,:])
    # print('ori',joint_orientations[:,0,:])
    

    
    # Read some basic data from the file
    # frame_rate = mvnx_file.frame_rate
    # print(frame_rate)

    com_array = convert_bvh_to_mujoco(joint_positions[:,1,:])
    com_orien_array = convert_quaternion(joint_orientations[:,1,:])
    # com_array[:, 2] *= 1.12
    # print(com_array)

    # L5
    # data_need_quat[1, :]
    # #L3
    waist_array = convert_bvh_to_mujoco(joint_positions[:,3,:])
    waist_orien_array = convert_quaternion(joint_orientations[:,3,:])
    # waist_array[:, 2] *= up_scal
    
    # #T12
    # data_need_quat[3, :] 
    # #T8
    # data_need_quat[4, :] 
    #neck
    neck_array = convert_bvh_to_mujoco(joint_positions[:,5,:])
    neck_orien_array = convert_quaternion(joint_orientations[:,5,:])
    neck_array[:, 2] *= up_scal
    #head
    head_array = convert_bvh_to_mujoco(joint_positions[:,6,:])
    head_orien_array = convert_quaternion(joint_orientations[:,6,:])
    head_array[:, 2] *= up_scal
    

    # 右肩
    rightsuogu_array = convert_bvh_to_mujoco(joint_positions[:,13,:]) 
    rightsuogu_orien_array = convert_quaternion(joint_orientations[:,13,:])
    rightsuogu_array[:, 2] *= up_scal
    
    # 右大臂
    rightshoulder_array = convert_bvh_to_mujoco(joint_positions[:,14,:]) 
    rightshoulder_orien_array = convert_quaternion(joint_orientations[:,14,:])
    rightshoulder_array[:, 2] *= up_scal
    # 右前臂
    rightelbow_array = convert_bvh_to_mujoco(joint_positions[:,15,:])
    rightelbow_orien_array = convert_quaternion(joint_orientations[:,15,:])
    rightelbow_array[:, 2] *= up_scal
    # 右手 叫555
    righthand_array = convert_bvh_to_mujoco(joint_positions[:,16,:])
    righthand_orien_array = convert_quaternion(joint_orientations[:,16,:])
    righthand_array[:, 2] *= up_scal
    #右手end
    righthandend_array = convert_bvh_to_mujoco(joint_positions[:,17,:])
    righthandend_orien_array = convert_quaternion(joint_orientations[:,17,:])
    righthandend_array[:, 2] *= up_scal
    # 左肩
    leftsuogu_array = convert_bvh_to_mujoco(joint_positions[:,8,:])
    leftsuogu_orien_array = convert_quaternion(joint_orientations[:,8,:])
    leftsuogu_array[:, 2] *= up_scal
    # 左大臂
    leftshoulder_array = convert_bvh_to_mujoco(joint_positions[:,9,:])
    leftshoulder_orien_array = convert_quaternion(joint_orientations[:,9,:])
    leftshoulder_array[:, 2] *= up_scal
    # 左前臂 
    leftelbow_array = convert_bvh_to_mujoco(joint_positions[:,10,:]) 
    leftelbow_orien_array = convert_quaternion(joint_orientations[:,10,:])
    leftelbow_array[:, 2] *= up_scal
    # 左手
    lefthand_array = convert_bvh_to_mujoco(joint_positions[:,11,:])
    lefthand_orien_array =  convert_quaternion(joint_orientations[:,11,:])
    lefthand_array[:, 2] *= up_scal
    #左手end
    lefthandend_array = convert_bvh_to_mujoco(joint_positions[:,12,:])
    lefthandend_orien_array =  convert_quaternion(joint_orientations[:,12,:])
    lefthandend_array[:, 2] *= up_scal
   
    # lefthand_orien_array = quaternion_multiply(q_y, data_need_quat[14, 3:7])
    #右上腿
    righthip_array = convert_bvh_to_mujoco(joint_positions[:,23,:])
    righthip_orien_array = convert_quaternion(joint_orientations[:,23,:])
    righthip_array[:,2] *= leg_scal
    #右下腿
    rightknee_array = convert_bvh_to_mujoco(joint_positions[:,24,:])
    rightknee_orien_array = convert_quaternion(joint_orientations[:,24,:])
    #右脚
    rightankle_array = convert_bvh_to_mujoco(joint_positions[:,25,:])
    rightankle_orien_array = convert_quaternion(joint_orientations[:,25,:])
    # rightankle_orien_array = [1,0,0,0]
    #右脚趾
    rightfoot_array = convert_bvh_to_mujoco(joint_positions[:,26,:]) 
    rightfoot_orien_array = convert_quaternion(joint_orientations[:,26,:])
    # rightfoot_orien_array = [1,0,0,0] 
    #左上腿
    lefthip_array = convert_bvh_to_mujoco(joint_positions[:,18,:]) 
    lefthip_orien_array = convert_quaternion(joint_orientations[:,18,:]) 
    lefthip_array[:,2] *= leg_scal
    #左下腿
    leftknee_array = convert_bvh_to_mujoco(joint_positions[:,19,:])
    leftknee_orien_array = convert_quaternion(joint_orientations[:,19,:])
    #左脚
    leftankle_array = convert_bvh_to_mujoco(joint_positions[:,20,:])
    leftankle_orien_array = convert_quaternion(joint_orientations[:,20,:])
    # leftankle_orien_array = [1,0,0,0]
    #左脚趾
    leftfoot_array = convert_bvh_to_mujoco(joint_positions[:,21,:])
    leftfoot_orien_array = convert_quaternion(joint_orientations[:,21,:])
    
    max_targets = len(com_array)

    #读取mujoco文件和设置跟踪任务
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    configuration = mink.Configuration(model)

    elbows = ["right_elbow", "left_elbow"]
    hands = ["right_wrist", "left_wrist"]
    shoulders = ["right_shoulder", "left_shoulder"]
    suogus = ["right_suogu", "left_suogu"]
    hips = ["right_hip", "left_hip"]
    knees = ["right_knee", "left_knee"]
    ankles = ["right_ankle", "left_ankle"]
    feet = ["right_foot", "left_foot"]
    handsend = ["right_handend", "left_handend"]

   

    tasks = [
        pelvis_orientation_task := mink.FrameTask(
            frame_name="base_link",
            frame_type="body",
            position_cost=0.0,
            orientation_cost=50.0,
        ),
        posture_task := mink.PostureTask(model, cost=1.0),
        com_task := mink.ComTask(cost=200.0),
    ]

   

    feet_tasks = []
    for foot in feet:
        task = mink.FrameTask(
            frame_name=foot,
            frame_type="site",
            position_cost=200.0,
            orientation_cost=10.0,
            lm_damping=1.0,
        )
        feet_tasks.append(task)
    tasks.extend(feet_tasks)

    hand_tasks = []
    for hand in hands:
        task = mink.FrameTask(
            frame_name=hand,
            frame_type="site",
            position_cost=200.0,
            orientation_cost=30.0,
            lm_damping=1.0,
        )
        hand_tasks.append(task)
    tasks.extend(hand_tasks)

    elbow_tasks = []
    for elbow in elbows:
        task = mink.FrameTask(
            frame_name=elbow,
            frame_type="site",
            position_cost=300.0,
            orientation_cost=10.0,
            lm_damping=1.0,
        )
        elbow_tasks.append(task)
    tasks.extend(elbow_tasks)

    shoulder_tasks = []
    for shouder in shoulders:
        task = mink.FrameTask(
            frame_name=shouder,
            frame_type="site",
            position_cost=200.0,
            orientation_cost=5.0,
            lm_damping=1.0,
        )
        shoulder_tasks.append(task)
    tasks.extend(shoulder_tasks)
    suogu_tasks = []
    for suogu in suogus:
        task = mink.FrameTask(
            frame_name=suogu,
            frame_type="site",
            position_cost=200.0,
            orientation_cost=5.0,
            lm_damping=1.0,
        )
        suogu_tasks.append(task)
    tasks.extend(suogu_tasks)

    knee_tasks = []
    for knee in knees:
        task = mink.FrameTask(
            frame_name=knee,
            frame_type="site",
            position_cost=200.0,
            orientation_cost=10.0,
            lm_damping=2.0,
        )
        knee_tasks.append(task)
    tasks.extend(knee_tasks)

    hip_tasks = []
    for hip in hips:
        task = mink.FrameTask(
            frame_name=hip,
            frame_type="site",
            position_cost=200.0,
            orientation_cost=10.0,
            lm_damping=2.0,
        )
        hip_tasks.append(task)
    tasks.extend(hip_tasks)

    ankle_tasks = []
    for ankle in ankles:
        task = mink.FrameTask(
            frame_name=ankle,
            frame_type="site",
            position_cost=200.0,
            orientation_cost=10.0,
            lm_damping=2.0,
        )
        ankle_tasks.append(task)
    tasks.extend(ankle_tasks)

    handend_tasks = []
    for handend in handsend:
        task = mink.FrameTask(
            frame_name=handend,
            frame_type="site",
            position_cost=100.0,
            orientation_cost=0.0,
            lm_damping=2.0,
        )
        handend_tasks.append(task)
    tasks.extend(handend_tasks)
    
    head_tasks = [
        head_task := mink.FrameTask(
            frame_name="head",
            frame_type="site",
            position_cost=200.0,
            orientation_cost=10.0,
            lm_damping=1.0,
            
        ),
        neck_task :=mink.FrameTask(
            frame_name="neck",
            frame_type="site",
            position_cost=200.0,
            orientation_cost=10.0,
            lm_damping=1.0,
        ),
         waist_task :=mink.FrameTask(
            frame_name="waist",
            frame_type="site",
            position_cost=200.0,
            orientation_cost=10.0,
            lm_damping=1.0,
        )
    ]
    tasks.extend(head_tasks)

    com_mid = model.body("com_target").mocapid[0]
    head_mid = model.body("head_target").mocapid[0]
    neck_mid = model.body("neck_target").mocapid[0]
    waist_mid = model.body("waist_target").mocapid[0]
    elbows_mid = [model.body(f"{elbow}_target").mocapid[0] for elbow in elbows]
    hands_mid = [model.body(f"{hand}_target").mocapid[0] for hand in hands]
    shoulders_mid = [model.body(f"{shoulder}_target").mocapid[0] for shoulder in shoulders]
    suogus_mid = [model.body(f"{suogu}_target").mocapid[0] for suogu in suogus]
    knees_mid = [model.body(f"{knee}_target").mocapid[0] for knee in knees]
    hips_mid = [model.body(f"{hip}_target").mocapid[0] for hip in hips]
    ankles_mid = [model.body(f"{ankle}_target").mocapid[0] for ankle in ankles]
    feet_mid = [model.body(f"{foot}_target").mocapid[0] for foot in feet]
    handsend_mid = [model.body(f"{handend}_target").mocapid[0] for handend in handsend]
    
    

    model = configuration.model
    data = configuration.data
    solver = "quadprog"
    q_y = (0, 1, 0, 0)
    
    
    # 创建CSV文件
    
    data_list = []
    data_list_vel = []
    data_list_end = []
    data_list_base = []
    time_column = []
   

  
    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize to the home keyframe.
        configuration.update_from_keyframe("home")
        posture_task.set_target_from_configuration(configuration)
        pelvis_orientation_task.set_target_from_configuration(configuration)

        # Initialize mocap bodies at their respective sites.
        for hand, foot,elbow,knee,shoulder,hip,ankle,suogu,handend  in zip(hands, feet,elbows,knees,shoulders,hips,ankles,suogus,handsend):
            mink.move_mocap_to_frame(model, data, f"{foot}_target", foot, "site")
            mink.move_mocap_to_frame(model, data, f"{hand}_target", hand, "site")
            mink.move_mocap_to_frame(model, data, f"{elbow}_target", elbow, "site")
            mink.move_mocap_to_frame(model, data, f"{knee}_target", knee, "site")
            mink.move_mocap_to_frame(model, data, f"{shoulder}_target", shoulder, "site")
            mink.move_mocap_to_frame(model, data, f"{hip}_target", hip, "site")
            mink.move_mocap_to_frame(model, data, f"{ankle}_target", ankle, "site")
            mink.move_mocap_to_frame(model, data, f"{suogu}_target", suogu, "site")
            mink.move_mocap_to_frame(model, data, f"{handend}_target", handend, "site")
        mink.move_mocap_to_frame(model, data, f"head_target", "head", "site")
        mink.move_mocap_to_frame(model, data, f"neck_target", "neck", "site")
        mink.move_mocap_to_frame(model, data, f"waist_target", "waist", "site")
        data.mocap_pos[com_mid] = data.subtree_com[1]

        prev_xpos_right_hand1 = data.body('Link_arm_r_07').xpos - data.body('base_link').xpos
        prev_xpos_left_hand1 = data.body('Link_arm_l_07').xpos - data.body('base_link').xpos
        prev_xpos_right_foot1 = data.body('Link_ankle_r_roll').xpos - data.body('base_link').xpos
        prev_xpos_left_foot1 = data.body('Link_ankle_l_roll').xpos - data.body('base_link').xpos
        prev_xpos_right_shoulder1 = data.body('Link_ankle_r_roll').xpos - data.body('base_link').xpos
        prev_xpos_left_shoulder1 = data.body('Link_ankle_l_roll').xpos - data.body('base_link').xpos


        body_pos_global = data.body('base_link').xpos
        body_quat1 = np.array(data.body('base_link').xquat)
        body_quat = np.array([body_quat1[1], body_quat1[2], body_quat1[3], body_quat1[0]])
        
        br = R.from_quat(body_quat)    
        body_rot = br.as_matrix()

        prev_xpos_right_hand = body_rot.T @ prev_xpos_right_hand1
        prev_xpos_left_hand = body_rot.T @ prev_xpos_left_hand1
        prev_xpos_right_foot = body_rot.T @ prev_xpos_right_foot1
        prev_xpos_left_foot = body_rot.T @ prev_xpos_left_foot1

        #末端相对于base的位置数据
        
        rate = RateLimiter(frequency=120.0, warn=False)

        com_current_target_index = 0
        last_time = time.time()
        mytime = 0.0
        ddt = 0.0

        #初始状态下的关节数据
        qpos = data.qpos
        vel = data.qvel
        xpos_right_hand = prev_xpos_right_hand
        xpos_left_hand = prev_xpos_left_hand
        xpos_right_foot = prev_xpos_right_foot
        xpos_left_foot = prev_xpos_left_foot

        velocity_right_hand = [0,0,0]
        velocity_left_hand = [0,0,0]
        velocity_right_foot = [0,0,0]
        velocity_left_foot = [0,0,0]

        base_linear_velocity_to_base = body_rot.T @ vel[:3]
        base_angular_velocity_to_base = body_rot.T @ vel[3:6]
        
        combined_end_effector = np.concatenate((xpos_right_hand, xpos_left_hand, xpos_right_foot, xpos_left_foot,velocity_right_hand,velocity_left_hand,velocity_right_foot,velocity_left_foot))
        combined_base_vel = np.concatenate((base_linear_velocity_to_base,base_angular_velocity_to_base))

        data_list.append(qpos.copy())
        data_list_vel.append(vel.copy())
        data_list_end.append(combined_end_effector.copy())
        data_list_base.append(combined_base_vel.copy())
        time_column.append(mytime)
        combined_data = np.hstack((data_list, data_list_vel,data_list_end,data_list_base))

        # R_quat = R.from_quat([-0.0361303 ,  0.00217463 ,-0.02868349 , 0.998933 ]) 
        # R_R = R_quat.as_matrix()
        R_R = np.array([[0,0,1],[0,1,0],[1,0,0]])

        while viewer.is_running() and com_current_target_index < max_targets:

            # # # # 定义每个部位的中间点和相应的阵列

            body_parts = {
                'com': (com_mid, com_array, com_orien_array),
                'head': (head_mid, head_array, head_orien_array),
                'neck': (neck_mid, neck_array, neck_orien_array),
                'waist': (waist_mid, waist_array, waist_orien_array),
                'feet': (feet_mid, rightfoot_array, rightfoot_orien_array, leftfoot_array, leftfoot_orien_array),
                'hands': (hands_mid, righthand_array, righthand_orien_array, lefthand_array, lefthand_orien_array),
                'elbows': (elbows_mid, rightelbow_array, rightelbow_orien_array, leftelbow_array, leftelbow_orien_array),
                'knees': (knees_mid, rightknee_array, rightknee_orien_array, leftknee_array, leftknee_orien_array),
                'shoulders': (shoulders_mid, rightshoulder_array, rightshoulder_orien_array, leftshoulder_array, leftshoulder_orien_array),
                'hips': (hips_mid, righthip_array, righthip_orien_array, lefthip_array, lefthip_orien_array),
                'ankles': (ankles_mid, rightankle_array, rightankle_orien_array, leftankle_array, leftankle_orien_array),
                'suogus': (suogus_mid, rightsuogu_array, rightsuogu_orien_array, leftsuogu_array, leftsuogu_orien_array),
                'handsend': (handsend_mid, righthandend_array, righthandend_orien_array, lefthandend_array, lefthandend_orien_array),

            }

            # 循环更新数据
            for part, values in body_parts.items():
                if part in ['com', 'head', 'neck','waist']:
                    mid_index, pos_array, quat_array = values
                    data.mocap_pos[mid_index] =    pos_array[com_current_target_index]
                    data.mocap_quat[mid_index] =  quat_array[com_current_target_index]
                    
                else:
                    mid_indices, right_array, right_orien_array, left_array, left_orien_array = values
                    data.mocap_pos[mid_indices[0]] =   right_array[com_current_target_index]
                    data.mocap_quat[mid_indices[0]] =  right_orien_array[com_current_target_index]                   
                    data.mocap_pos[mid_indices[1]] =   left_array[com_current_target_index]
                    data.mocap_quat[mid_indices[1]] = left_orien_array[com_current_target_index]
                    
             # Update task targets.
            com_task.set_target(data.mocap_pos[com_mid])
            pelvis_orientation_task.set_target(mink.SE3.from_mocap_id(data, com_mid))
            head_task.set_target(mink.SE3.from_mocap_id(data, head_mid)) #头
            neck_task.set_target(mink.SE3.from_mocap_id(data, neck_mid))
            waist_task.set_target(mink.SE3.from_mocap_id(data, waist_mid))
            
            for i, (hand_task,foot_task,elbow_task,knee_task,shoulder_task,hip_task,ankle_task,suogu_task,handend_task) in enumerate(zip(hand_tasks,feet_tasks, elbow_tasks,knee_tasks,shoulder_tasks,hip_tasks,ankle_tasks,suogu_tasks,handend_tasks)):           
                foot_task.set_target(mink.SE3.from_mocap_id(data, feet_mid[i]))
                hand_task.set_target(mink.SE3.from_mocap_id(data, hands_mid[i]))
                elbow_task.set_target(mink.SE3.from_mocap_id(data, elbows_mid[i]))
                knee_task.set_target(mink.SE3.from_mocap_id(data, knees_mid[i]))
                shoulder_task.set_target(mink.SE3.from_mocap_id(data, shoulders_mid[i]))
                hip_task.set_target(mink.SE3.from_mocap_id(data, hips_mid[i]))
                ankle_task.set_target(mink.SE3.from_mocap_id(data, ankles_mid[i]))
                suogu_task.set_target(mink.SE3.from_mocap_id(data, suogus_mid[i]))
                handend_task.set_target(mink.SE3.from_mocap_id(data, handsend_mid[i]))
            
            
            #记录当前时间
            current_time = time.time()
                
            vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-1)
            
            qpos=configuration.integrate(vel, rate.dt)
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)

            body_pos_global = (data.body('base_link').xpos).copy()
            body_quat1 = np.array(data.body('base_link').xquat).copy()            
            body_quat = np.array([body_quat1[1], body_quat1[2], body_quat1[3], body_quat1[0]])          
            br = R.from_quat(body_quat)
            body_rot = br.as_matrix()
            
            base_linear_velocity_to_base = body_rot.T @ vel[:3]
            base_angular_velocity_to_base = body_rot.T @ vel[3:6]
          
            
            xpos_right_hand1 = data.body('Link_arm_r_07').xpos - data.body('base_link').xpos
            xpos_left_hand1 = data.body('Link_arm_l_07').xpos - data.body('base_link').xpos
            xpos_right_foot1 = data.body('Link_ankle_r_roll').xpos - data.body('base_link').xpos
            xpos_left_foot1 = data.body('Link_ankle_l_roll').xpos - data.body('base_link').xpos

            xpos_right_hand = body_rot.T @ xpos_right_hand1
            xpos_left_hand = body_rot.T @ xpos_left_hand1
            xpos_right_foot = body_rot.T @ xpos_right_foot1
            xpos_left_foot = body_rot.T @ xpos_left_foot1

            velocity_right_hand = (xpos_right_hand - prev_xpos_right_hand) / rate.dt
            velocity_left_hand = (xpos_left_hand - prev_xpos_left_hand) / rate.dt
            velocity_right_foot = (xpos_right_foot - prev_xpos_right_foot) / rate.dt
            velocity_left_foot = (xpos_left_foot - prev_xpos_left_foot) / rate.dt
            mytime = mytime + rate.dt

            #     vel = mink.solve_ik(configuration, tasks, dt, solver, 1e-1)
            #     qpos=configuration.integrate(vel, dt)
            #     configuration.integrate_inplace(vel, dt)
            #     mujoco.mj_camlight(model, data)
            #     xpos_right_hand = data.body('Link_arm_r_07').xpos - data.body('base_link').xpos
            #     xpos_left_hand = data.body('Link_arm_l_07').xpos - data.body('base_link').xpos
            #     xpos_right_foot = data.body('Link_ankle_r_roll').xpos - data.body('base_link').xpos
            #     xpos_left_foot = data.body('Link_ankle_l_roll').xpos - data.body('base_link').xpos

            #     velocity_right_hand = (xpos_right_hand - prev_xpos_right_hand) / dt
            #     velocity_left_hand = (xpos_left_hand - prev_xpos_left_hand) / dt
            #     velocity_right_foot = (xpos_right_foot - prev_xpos_right_foot) / dt
            #     velocity_left_foot = (xpos_left_foot - prev_xpos_left_foot) / dt

            # mytime = mytime + dt           

            ddt = current_time - last_time                       

            last_time = current_time
            prev_xpos_right_hand = xpos_right_hand
            prev_xpos_left_hand = xpos_left_hand
            prev_xpos_right_foot = xpos_right_foot
            prev_xpos_left_foot = xpos_left_foot
            

            combined_end_effector = np.concatenate((xpos_right_hand, xpos_left_hand, xpos_right_foot, xpos_left_foot,
                                                    velocity_right_hand,velocity_left_hand,velocity_right_foot,velocity_left_foot))
                                                   
            combined_base_vel = np.concatenate((base_linear_velocity_to_base,base_angular_velocity_to_base))
   
            data_list.append(qpos.copy())
            data_list_vel.append(vel.copy())
            data_list_end.append(combined_end_effector.copy())
            data_list_base.append(combined_base_vel.copy())

            combined_data = np.hstack((data_list, data_list_vel,data_list_end,data_list_base))
 
            # mytime = rate.dt * com_current_target_index
            
            time_column.append(mytime)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()

            com_current_target_index = com_current_target_index + 1

   

    df = pd.DataFrame(combined_data)#combined_data
    df.insert(0, 'Time', time_column)
    # header = [#base的位置(x,y,z)及姿态（w,x,y,z）,以及各关节的旋转角度,38维
    #       'global_base_link_position.x', 'global_base_link_position.y', 'global_base_link_position.z']
    header = ['Time'] + [#base的位置(x,y,z)及姿态（w,x,y,z）,以及各关节的旋转角度,38维
          'global_base_link_position.x', 'global_base_link_position.y', 'global_base_link_position.z', 
          'global_base_link_rotation.w', 'global_base_link_rotation.x', 'global_base_link_rotation.y','global_base_link_rotation.z',
          'local_Link_head_yaw_rotation', 'local_Link_head_pitch_rotation',
          'local_Link_arm_r_01_rotation', 'local_Link_arm_r_02_rotation', 'local_Link_arm_r_03_rotation', 'local_Link_arm_r_04_rotation','local_Link_arm_r_05_rotation','local_Link_arm_r_06_rotation','local_Link_arm_r_07_rotation',
          'local_Link_arm_l_01_rotation', 'local_Link_arm_l_02_rotation', 'local_Link_arm_l_03_rotation', 'local_Link_arm_l_04_rotation','local_Link_arm_l_05_rotation','local_Link_arm_l_06_rotation','local_Link_arm_l_07_rotation',
          'local_Link_waist_pitch_rotation', 'local_Link_waist_roll_rotation', 'local_Link_waist_yaw_rotation',
          'local_Link_hip_r_roll_rotation','local_Link_hip_r_yaw_rotation','local_Link_hip_r_pitch_rotation','local_Link_knee_r_pitch_rotation','local_Link_ankle_r_pitch_rotation','local_Link_ankle_r_roll_rotation',
          'local_Link_hip_l_roll_rotation','local_Link_hip_l_yaw_rotation','local_Link_hip_l_pitch_rotation','local_Link_knee_l_pitch_rotation','local_Link_ankle_l_pitch_rotation','local_Link_ankle_l_roll_rotation',
          #base的线速度（x,y,z）和角速度(x,y,z)，以及各关节旋转的角速度，37维
          'global_base_link_velocity.x', 'global_base_link_velocity.y', 'global_base_link_velocity.z', 'global_base_link_angular_velocity.x', 'global_base_link_angular_velocity.y','global_base_link_angular_velocity.z',
          'local_Link_head_yaw_velocity', 'local_Link_head_pitch_velocity',
          'local_Link_arm_r_01_velocity', 'local_Link_arm_r_02_velocity', 'local_Link_arm_r_03_velocity', 'local_Link_arm_r_04_velocity','local_Link_arm_r_05_velocity','local_Link_arm_r_06_velocity','local_Link_arm_r_07_velocity',
          'local_Link_arm_l_01_velocity', 'local_Link_arm_l_02_velocity', 'local_Link_arm_l_03_velocity', 'local_Link_arm_l_04_velocity','local_Link_arm_l_05_velocity','local_Link_arm_l_06_velocity','local_Link_arm_l_07_velocity',
          'local_Link_waist_pitch_velocity', 'local_Link_waist_roll_velocity', 'local_Link_waist_yaw_velocity',
          'local_Link_hip_r_roll_velocity','local_Link_hip_r_yaw_velocity','local_Link_hip_r_pitch_velocity','local_Link_knee_r_pitch_velocity','local_Link_ankle_r_pitch_velocity','local_Link_ankle_r_roll_velocity',
          'local_Link_hip_l_roll_velocity','local_Link_hip_l_yaw_velocity','local_Link_hip_l_pitch_velocity','local_Link_knee_l_pitch_velocity','local_Link_ankle_l_pitch_velocity','local_Link_ankle_l_roll_velocity',
          #右手、左手、右脚、左脚相对于base的位置（x,y,z）,12维
          'right_hand_to_base.x','right_hand_to_base.y','right_hand_to_base.z', 'left_hand_to_base.x', 'left_hand_to_base.y', 'left_hand_to_base.z',
          'right_foot_to_base.x','right_foot_to_base.y','right_foot_to_base.z', 'left_foot_to_base.x','left_foot_to_base.y','left_foot_to_base.z',
          #双手、双脚相对于base的速度
          'right_hand_velocity_to_base.x','right_hand_velocity_to_base.y','right_hand_velocity_to_base.z', 'left_hand_velocity_to_base.x', 'left_velocity_hand_to_base.y', 'left_velocity_hand_to_base.z',
          'right_foot_velocity_to_base.x','right_foot_velocity_to_base.y','right_foot_velocity_to_base.z', 'left_foot_velocity_to_base.x','left_foot_velocity_to_base.y','left_foot_velocity_to_base.z',
          #机身坐标系下的线速度和角速度
          'base_linear_velocity_to_base.x','base_linear_velocity_to_base.y','base_linear_velocity_to_base.z','base_angular_velocity_to_base.x','base_angular_velocity_to_base.y','base_angular_velocity_to_base.z'
          
           ]  
    df.columns = header 
    df.to_csv(csv_file, index=False, header=True) 
    print(f'The motion has been retargeted and the file has been saved to path {csv_file}')
