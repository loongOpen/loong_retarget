import os
import sys

sys.path.append(os.getcwd())
import time

import joblib
import mujoco
import numpy as np
import open3d as o3d
import torch
from mujoco import MjData, MjModel
from phc.smpllib.smpl_parser import SMPL_Parser
from phc.utils.torch_openloong_humanoid_batch import (OPENLOONG_ROTATION_AXIS,
                                                      Humanoid_Batch)
from scipy.spatial.transform import Rotation as sRot
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES
from torch.autograd import Variable

openloong_joint_names = ["base_link", 
                      'Link_head_yaw', 'Link_head_pitch',
                      'Link_arm_r_01', 'Link_arm_r_02', 'Link_arm_r_03', 'Link_arm_r_04', 'Link_arm_r_05', 'Link_arm_r_06', 'Link_arm_r_07',
                      'Link_arm_l_01', 'Link_arm_l_02', 'Link_arm_l_03', 'Link_arm_l_04', 'Link_arm_l_05', 'Link_arm_l_06', 'Link_arm_l_07',
                      'Link_waist_pitch', 'Link_waist_roll', 'Link_waist_yaw',
                      'Link_hip_r_roll', 'Link_hip_r_yaw', 'Link_hip_r_pitch', 'Link_knee_r_pitch', 'Link_ankle_r_pitch', 'Link_ankle_r_roll', 
                      'Link_hip_l_roll', 'Link_hip_l_yaw', 'Link_hip_l_pitch', 'Link_knee_l_pitch', 'Link_ankle_l_pitch', 'Link_ankle_l_roll']
mj_path = 'robot/openloong/OpenLoong.xml'
openloong_fk = Humanoid_Batch(mjcf_file=mj_path) # load forward kinematics model
#### Define corresonpdances between openloong and smpl joints

openloong_joint_pick = ['base_link', 'Link_waist_yaw',  'Link_hip_l_yaw', "Link_knee_l_pitch", "Link_ankle_l_pitch",  'Link_hip_r_yaw', 'Link_knee_r_pitch', 'Link_ankle_r_pitch', "Link_arm_l_01", "Link_arm_l_04", "Link_arm_l_07", "Link_arm_r_01", "Link_arm_r_04", "Link_arm_r_07", "Link_head_yaw"]
smpl_joint_pick = ["Torso", "Pelvis", "L_Hip",  "L_Knee", "L_Ankle",  "R_Hip", "R_Knee", "R_Ankle", "L_Shoulder", "L_Elbow", "L_Hand", "R_Shoulder", "R_Elbow", "R_Hand", "Neck"]
openloong_joint_pick_idx = [ openloong_joint_names.index(j) for j in openloong_joint_pick]
smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]


#### Preparing fitting varialbes
device = "cpu"

# dof = [[0, 0, 0, -np.pi/2, -np.pi/2, np.pi/2, 0, 0, 0, 0, -np.pi/2, np.pi/2, np.pi/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
dof = np.zeros((1, 31))
dof_pos = torch.zeros((1, 31))
# dof_pos[0, 2] = 1

pose_aa_openloong = torch.cat([torch.zeros((1, 1, 3)), OPENLOONG_ROTATION_AXIS * dof_pos[..., None]], axis = 1) 

###### prepare SMPL default pause for openloong
pose_aa_stand = np.zeros((1, 72))
rotvec = sRot.from_quat([0.5, 0.5, 0.5, 0.5]).as_rotvec()
pose_aa_stand[:, :3] = rotvec
pose_aa_stand = pose_aa_stand.reshape(-1, 24, 3)
pose_aa_stand = torch.from_numpy(pose_aa_stand.reshape(-1, 72))

smpl_parser_n = SMPL_Parser(model_path="script/retarget/smpl/model/smpl", gender="neutral")

###### Shape fitting
trans = torch.zeros([1, 3])
beta = torch.zeros([1, 10])
verts, joints = smpl_parser_n.get_joints_verts(pose_aa_stand, beta , trans)
offset = joints[:, SMPL_BONE_ORDER_NAMES.index("Torso")] - trans
root_trans_offset = trans + offset

fk_return = openloong_fk.fk_batch(pose_aa_openloong[None, ], root_trans_offset[None, 0:1])

shape_new = Variable(torch.zeros([1, 10]).to(device), requires_grad=True)
scale = Variable(torch.ones([1]).to(device), requires_grad=True) 
optimizer_shape = torch.optim.Adam([shape_new, scale],lr=0.08)

# Generate joint positions for Mujoco
# Load the Mujoco model
model = MjModel.from_xml_path(mj_path)
data = MjData(model)

# Set joint angles based on dof_pos
for i, joint_angle in enumerate(dof_pos[0].cpu().detach().numpy()):
    data.qpos[i] = joint_angle

# Forward simulation to compute joint positions
mujoco.mj_forward(model, data)

# Extract joint positions
mujoco_joint_positions = data.xpos

for iteration in range(2000):
    verts, joints = smpl_parser_n.get_joints_verts(pose_aa_stand, shape_new, trans[0:1])
    root_pos = joints[:, SMPL_BONE_ORDER_NAMES.index("Torso")]
    joints = (joints - root_pos) * scale + root_pos
    diff = fk_return.global_translation_extend[:, :, openloong_joint_pick_idx] - joints[:, smpl_joint_pick_idx]
    diff = diff.reshape(15, 3)

    diagonal_elements = torch.tensor([1.0, 5.0, 4.0, 4.0, 10.0, 4.0, 4.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0 ,1.0, 1.0])
    diagonal_matrix = torch.diag(diagonal_elements)
    loss_g = diff.transpose(0, 1) @ diagonal_matrix @ diff
    loss = loss_g.trace()
    if iteration % 100 == 0:
        print(iteration, loss.item() * 1000)
    
    if iteration == 0 or iteration == 1900:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        coord_frame.translate([0, 0, 0], relative=False)
        coord_frame.rotate(sRot.from_quat([0, 0, 0, 1]).as_matrix(), center=coord_frame.get_center())

        pcd1 = o3d.geometry.PointCloud()
        pcd2 = o3d.geometry.PointCloud()
        pcd3 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(fk_return['global_translation_extend'][0, 0, openloong_joint_pick_idx].cpu().detach().numpy())
        pcd1.colors = o3d.utility.Vector3dVector(o3d.utility.Vector3dVector(np.tile(np.array([[1, 0, 0]]), (fk_return['global_translation_extend'][0, 0, openloong_joint_pick_idx].cpu().detach().numpy().shape[0], 1))))
        pcd2.points = o3d.utility.Vector3dVector(joints[0, smpl_joint_pick_idx].cpu().detach().numpy())
        pcd2.colors = o3d.utility.Vector3dVector(o3d.utility.Vector3dVector(np.tile(np.array([[0, 0, 1]]), (joints[0, smpl_joint_pick_idx].cpu().detach().numpy().shape[0], 1))))
        # 使用 mujoco 计算各个关节点的位置
        pcd3.points = o3d.utility.Vector3dVector(mujoco_joint_positions + root_pos.cpu().detach().numpy())
        pcd3.colors = o3d.utility.Vector3dVector(o3d.utility.Vector3dVector(np.tile(np.array([[0, 1, 0]]), (mujoco_joint_positions.shape[0], 1))))

        o3d.visualization.draw_geometries([pcd1, pcd2, coord_frame]) 

    optimizer_shape.zero_grad()
    loss.backward()
    optimizer_shape.step()

os.makedirs("script/retarget/smpl/model/openloong", exist_ok=True)
joblib.dump((shape_new.detach(), scale), "script/retarget/smpl/model/openloong/shape_optimized_v2.pkl") # V2 has hip jointsrea
print(f"shape fitted and saved to model/openloong/shape_optimized_v2.pkl")