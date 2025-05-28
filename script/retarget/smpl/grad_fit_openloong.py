import glob
import os
import sys

sys.path.append(os.getcwd())
import argparse
import time

import joblib
import numpy as np
import open3d as o3d
import phc.utils.rotation_conversions as tRot
import torch
from phc.smpllib.smpl_parser import SMPL_BONE_ORDER_NAMES, SMPL_Parser
from phc.utils.torch_openloong_humanoid_batch import (OPENLOONG_ROTATION_AXIS,
                                                      Humanoid_Batch)
from scipy.spatial.transform import Rotation as sRot
from torch.autograd import Variable
from tqdm import tqdm

from script.vis.vis_mujoco import MjVis


def create_smooth_difference_matrix(size, alpha=0.1):
    """创建平滑差分矩阵，alpha控制平滑程度"""
    matrix = torch.eye(size)
    if size >= 2:
        indices = torch.arange(size - 1)
        matrix[indices + 1, indices] = -1
    # 添加对角正则化项，使差分更平滑
    return matrix + alpha * torch.eye(size)

def load_amass_data(data_path):
    entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))

    if not 'mocap_framerate' in  entry_data:
        return None
    framerate = entry_data['mocap_framerate']

    root_trans = entry_data['trans']
    pose_aa = np.concatenate([entry_data['poses'][:, :66], np.zeros((root_trans.shape[0], 6))], axis = -1)
    betas = entry_data['betas']
    gender = entry_data['gender']
    return {
        "pose_aa": pose_aa,
        "gender": gender,
        "trans": root_trans, 
        "betas": betas,
        "fps": framerate
    }

def o3d_visualizer(openloong_fk_result, smpl_joints, smpl_joints_quat, fps):
    # Visualize the result
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd1 = o3d.geometry.PointCloud()
    pcd2 = o3d.geometry.PointCloud()

    for i in range(len(smpl_joints)):
        # 更新红色点云（openloong关节）
        points1 = openloong_fk_result['global_translation_extend'][0, i, openloong_joint_pick_idx].cpu().detach().numpy()
        pcd1.points = o3d.utility.Vector3dVector(points1)
        pcd1.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([[1, 0, 0]]), (points1.shape[0], 1)))

        # 更新蓝色点云（SMPL关节）
        points2 = smpl_joints[i, smpl_joint_pick_idx].cpu().detach().numpy()
        pcd2.points = o3d.utility.Vector3dVector(points2)
        pcd2.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([[0, 0, 1]]), (points2.shape[0], 1)))

        # 获取当前关节的四元数 [w, x, y, z]
        quat = smpl_joints_quat[i, 8].detach().cpu().numpy()
        position = smpl_joints[i, 8].cpu().detach().numpy()
        coord_frame_smpl = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        coord_frame_smpl.rotate(coord_frame_smpl.get_rotation_matrix_from_quaternion(quat))
        coord_frame_smpl.translate(position, relative=False)

        quat = openloong_fk_result['global_rotation_extend'][0, i, 31-6].cpu().detach().numpy()
        position = openloong_fk_result['global_translation_extend'][0, i, 31-6].cpu().detach().numpy()
        coord_frame_2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        coord_frame_2.rotate(coord_frame_2.get_rotation_matrix_from_quaternion(quat))
        coord_frame_2.translate(position, relative=False)

        if i == 0:
            vis.add_geometry(pcd1)
            vis.add_geometry(pcd2)
            
        # 更新可视化
        vis.add_geometry(coord_frame_smpl, reset_bounding_box=False)
        vis.add_geometry(coord_frame_2, reset_bounding_box=False)

        vis.update_geometry(pcd1)
        vis.update_geometry(pcd2)
        vis.poll_events()
        vis.update_renderer()

        vis.remove_geometry(coord_frame_smpl, reset_bounding_box=False)
        vis.remove_geometry(coord_frame_2, reset_bounding_box=False)

        # 控制帧率
        time.sleep(1/fps)  # 每帧50ms
    vis.destroy_window()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_path", type=str, default="datas/mocap/demo")
    parser.add_argument("--save_path", type=str, default="datas/retargeted/demo")
    args = parser.parse_args()
    save_path = args.save_path
    
    device = torch.device("cuda:0")

    openloong_fk = Humanoid_Batch(mjcf_file='robot/openloong/OpenLoong.xml', device = device)
    # get joint_idx-joint_name key-value pair from the mjcf file
    openloong_joint_names = openloong_fk.model_names
    openloong_rotation_axis = OPENLOONG_ROTATION_AXIS.to(device)

    openloong_joint_pick = ['base_link', 'Link_waist_yaw', 'Link_hip_l_yaw', "Link_knee_l_pitch", "Link_ankle_l_pitch",  'Link_hip_r_yaw', 'Link_knee_r_pitch', 'Link_ankle_r_pitch', "Link_arm_l_01", "Link_arm_l_03", "Link_arm_l_06", "Link_arm_r_01", "Link_arm_r_03", "Link_arm_r_06", "Link_head_yaw"]
    smpl_joint_pick = ["Torso", "Pelvis", "L_Hip",  "L_Knee", "L_Ankle",  "R_Hip", "R_Knee", "R_Ankle", "L_Shoulder", "L_Elbow", "L_Hand", "R_Shoulder", "R_Elbow", "R_Hand", "Neck"]
    openloong_joint_rot_pick = ["Link_ankle_l_roll", "Link_ankle_r_roll"]
    smpl_joint_rot_pick = ["L_Ankle", "R_Ankle"]

    openloong_joint_pick_idx = [ openloong_joint_names.index(j) for j in openloong_joint_pick]
    smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]
    openloong_joint_rot_pick_idx = [openloong_joint_names.index(j) for j in openloong_joint_rot_pick]
    smpl_joint_rot_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_rot_pick]

    smpl_parser_n = SMPL_Parser(model_path="script/retarget/smpl/model/smpl", gender="neutral")
    smpl_parser_n.to(device)

    shape_new, scale = joblib.load("script/retarget/smpl/model/openloong/shape_optimized_v2.pkl")
    shape_new = shape_new.to(device)
    scale = scale.detach().to(device)

    # read all AMASS datum (SMPL + H G) under folder ../../datas/mocap/demo(default folder) recursively
    motion_path = args.motion_path
    all_pkls = glob.glob(f"{motion_path}/**/*.npz", recursive=True)
    split_len = len(motion_path.split("/"))
    key_name_to_pkls = {"0-" + "_".join(data_path.split("/")[split_len:]).replace(".npz", ""): data_path for data_path in all_pkls}
    
    if len(key_name_to_pkls) == 0:
        raise ValueError(f"No motion files found in {motion_path}")
    FPS_required = 30

    data_dump = {}
    pbar = tqdm(key_name_to_pkls.keys())
    for data_key in pbar:
        amass_data = load_amass_data(key_name_to_pkls[data_key])
        if amass_data is None:
            print(f"Skipping invalid data: {data_key}")
            continue
        skip = int(amass_data['fps']//FPS_required)

        # 先接收amass格式
        pose_aa = amass_data['pose_aa'][::skip, :66]
        trans = torch.from_numpy(amass_data['trans'][::skip]).float().to(device)
        N = trans.shape[0]
        pose_aa = torch.from_numpy(np.concatenate((pose_aa, np.zeros((N, 6))), axis = -1)).float().to(device)

        # 对齐SMPL格式数据
        smpl_pose = pose_aa.clone().detach()
        # get the joints from SMPL, notice that joints[:,0] is the Pelvis joint position
        verts, joints = smpl_parser_n.get_joints_verts(smpl_pose, shape_new, trans)
        # 直接正运动学求各link的旋转矩阵
        smpl_joints_quat = smpl_parser_n.get_joints_quat(smpl_pose)

        # get the root rotation and position
        # 对齐SMPL坐标系和青龙坐标系
        pose_aa[:, 0:3] = torch.from_numpy((sRot.from_rotvec(pose_aa.cpu().detach()[:, 0:3]) * sRot.from_quat([-0.5,-0.5,-0.5, 0.5])).as_rotvec())
        root_idx = SMPL_BONE_ORDER_NAMES.index("Torso")
        root_trans_offset = joints[:, root_idx]
        reshape_root_trans_offset = joints[:, root_idx:root_idx+1, :]
        joints = (joints - reshape_root_trans_offset) * scale  + reshape_root_trans_offset
        root_rot_offset = pose_aa[:, 0:3]

        # 31 joints total, index see `openloong_joint_names`
        dof = [0, 0, 0, -np.pi/2, -np.pi/2, np.pi/2, 0, 0, 0, 0, -np.pi/2, np.pi/2, np.pi/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        dof_single_frame = dof

        dof *= N
        dof_pos = torch.tensor(dof).to(device).reshape((1, N, -1, 1))

        dof_pos_AD = Variable(dof_pos, requires_grad=True)
        last_dof_pos = dof_pos

        # 使用Adam优化器
        optimizer_pose = torch.optim.Adam([dof_pos_AD], lr=0.08, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer_pose, step_size=20, gamma=0.8)

        # 使用平滑差分矩阵
        differential_matrix = create_smooth_difference_matrix(N, alpha=0.3).to(device)
     
        for iteration in range(500):
            pose_aa_openloong_new = torch.cat([root_rot_offset[None, :, None], openloong_rotation_axis * dof_pos_AD], axis = 2).to(device)
            fk_return = openloong_fk.fk_batch(pose_aa_openloong_new, root_trans_offset[None, ])
    
            # keypoints 位置误差
            diff = fk_return['global_translation_extend'][:, :, openloong_joint_pick_idx] - joints[:, smpl_joint_pick_idx]
            loss_g = diff.norm(dim=-1).mean()
    
            # 关节角度平滑性惩罚
            loss_angle_change = torch.norm(dof_pos_AD[:, 1:] - dof_pos_AD[:, :-1], dim=2).mean()
    
            # 对腿部关节应用差分约束，提取青龙所有腿部关节的索引，索引顺序见 `openloong_joint_names`
            leg_joint_indices = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
            leg_dof = dof_pos_AD[:, :, leg_joint_indices, :].reshape(N, -1)
            loss_leg_diff = (differential_matrix @ leg_dof).norm(dim=-1).mean()

            # 求关键link的旋转矩阵误差，例如足部误差
            smpl_joint_rot = smpl_joints_quat[:, smpl_joint_rot_pick_idx]
            openloong_joint_rot = fk_return['global_rotation_extend'][0, :, openloong_joint_rot_pick_idx] 
            loss_key_link_rot = tRot.quaternion_multiply(tRot.quaternion_invert(openloong_joint_rot), smpl_joint_rot)
            loss_key_link_rot = tRot.quaternion_to_axis_angle(loss_key_link_rot).norm(dim=-1).mean()
    
            # 组合损失
            loss = 1.1 * loss_g + 0.06 * loss_angle_change + 0.007 * loss_leg_diff + 0.1 * loss_key_link_rot

            optimizer_pose.zero_grad()
            loss.backward()
            optimizer_pose.step()
            scheduler.step()
            
            pbar.set_description_str(f"{data_key} FPS:{amass_data['fps']}->{iteration + 1} {loss.item() * 1000}")
            dof_pos_AD.data.clamp_(openloong_fk.joints_range[:, 0, None], openloong_fk.joints_range[:, 1, None])
        
        # open3d 可视化
        # o3d_visualizer(fk_return, joints, smpl_joints_quat, fps=FPS_required)

        # 对齐躯干高度
        root_trans_offset = root_trans_offset - root_trans_offset[0] + torch.from_numpy(np.array([0, 0, 1.15])).to(device)
        rot = tRot.axis_angle_to_quaternion(root_rot_offset).cpu().detach().numpy()
        dofs = dof_pos_AD[0].squeeze().cpu().detach().numpy()
        pos = root_trans_offset.cpu().detach().numpy()
        poses = np.concatenate((pos, rot, dofs), axis=1)
        mujoco_player = MjVis(dt=1/FPS_required)
        mujoco_player.play(poses)
        mujoco_player.stop()

        pose_aa_openloong_new = torch.cat([root_rot_offset[None, :, None], openloong_rotation_axis * dof_pos_AD, torch.zeros((1, N, 3, 3), device=device)], axis = 2)
        fk_return = openloong_fk.fk_batch(pose_aa_openloong_new, root_trans_offset[None, ])

        # 差分求关节角速度
        dof_velocities = np.diff(dofs, axis=0, prepend=dofs[:1]) * FPS_required
        result = {
            'fps': FPS_required,
            'dof_names': openloong_joint_names[1:],
            'body_names': openloong_joint_names[1:],
            'dof_positions': dofs,
            'dof_velocities': dof_velocities,
            'body_positions': root_trans_offset.squeeze().cpu().detach().numpy(),
            'body_rotations': root_rot_offset.squeeze().cpu().detach().numpy(),
            'body_linear_velocities': np.zeros((N, len(openloong_joint_names), 3), dtype=np.float32),
            'body_angular_velocities': np.zeros((N, len(openloong_joint_names), 3), dtype=np.float32)
        }
        np.savez(save_path + "/" + data_key + "_retargeted.npz", **result)


        data_dump[data_key]={
            "root_trans_offset": root_trans_offset.squeeze().cpu().detach().numpy(),
            "pose_aa": pose_aa_openloong_new.squeeze().cpu().detach().numpy(),   
            "dof": dofs, 
            "root_rot": sRot.from_rotvec(root_rot_offset.cpu().numpy()).as_quat(),
            "fps": 30
            }
        
        # print(f"dumping {data_key} for testing, remove the line if you want to process all data")
    joblib.dump(data_dump, "datas/retargeted/openloong/demo.pkl")