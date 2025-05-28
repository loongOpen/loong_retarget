import os
import sys

import numpy as np

sys.path.append(os.getcwd())
import argparse
import glob
import time

import joblib
import open3d as o3d
import smplx
import torch


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

betas, scale = joblib.load("script/retarget/smpl/model/openloong/shape_optimized_v2.pkl")

model = smplx.create(
    "script/retarget/smpl/model/smpl/SMPL_NEUTRAL.pkl",
    model_type="smpl",
)

def visualize_sequence(vertices, faces, target_number_of_faces=500):
    """可视化3D模型序列，支持交互操作"""
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    # 创建初始网格
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices[0])
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    # 添加网格到可视化器
    vis.add_geometry(mesh)
    
    # 动画控制参数
    current_frame = 0
    paused = False
    step = 1  # 帧步长，可通过按键调整
    
    def toggle_pause(vis):
        nonlocal paused
        paused = not paused
        return False
    
    def next_frame(vis):
        nonlocal current_frame
        current_frame = (current_frame + 1) % len(vertices)
        update_mesh()
        return False
    
    def prev_frame(vis):
        nonlocal current_frame
        current_frame = (current_frame - 1) % len(vertices)
        update_mesh()
        return False
    
    def increase_speed(vis):
        nonlocal step
        step = min(10, step + 1)
        print(f"Frame step increased to: {step}")
        return False
    
    def decrease_speed(vis):
        nonlocal step
        step = max(1, step - 1)
        print(f"Frame step decreased to: {step}")
        return False
    
    def update_mesh():
        """更新网格数据"""
        nonlocal mesh
        mesh.vertices = o3d.utility.Vector3dVector(vertices[current_frame])
        mesh.compute_vertex_normals()
        vis.update_geometry(mesh)
    
    # 注册按键事件
    vis.register_key_callback(ord(" "), toggle_pause)  # 空格键暂停/继续
    vis.register_key_callback(ord("N"), next_frame)    # N键下一帧
    vis.register_key_callback(ord("B"), prev_frame)    # B键上一帧
    vis.register_key_callback(ord("+"), increase_speed)  # +键增加速度
    vis.register_key_callback(ord("-"), decrease_speed)  # -键减少速度
    
    # 主循环
    while vis.poll_events():
        if not paused:
            current_frame = (current_frame + step) % len(vertices)
            update_mesh()
        
        vis.update_renderer()
        time.sleep(0.05)  # 减少CPU使用率
    
    vis.destroy_window()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_path", type=str, default="datas/mocap/demo")
    args = parser.parse_args()
    motion_path = args.motion_path
    all_pkls = glob.glob(f"{motion_path}/**/*.npz", recursive=True)
    split_len = len(motion_path.split("/"))
    key_name_to_pkls = {"0-" + "_".join(data_path.split("/")[split_len:]).replace(".npz", ""): data_path for data_path in all_pkls}
    
    # 让用户选择一个序列而不是全部播放
    print("找到以下序列:")
    for i, key in enumerate(key_name_to_pkls.keys()):
        print(f"{i+1}. {key}")
    
    selection = input("请选择要可视化的序列编号 (1-{}), 或按Enter跳过: ".format(len(key_name_to_pkls)))
    
    if selection:
        try:
            selection_idx = int(selection) - 1
            if 0 <= selection_idx < len(key_name_to_pkls):
                data_key = list(key_name_to_pkls.keys())[selection_idx]
                amass_data = load_amass_data(key_name_to_pkls[data_key])
                if amass_data is None:
                    print(f"无效数据: {data_key}")
                else:
                    print(f"加载序列: {data_key}")
                    skip = int(amass_data['fps']//30)
                    pose_aa = amass_data['pose_aa'][::skip, :72]
                    global_trans = amass_data['trans'][::skip, :3]
                    
                    global_orient = torch.from_numpy(pose_aa[:, :3]).float() 
                    body_pose = torch.from_numpy(pose_aa[:, 3:]).float()
                    global_trans = torch.from_numpy(global_trans).float()
                    
                    output = model(
                        betas=betas,
                        body_pose=body_pose,
                        global_orient=global_orient,
                        transl=global_trans,
                        return_verts=True,
                    )
                    
                    vertices = output.vertices.detach().cpu().numpy()
                    faces = model.faces.astype(np.int32)
                    
                    # 可视化序列
                    visualize_sequence(vertices, faces)
            else:
                print("无效的选择")
        except ValueError:
            print("请输入有效的数字")
    else:
        print("跳过可视化")
