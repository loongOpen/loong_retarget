import os
import sys

import numpy as np

sys.path.append(os.getcwd())
import argparse
import glob

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def play_retargeted_motion(npz_path):
    # 读取 retargeted motion 数据 (npz)
    entry_data = dict(np.load(open(npz_path, "rb"), allow_pickle=True))

    fps = entry_data['fps']
    dof_names = entry_data['dof_names']
    body_names = entry_data['body_names']
    dof_poistions = entry_data['dof_positions']
    dof_velocities = entry_data['dof_velocities']
    body_positions = entry_data['body_positions']
    body_rotations = entry_data['body_rotations']
    body_linear_velocities = entry_data['body_linear_velocities']
    body_angular_velocities = entry_data['body_angular_velocities']

    frame_num = body_positions.shape[0]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Humanoid Body Positions")

    # 取出所有 body 的轨迹范围，确定画布大小
    all_positions = body_positions.reshape(-1, 3)
    xmin, ymin, zmin = np.min(all_positions, axis=0)
    xmax, ymax, zmax = np.max(all_positions, axis=0)
    ax.set_xlim([xmin-0.1, xmax+0.1])
    ax.set_ylim([ymin-0.1, ymax+0.1])
    ax.set_zlim([zmin-0.1, zmax+0.1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # 初始化散点图
    scat = ax.scatter([], [], [], c='blue')

    # 更新函数
    def update(frame):
        nonlocal body_positions, scat  
        positions = body_positions[None, frame]
        scat._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
        ax.set_title(f"Frame: {frame+1}/{frame_num}")
        return scat

    # 创建动画
    ani = FuncAnimation(fig, update, frames=frame_num, interval=1000/fps, blit=False)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", type=str, default="datas/retargeted/demo")
    args = parser.parse_args()
    npz_path = args.npz_path
    all_npz = glob.glob(f"{npz_path}/**/*.npz", recursive=True)
    split_len = len(npz_path.split("/"))
    key_name_to_npz = {"0-" + "_".join(data_path.split("/")[split_len:]).replace(".npz", ""): data_path for data_path in all_npz}

    print("找到以下序列:")
    for i, key in enumerate(key_name_to_npz.keys()):
        print(f"{i+1}. {key}")

    selection = input("请选择要可视化的序列编号 (1-{}), 或按Enter跳过: ".format(len(key_name_to_npz)))
    
    if selection:
        try:
            selection_idx = int(selection) - 1
            if 0 <= selection_idx < len(key_name_to_npz):
                data_key = list(key_name_to_npz.keys())[selection_idx]
                npz_name = key_name_to_npz[data_key]
                play_retargeted_motion(npz_name)
            else:
                print("无效的选择")
        except ValueError:
            print("请输入有效的数字")
    else:
        print("跳过可视化")