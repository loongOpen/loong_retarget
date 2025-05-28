import csv
import os
import sys

sys.path.append(os.getcwd())
import time

import mujoco
import mujoco.viewer


class MjVis():
    def __init__(self, mjcf_path='robot/openloong/scene.xml',dt=0.02):
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.cam.lookat[:] = [0, 0, 0.4]  # 设置目标点
        self.viewer.cam.distance = 7.5         # 设置摄像头距离
        self.viewer.cam.elevation = -25        # 设置俯仰角
        self.viewer.cam.azimuth = 90           # 设置方位角
        self.dt = dt
        mujoco.mj_forward(self.model, self.data)

    def step(self, qpos):
        self.data.qpos[:38] = qpos
        mujoco.mj_forward(self.model, self.data)
        self.viewer.sync()
        time.sleep(self.dt)
    
    def play(self, all_qpos):
        for qpos in all_qpos:
            self.step(qpos)
    
    def stop(self):
        self.viewer.close()
        

if __name__ == "__main__":
    csv_path = 'datas/retargeted/goose-step/goose-step184_Skeleton0.csv'
    env = MjVis(dt=0.005)
    while(True):
        with open(csv_path, mode='r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)
            for row in csv_reader:
                data = [float(dof) for dof in row[1:39]]
                data[2] += 0.02
                env.step(data)