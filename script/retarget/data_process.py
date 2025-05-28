import os
import sys

sys.path.append(os.getcwd())
import csv

import cvxpy as cp
import numpy as np
import pandas as pd

from script.vis.vis_mujoco import MjVis


class DataProcess():
	def __init__(self, data_path, save_path):
		self.data_path = data_path
		self.save_path = save_path
		self.vis = MjVis(dt=0.005)

		# === 参数设置 ===
		FPS = 120
		self.t = 1/FPS
		self.v_max = 10.0
		self.a_max = 20.0
		self.lambda_v = 0.0005
		self.lambda_a = 0.0001

	def handle(self): # FIXME
		ori_qposes = self.load_data(self.data_path)
		res_qposes = []
		filter_rate = 0.06
		qpos_clip = 0.02
		fix_height = 0.006
		pre_qpos = np.array(ori_qposes[0])
		for qpos in ori_qposes:
			qpos = np.array(qpos)
			qpos = np.clip(qpos, pre_qpos-qpos_clip, pre_qpos+qpos_clip)
			pre_qpos = pre_qpos * (1 - filter_rate) + qpos * filter_rate
			pre_qpos[2] += fix_height
			res_qposes.append(pre_qpos.copy())
		self.mj_vis(res_qposes)
		self.save_data(self.save_path, res_qposes)
	
	def optimize(self):
		# === 读取轨迹（从第10行开始） ===
		original_traj = self.load_data(self.data_path, 15)
		original_traj = np.array(original_traj)
		N, J = original_traj.shape
		print(f"The value of N,J is: {N} {J}")

		# === 初始化变量 ===
		X = cp.Variable((N, J)) #优化变量，目标轨迹X
		V = cp.vstack([np.zeros((1, J)), (X[1:] - X[:-1]) / self.t])  # 速度，行数为 N
		A = cp.vstack([np.zeros((1, J)), (V[1:] - V[:-1]) / self.t])  # 加速度，行数为 N

		# === 目标函数 ===
		objective = cp.Minimize(
		cp.sum_squares(X - original_traj) +
		self.lambda_v * cp.sum_squares(V) +
		self.lambda_a * cp.sum_squares(A))

		# === 约束 ===
		constraints = [
			cp.abs(V) <= self.v_max,
			cp.abs(A) <= self.a_max,]

		# === 求解 ===
		print("🚀 开始优化轨迹...")
		prob = cp.Problem(objective, constraints)
		prob.solve(solver=cp.OSQP, verbose=True)

		# === 构造时间序列 ===
		time_array = np.arange(N).reshape(-1, 1) * self.t  # shape: (N, 1)

		# === 保存优化结果 ===
		if X.value is not None:
			optimized_traj = X.value
			output = np.hstack([time_array, optimized_traj])
			self.save_data(self.save_path, output)
			print("✅ 优化完成，结果已保存")
		else:
			print("❌ 优化失败")
			exit()
	
	def mj_vis(self, all_qpos):
		for qpos in all_qpos:
			self.vis.step(qpos)

	def load_data(self, load_path, skip_rows=0):
		with open(load_path, mode='r', encoding='utf-8') as file:
			csv_reader = csv.reader(file)
			self.header = next(csv_reader)
			all_data = []
			# Skip the specified number of rows after reading the header
			for _ in range(skip_rows):
				next(csv_reader, None)
			for row in csv_reader:
				data = [float(dof) for dof in row[1:39]]
				all_data.append(data)
		return all_data

	def save_data(self, save_path, save_data):
		save_data = np.array(save_data)
		df = pd.DataFrame(save_data, columns=self.header[0:39])
		df.to_csv(save_path, index=False)


if __name__ == "__main__":
	csv_path = 'datas/retargeted/base/walk_Skeleton0.csv'
	save_path = 'datas/output/walk.csv'

	test = DataProcess(csv_path, save_path)
	test.optimize()
