import os
import sys

sys.path.append(os.getcwd())
import csv

import numpy as np
from isaacgym import gymapi, gymtorch, gymutil
import torch

# load asset
asset_root = "robot/openloong"
asset_file = 'OpenLoong.urdf'

csv_path = 'datas/retargeted/goose-step/goose-step184_Skeleton0.csv'


# parse arguments
args = gymutil.parse_arguments()
gym = gymapi.acquire_gym()

# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = dt = 1.0 / 60.0
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
if args.physics_engine == gymapi.SIM_FLEX:
    pass
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

if not args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

asset_options = gymapi.AssetOptions()
asset_options.use_mesh_materials = True

print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# set up the env grid
num_envs = 1
num_per_row = 5
spacing = 5
env_lower = gymapi.Vec3(-spacing, spacing, 0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# position the camera
cam_pos = gymapi.Vec3(0, -10.0, 3)
cam_target = gymapi.Vec3(0, 0, 0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# cache useful handles
envs = []
actor_handles = []

num_dofs = gym.get_asset_dof_count(asset)
print("Creating %d environments" % num_envs)
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)
    # add actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0, 0.0)
    pose.r = gymapi.Quat(0, 0.0, 0.0, 1)

    actor_handle = gym.create_actor(env, asset, pose, "actor", i, 1)
    actor_handles.append(actor_handle)

    # set default DOF positions
    dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
    gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)

gym.prepare_sim(sim)
device = (torch.device("cuda", index=0) if torch.cuda.is_available() else torch.device("cpu"))

rigidbody_state = gym.acquire_rigid_body_state_tensor(sim)
rigidbody_state = gymtorch.wrap_tensor(rigidbody_state)
rigidbody_state = rigidbody_state.reshape(num_envs, -1, 13)

actor_root_state = gym.acquire_actor_root_state_tensor(sim)
actor_root_state = gymtorch.wrap_tensor(actor_root_state)

env_ids = torch.arange(num_envs).int().to(args.sim_device)
with open(csv_path, mode='r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    header = next(csv_reader)
    all_data = []
    for row in csv_reader:
        data = [float(dof) for dof in row[1:39]]
        all_data.append(data)

while not gym.query_viewer_has_closed(viewer):
    for dof in all_data:
        gym.clear_lines(viewer)
        gym.refresh_rigid_body_state_tensor(sim)

        root_pos, root_rot = torch.tensor(dof[:3]), torch.roll(torch.tensor(dof[3:7]), shifts=-1, dims=0)
        root_vel, root_ang_vel = torch.zeros_like(root_pos), torch.zeros_like(root_pos)
        root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1).repeat(num_envs, 1).to(device=device)
        
        dof_pos = torch.tensor(dof[7:], device=device).reshape(1, -1)
        
        gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(root_states), gymtorch.unwrap_tensor(env_ids), len(env_ids))
        gym.refresh_actor_root_state_tensor(sim)

        dof_pos_az = torch.zeros_like(dof_pos)
        dof_pos_az[0][:7] = dof_pos[0][9:16]
        dof_pos_az[0][7:14] = dof_pos[0][2:9]
        dof_pos_az[0][14:16] = dof_pos[0][:2]
        dof_pos_az[0][16:19] = dof_pos[0][16:19]
        dof_pos_az[0][19:25] = dof_pos[0][25:]
        dof_pos_az[0][25:] = dof_pos[0][19:25]
        
        dof_state = torch.stack([dof_pos_az, torch.zeros_like(dof_pos_az)], dim=-1).squeeze().repeat(num_envs, 1)
        gym.set_dof_state_tensor_indexed(sim, gymtorch.unwrap_tensor(dof_state), gymtorch.unwrap_tensor(env_ids), len(env_ids))
        gym.simulate(sim)
        gym.refresh_rigid_body_state_tensor(sim)
        gym.fetch_results(sim, True)

        # update the viewer
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

        # Wait for dt to elapse in real time.
        # This synchronizes the physics simulation with the rendering rate.
        gym.sync_frame_time(sim)

print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)