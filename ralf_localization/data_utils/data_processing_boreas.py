import faiss
import numpy as np
import ipdb
from tqdm import tqdm
import json
from interpolate_poses import *
import os
import os.path as osp
import matplotlib.pyplot as plt
from pyboreas.utils.utils import yawPitchRollToRot


def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def closest_lidar_timestamps(radar_time, lidar_time, lidar_all_poses):
    radar_time = np.array(radar_time)
    lidar_time = np.array(lidar_time)

    lidar_closest_times = []
    lidar_poses = []

    for time in radar_time:
        nearest_idx = find_nearest_idx(lidar_time, time)
        lidar_closest_times.append(lidar_time[nearest_idx])
        lidar_poses.append(lidar_all_poses[nearest_idx])

    # lidar_closest_times = list(lidar_closest_times)

    return lidar_closest_times, lidar_poses


def extract_poses(pose_file):
    base2lidar = np.eye(4, dtype=np.float64)
    poses = []
    stamps = []
    first_idx = True
    with open(pose_file, 'r') as f:
        f.readline()  # Skip header
        for x in f:
            x = x.strip().split(',')
            stamps.append(int(x[0]))
            x = [float(v) for v in x[1:]]
            pose = np.eye(4, dtype=np.float64)
            rot_mat = yawPitchRollToRot(x[8], x[7], x[6])  # Conversion from xyzw to wxyz, and then to rotation matrix
            pose[:3, :3] = rot_mat
            pose[:3, 3] = x[:3]
            if first_idx:
                offset_x, offset_y, offset_z = pose[0, 3], pose[1, 3], pose[2, 3]
                first_idx = False
            pose[0, 3] -= offset_x #623000
            pose[1, 3] -= offset_y #4848000
            pose[2, 3] -= offset_z #150
            # pose = cam0_to_velo.inverse() @ (pose @ cam0_to_velo)
            pose = pose @ base2lidar
            poses.append(pose)

    return poses, stamps


if __name__ == '__main__':

    # Set the correct dataroot for the boreas dataset. This directory should contain the sequence directories
    dataroot = '/data/boreas'

    sequences = [item for item in sorted(os.listdir(dataroot)) if os.path.isdir(osp.join(dataroot, item))]

    # Placeholders to store values from all sequences
    gps_poses = []
    radar_odom_gt_poses = []
    seq_len_cumulative = []
    seq_len_last_value = 0
    all_radar_poses = []
    all_lidar_poses = []
    all_radar_stamps = []
    all_sequences = []

    for seq in sequences:
        seq_path = osp.join(dataroot, seq)
        radar_pose_file = osp.join(seq_path, 'applanix', 'radar_poses.csv')
        lidar_pose_file = osp.join(seq_path, 'applanix', 'lidar_poses.csv')

        # Extract poses and stamps from individual sensors        
        radar_poses, radar_stamps = extract_poses(radar_pose_file)
        lidar_poses, lidar_stamps = extract_poses(lidar_pose_file)

        all_sequences += [seq] * len(radar_poses)

        closest_lidar_stamps, closest_lidar_poses = closest_lidar_timestamps(radar_stamps, lidar_stamps, lidar_poses)

        all_radar_poses += radar_poses
        all_lidar_poses += closest_lidar_poses
        all_radar_stamps += radar_stamps

        # Accumulate total length of sequences
        seq_len_cumulative.append(seq_len_last_value + len(radar_poses))
        seq_len_last_value = seq_len_cumulative[-1]

    seq_len_cumulative.insert(0, 0)

    # Generate Validation files for each sequence
    for i in range(len(seq_len_cumulative) - 1):
        start_idx = seq_len_cumulative[i]
        end_idx = seq_len_cumulative[i + 1]

        val_data = {}
        for idx in range(start_idx, end_idx):
            val_data[all_radar_stamps[idx]] = {"seq": all_sequences[idx], 'pose': all_lidar_poses[idx].astype(float).tolist(), 'gps_pos': [all_lidar_poses[idx, 0], all_lidar_poses[idx, 1]]}
                
        with open(f'{dataroot}/val_data_seq{i}.json', 'w') as f:
            json.dump(val_data, f)