import numpy as np
import json
import os
from interpolate_poses import *
import ipdb
import os.path as osp
import matplotlib.pyplot as plt

def extract_radar_odom_poses(odom_file, rtime_file):
    fp = open(rtime_file, 'r')
    timestamps = [int(time.strip().split(' ')[0]) for time in fp.readlines()]

    gt_poses = interpolate_vo_poses(odom_file, timestamps, timestamps[0])
    print(rtime_file)

    assert len(gt_poses) == len(timestamps[1:])
    
    return gt_poses, timestamps[1:]

def get_validation_filter(pose_x, pose_y, xmin, xmax, ymin, ymax):
    fx1 = pose_x >= xmin
    fx2 = pose_x <= xmax
    fy1 = pose_y >= ymin 
    fy2 = pose_y <= ymax

    val_filter = fx1 * fx2 * fy1 * fy2

    return val_filter

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def extract_gps_poses(gps_file, ins_file, rtime_file, use_rtk=False):
    fp = open(rtime_file, 'r')
    timestamps = [int(time.strip().split(' ')[0]) for time in fp.readlines()]
    
    northings = []
    eastings = []
    gps_time = []

    fp = open(gps_file, 'r')
    header = fp.readline()

    for row in fp.readlines():
        row_values = row.strip().split(',')
        northings.append(float(row_values[-4]))
        eastings.append(float(row_values[-3]))
        gps_time.append(int(row_values[0]))

    gps_closest_indices = []

    for radar_time in timestamps:
        gps_closest_indices.append(find_nearest(gps_time, radar_time))
    
    northings = np.array(northings)
    eastings = np.array(eastings)

    n = northings[gps_closest_indices]
    e = eastings[gps_closest_indices]

    gt_gps_poses, _ = interpolate_ins_poses(ins_file, timestamps, timestamps[0], use_rtk=use_rtk)
    return n, e, gt_gps_poses


if __name__ == "__main__":

    # Set the correct dataroot path here. This folder should contain the sequences that you want to process
    dataroot = '/data/radar_robotcar_sequences'

    sequences = sorted([f for f in os.listdir(dataroot) if os.path.isdir(os.path.join(dataroot, f))])
    
    all_poses = []
    all_times = []
    all_sequences = []
    seq_len_cumulative = []
    all_n = []
    all_e = []
    seq_len_last_value = 0

    validation_filters = None

    # Validation region in WGS coordinates
    p1 = [5735375.5, 620014.5]
    wx, wy = 351 / 2.0, 240 / 2.0

    for f in sequences:
        sequence_path = os.path.join(dataroot, f)
        radar_time_file = os.path.join(sequence_path, 'radar.timestamps')
        radar_odom_file = os.path.join(sequence_path, 'gt/radar_odometry.csv')
        gps_file_path = osp.join(sequence_path, 'gps/gps.csv')
        ins_file_path = osp.join(sequence_path, 'gps/ins.csv')

        # Extract Northings, Eastings and gps poses
        n, e, seq_gps_poses = extract_gps_poses(gps_file_path, ins_file_path, radar_time_file, use_rtk=False)

        seq_poses, timestamps = extract_radar_odom_poses(radar_odom_file, radar_time_file)

        # Accumulate all data in these lists
        all_poses += seq_poses
        all_times += timestamps
        all_sequences += [f] * len(seq_poses)
        all_n += list(n)
        all_e += list(e)

        seq_len_cumulative.append(seq_len_last_value + len(seq_poses))
        seq_len_last_value = seq_len_cumulative[-1]

        # Define Validation filter based on the region
        val_filter1 = get_validation_filter(n, e, p1[0] - wx, p1[0] + wx, p1[1] - wy, p1[1] + wy)

        vfilter = val_filter1 
        if validation_filters is None:
            validation_filters = vfilter
        else:
            validation_filters = np.hstack((validation_filters, vfilter))

        # Train and Validation regions
        train_idx = np.where(vfilter == False)[0]
        val_idx = np.where(vfilter == True)[0]

    train_data = {}
    last_pose = None
    
    for idx in range(len(all_times)):
        if validation_filters[idx] == True:
            continue
        else:
            if last_pose is not None and np.linalg.norm(last_pose - all_poses[idx]) < 0.2:
                # Should have moved a certain distance to store sample as valid data
                continue
            last_pose = all_poses[idx]

            # Accumulate dictionary with the sequence and pose info
            train_data[all_times[idx]] = {"seq": all_sequences[idx], 'pose': all_poses[idx].astype(float).tolist(), 'gps_pos': [all_n[idx], all_e[idx]]}

    # Store train_data dictionary in the json file
    with open(f'{dataroot}/train_data.json', 'w') as f:
        json.dump(train_data, f)

    seq_len_cumulative.insert(0, 0)

    # Generate Validation files for each sequence
    for i in range(len(seq_len_cumulative) - 1):
        start_idx = seq_len_cumulative[i]
        end_idx = seq_len_cumulative[i + 1]

        val_data = {}
        for idx in range(start_idx, end_idx):
            if validation_filters[idx] == True:
                val_data[all_times[idx]] = {"seq": all_sequences[idx], 'pose': all_poses[idx].astype(float).tolist(), 'gps_pos': [all_n[idx], all_e[idx]]}
                
        with open(f'{dataroot}/val_data_seq{i}.json', 'w') as f:
            json.dump(val_data, f)