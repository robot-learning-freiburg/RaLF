import numpy as np
import json
import os

try:
    from interpolate_poses import *
except:
    from data_utils.interpolate_poses import *
import ipdb
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Set the dataroot, query and map file names correctly
    dataroot = '/data/radar_robotcar_sequences'
    query_file = 'val_data_seq1.json'
    map_file = 'val_data_seq2.json'

    map_poses = []
    query_poses = []

    with open(f"{dataroot}/{query_file}", 'r') as f:
        val_query = json.load(f)

    with open(f"{dataroot}/{map_file}", 'r') as f:
        val_map = json.load(f)

    for key, value in val_query.items():
        query_poses.append(value['gps_pos'])

    for key, value in val_map.items():
        map_poses.append(value['gps_pos'])

    map_poses = np.array(map_poses)
    query_poses = np.array(query_poses)

    plt.scatter(query_poses[:, 0], query_poses[:, 1], s=0.1, label='query')
    plt.scatter(map_poses[:, 0], map_poses[:, 1], s=0.1, label='map')
    plt.legend()
    plt.show()

    count = 0
    dist_threshold = 3.0

    min_samples = np.inf

    for pose in query_poses:
        diff = np.linalg.norm(pose - map_poses, axis=1)
        indices = np.where(diff < dist_threshold)[0]

        min_samples = min(min_samples, len(indices))
        
        if len(indices):
            # Atleast one sample available
            count += 1
    
    print(f"Min Samples: {min_samples}")
    
    print(f"Total queries: {len(query_poses)}, Queries with map samples within {dist_threshold}m: {count}")