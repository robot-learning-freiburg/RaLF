from utils.interpolate_poses import *
from utils.utils import *
from utils.config import kaist_dataroot


def extract_time_from_file(time_file):
    stamps = []
    with open(time_file, 'r') as f:
        for x in f:
            x = x.strip().split(',')
            stamps.append(int(x[0]))
    return stamps

def closest_timestamps(t1, t2):
    t1 = np.array(t1)
    t2 = np.array(t2)

    result_stamps = []

    for time in t1:
        nearest_idx = find_nearest_idx(t2, time)
        result_stamps.append(t2[nearest_idx])

    return result_stamps

def closest_timestamps_poses(t1, t2, all_poses):
    t1 = np.array(t1)
    t2 = np.array(t2)

    closest_stamps = []
    closest_poses = []

    for time in t1:
        nearest_idx = find_nearest_idx(t2, time)
        closest_stamps.append(t2[nearest_idx])
        closest_poses.append(all_poses[nearest_idx])

    return closest_poses, closest_stamps

def extract_poses_timestamps_from_file(pose_file):
    pose_stamps = []
    poses = []

    with open(pose_file, 'r') as f_obj:
        for val in f_obj:
            values = val.strip().split(",")
            t = int(values[0])

            p = np.array([float(v) for v in values[1:]])
            scan_pose = np.zeros((4, 4))
            scan_pose[0:3, 0:4] = np.reshape(p, (3, 4))
            scan_pose[3, 3] = 1.0

            poses.append(scan_pose)
            pose_stamps.append(t)

    return poses, pose_stamps


if __name__ == '__main__':
    dataroot = kaist_dataroot

    sequences = [item for item in sorted(os.listdir(dataroot)) if os.path.isdir(os.path.join(dataroot, item))]

    gps_poses = []
    radar_odom_gt_poses = []
    seq_len_cumulative = []
    seq_len_last_value = 0
    all_poses = []
    all_lidar_poses = []
    all_radar_stamps = []
    all_sequences = []

    northings = []
    eastings = []
    validation_filters = None

    p1=[353168.25, 4026302.4]
    wx, wy = 200 / 2.0, 200 / 2.0

    p2 = [353526.8, 4026080.7]
    wx2, wy2 = 200.0 / 2, 200.0 / 2

    for seq in sequences:
        seq_path = osp.join(dataroot, seq)
        print(seq_path)
        radar_time_file = osp.join(seq_path, 'navtech_top_stamp.csv')
        lidar_time_file = osp.join(seq_path, 'ouster_front_stamp.csv')

        global_pose_file = osp.join(seq_path, 'global_pose.csv')

        radar_stamps = extract_time_from_file(radar_time_file)
        lidar_stamps = extract_time_from_file(lidar_time_file)

        poses, pose_stamps = extract_poses_timestamps_from_file(global_pose_file)

        closest_lidar_stamps = closest_timestamps(radar_stamps, lidar_stamps)

        poses, pose_stamps = closest_timestamps_poses(closest_lidar_stamps, pose_stamps, poses)

        poses_x = [pose[0, 3] for pose in poses]
        poses_y = [pose[1, 3] for pose in poses]

        all_poses += poses
        all_radar_stamps += radar_stamps

        northings += poses_x
        eastings += poses_y

        all_sequences += [seq] * len(poses)

        seq_len_cumulative.append(seq_len_last_value + len(poses))
        seq_len_last_value = seq_len_cumulative[-1]

        poses_x = np.array(poses_x)
        poses_y = np.array(poses_y)

        val_filter1 = get_validation_filter(poses_x, poses_y, p1[0] - wx, p1[0] + wx, p1[1] - wy, p1[1] + wy)
        val_filter2 = get_validation_filter(poses_x, poses_y, p2[0] - wx2, p2[0] + wx2, p2[1] - wy2, p2[1] + wy2)

        vfilter = val_filter1 + val_filter2 

        if validation_filters is None:
            validation_filters = vfilter
        else:
            validation_filters = np.hstack((validation_filters, vfilter))
    
    train_data = {}

    for idx in range(len(all_radar_stamps)):
        if validation_filters[idx]:
            continue
        train_data[all_radar_stamps[idx]] = {"seq": all_sequences[idx], 'pose': all_poses[idx].astype(float).tolist(), 'gps_pos': [northings[idx], eastings[idx]]}

    with open(f'{dataroot}/train_data.json', 'w') as f:
        json.dump(train_data, f)

    seq_len_cumulative.insert(0, 0)

    for i in range(len(seq_len_cumulative) - 1):
        start_idx = seq_len_cumulative[i]
        end_idx = seq_len_cumulative[i + 1]

        val_data = {}
        last_pose = None

        for idx in range(start_idx, end_idx):
            if not validation_filters[idx]:
                continue
            if last_pose is not None and np.linalg.norm(last_pose - all_poses[idx]) < 0.2:
                continue
            last_pose = all_poses[idx]
            n, e = northings[idx], eastings[idx]
            val_data[all_radar_stamps[idx]] = {"seq": all_sequences[idx], 'pose': all_poses[idx].astype(float).tolist(),
                                                'gps_pos': [n, e]}

        with open(f'{dataroot}/val_data_seq{i}.json', 'w') as f:
            json.dump(val_data, f)