from utils.utils import *
from utils.config import kaist_dataroot


# Some transformation parameters for KAIST
se3_6d = [1.7042, -0.021, 1.8047, 0.0001, 0.0003, 179.6654]

trans = se3_6d[0:3]
roll = se3_6d[3]
pitch = se3_6d[4]
yaw = se3_6d[5]

lidar_to_base_init_se3 = np.array([[-1.0000, -0.0058, -0.0000, 1.7042],
                                   [0.0058, -1.0000, 0.0000, -0.0210],
                                   [-0.0000, 0.0000, 1.0000, 1.8047],
                                   [0, 0, 0, 1.0000]])

radar_to_base_init_se3 = np.array([[ 0.99987663, -0.01570732, 0.0, 1.5],
                                   [ 0.01570732, 0.99987663, 0.0, -0.04],
                                   [ 0.0, 0.0, 1.0, 1.97],
                                   [ 0.0, 0.0, 0.0, 1.0]])


def closest_gt_timestamps(gt_time, lidar_time, lidar_all_poses):
    gt_time = np.array(gt_time)
    lidar_time = np.array(lidar_time)

    closest_times = []
    lidar_poses = []

    for time in lidar_time:
        nearest_idx = find_nearest_idx(gt_time, time)
        closest_times.append(gt_time[nearest_idx])
        lidar_poses.append(lidar_all_poses[nearest_idx])

    return closest_times, lidar_poses


def readBin(bin_path):
    points = np.fromfile(bin_path, dtype=np.float32)
    points = points.reshape(-1, 4)

    scan = points[:, 0:3]
    intensities = points[:, 3]

    return scan, intensities


if __name__ == '__main__':
    # Check the config.py file in the utils directory! Set the kaist_dataroot in the file
    dataroot = kaist_dataroot

    sequences = sorted([os.path.join(dataroot, f) for f in os.listdir(dataroot) if os.path.isdir(os.path.join(dataroot, f))])

    for seq_dir in sequences:
        print(f"Processing sequence: {seq_dir}")

        pose_file = f"{seq_dir}/global_pose.csv"
        lidar_path = f"{seq_dir}/Ouster"
        lidar_time_file = f"{seq_dir}/ouster_front_stamp.csv"

        timestamps = []
        lidar_stamps = []
        poses = []
        seq_map = None


        with open(pose_file, 'r') as f_obj:
            for val in f_obj:
                values = val.strip().split(",")
                t = int(values[0])

                p = np.array([float(v) for v in values[1:]])
                scan_pose = np.zeros((4, 4))
                scan_pose[0:3, 0:4] = np.reshape(p, (3, 4))
                scan_pose[3, 3] = 1.0

                poses.append(scan_pose)
                timestamps.append(t)

        with open(lidar_time_file, 'r') as f:
            f.readline()
            for t in f:
                lidar_stamps.append(int(t.strip()))

        closest_stamps, lidar_poses = closest_gt_timestamps(timestamps, lidar_stamps, poses)

        for idx, time in enumerate(tqdm(lidar_stamps)):
            if idx % 2 != 0:
                continue

            bin_path = f"{lidar_path}/{time}.bin"
            scan, i_pts = readBin(bin_path)

            indices = np.where(scan[:, 2] > 1.5)[0]
            scan = scan[indices]
            i_pts = i_pts[indices]

            scan_global = lidar_poses[idx] @ lidar_to_base_init_se3 @ np.hstack((scan[:, 0:3], np.ones(scan.shape[0])[:, np.newaxis])).T
            scan_global = scan_global.T

            save_every = 50

            if seq_map is None:
                seq_map = scan_global[0:scan_global.shape[0]:save_every, 0:3]
                i_map = i_pts[0:i_pts.shape[0]:save_every, np.newaxis]

            else:
                i_map = np.vstack((i_map, i_pts[0:i_pts.shape[0]:save_every, np.newaxis]))
                seq_map = np.vstack((seq_map, scan_global[0:scan_global.shape[0]:save_every, 0:3]))

        ptcld = np.hstack((seq_map, i_map))

        np.savez(f"{seq_dir}/downsampled_map.npz", ptcld)
