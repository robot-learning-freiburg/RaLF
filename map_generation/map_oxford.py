from utils.utils import *
from utils.interpolate_poses import *
from utils.lidar import *
from utils.config import ox_dataroot, ox_extr_dir


# Synchronize radar and lidar timestamps. Sample lidar timestamps(higher data collection frequency)

def sample_velodyne_timestamps(radar_fp, vl_fp, vr_fp):
    radar_file = open(radar_fp, 'r')
    velodyne_right_file = open(vr_fp, 'r')
    velodyne_left_file = open(vl_fp, 'r')

    radar_lines = radar_file.readlines()
    velodyne_right_lines = velodyne_right_file.readlines()
    velodyne_left_lines = velodyne_left_file.readlines()

    radar_time, vr_time, vl_time = [], [], []

    for line in radar_lines:
        radar_time.append(int(line.strip().split(" ")[0]))

    for line in velodyne_right_lines:
        vr_time.append(int(line.strip().split(" ")[0]))

    for line in velodyne_left_lines:
        vl_time.append(int(line.strip().split(" ")[0]))

    radar_time = np.array(radar_time)
    vr_time = np.array(vr_time)
    vl_time = np.array(vl_time)

    vr_closest_times = []
    vl_closest_times = []

    for time in radar_time:
        vr_closest_times.append(find_nearest(vr_time, time))
        vl_closest_times.append(find_nearest(vl_time, time))

    vr_closest_times = list(vr_closest_times)
    vl_closest_times = list(vl_closest_times)

    return vr_closest_times, vl_closest_times


# Get the point clouds from the left and right sensors

def extract_ptcld(vl_path, vr_path):
    side_range = (-100, 100)  # left-most to right-most
    fwd_range = (-100, 100)  # back-most to forward-most
    vl_ptcld, vr_ptcld = return_ptcld(vl_path, vr_path, side_range, fwd_range)

    return vl_ptcld, vr_ptcld


# Construct the map -----> Transform point clouds to global coordinates and concatenate

def create_map(vr_time, vl_time, vr_loc, vl_loc, gt_pose_file, extrinsics_dir):

    # Interpolate poses for the required timestamps. For the left and right velodyne
    gt_poses_vr = interpolate_vo_poses(gt_pose_file, vr_time, vr_time[0])
    gt_poses_vl = interpolate_vo_poses(gt_pose_file, vl_time, vl_time[0])

    # Extract sensor transformation matrices
    with open(os.path.join(extrinsics_dir, 'velodyne_left.txt')) as extrinsics_file:
        extrinsics = next(extrinsics_file)
    G_posesource_vl = build_se3_transform([float(x) for x in extrinsics.split(' ')])

    with open(os.path.join(extrinsics_dir, 'velodyne_right.txt')) as extrinsics_file:
        extrinsics = next(extrinsics_file)
    G_posesource_vr = build_se3_transform([float(x) for x in extrinsics.split(' ')])

    # LIDAR to RADAR Transformation
    with open(os.path.join(extrinsics_dir, 'radar.txt')) as extrinsics_file:
        extrinsics = next(extrinsics_file)
        G_posesource_vl = np.matmul(np.linalg.inv(build_se3_transform([float(x) for x in extrinsics.split(' ')])), G_posesource_vl)

        G_posesource_vr = np.matmul(np.linalg.inv(build_se3_transform([float(x) for x in extrinsics.split(' ')])), G_posesource_vr)

    # Placeholder for map data array
    map = None

    for idx in tqdm(range(len(gt_poses_vl))):
        # Process Velodyne
        vl_im_path = os.path.join(vl_loc, f"{vl_time[idx+1]}.png")
        vr_im_path = os.path.join(vr_loc, f"{vr_time[idx+1]}.png")

        vl_ptcld, vr_ptcld = extract_ptcld(vl_im_path, vr_im_path)

        # Save point intensities for future use
        i_l = vl_ptcld[3, :]
        i_r = vr_ptcld[3, :]

        # Transform lidar points from local to the global coordinate system
        scan_l = np.dot(np.dot(gt_poses_vl[idx], G_posesource_vl),
                        np.vstack((vl_ptcld[0:3, :], np.ones(vl_ptcld.shape[1]))))
        scan_r = np.dot(np.dot(gt_poses_vr[idx], G_posesource_vr),
                        np.vstack((vr_ptcld[0:3, :], np.ones(vr_ptcld.shape[1]))))
        
        # Concatenate point clouds from the two sensors
        scan = np.hstack((scan_l, scan_r))

        # Append point intensities
        scan[3, :] = np.hstack((i_l, i_r))

        # Save map array
        if map is None:
            map = scan
        else:
            map = np.hstack((map, scan))

    return map


if __name__ == '__main__':
    # Check the config.py file in the utils directory! Set the oxford dataroot and extrinsics directory path in the file
    dataroot = ox_dataroot
    extrinsics_dir = ox_extr_dir

    sequences = ['2019-01-18-12-42-34-radar-oxford-10k', '2019-01-18-14-14-42-radar-oxford-10k',
                 '2019-01-18-14-46-59-radar-oxford-10k', '2019-01-18-15-20-12-radar-oxford-10k']

    for seq_dir in sequences:
        print(f"Processing sequence: {seq_dir}")
        
        # Get Radar timestamps
        fp = open(f"{seq_dir}/radar.timestamps")
        timestamps = [int(time.strip().split(' ')[0]) for time in fp.readlines()]

        # Plot Trajectory
        plot = False
        if plot:
            poses = interpolate_vo_poses(f'{seq_dir}/gt/radar_odometry.csv', timestamps, timestamps[0])
            x_poses = [pose[0, 3] for pose in poses]
            y_poses = [pose[1, 3] for pose in poses]
            plt.plot(x_poses, y_poses)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title("Absolute pose")
            plt.show()

        # Set these if you want to construct the map or just visualize a map
        construct_map = True
        visualize = False

        if construct_map:
            radar_fp = f'{seq_dir}/radar.timestamps'
            velodyne_right_fp = f'{seq_dir}/velodyne_right.timestamps'
            velodyne_left_fp = f'{seq_dir}/velodyne_left.timestamps'

            # Read Velodyne timestamps closest to radar timestamps
            vr_time, vl_time = sample_velodyne_timestamps(radar_fp, velodyne_left_fp, velodyne_right_fp)

            # Create the map
            vl_loc = f'{seq_dir}/velodyne_left'
            vr_loc = f'{seq_dir}/velodyne_right'
            gt_pose_file = f'{seq_dir}/gt/radar_odometry.csv'
            

            # Shape (4 , N) - (x, y, z, i)
            map = create_map(vr_time, vl_time, vr_loc, vl_loc, gt_pose_file, extrinsics_dir)

            # Shape (N , 4)
            map = np.array(map.transpose())

            # Save Map
            map_file = 'map_final.npz'
            np.savez(f"{seq_dir}/{map_file}", map)

            # Also save a downsampled version to save memory and for easier access
            # Save every kth point from the point cloud
            save_every = 50
            downsampled_map = map[0:map.shape[0]: save_every]

            np.savez(f'{seq_dir}/downsampled_map.npz', downsampled_map)

        else:
            map_file = f'{seq_dir}/downsampled_map.npz'

            # Shape (N * 4) - (x, y, z, i)
            map = np.load(map_file)['arr_0']

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(map[:, 0:3])   

        # Visualize the stored or created map
        if visualize:
            o3d.visualization.draw_geometries([pcd],
                                            zoom=0.3412,
                                            front=[0.4257, -0.2125, -0.8795],
                                            lookat=[np.mean(map[:, 0]), np.mean(map[:, 1]), np.mean(map[:, 2])],
                                            up=[-0.0694, -0.9768, 0.2024])
