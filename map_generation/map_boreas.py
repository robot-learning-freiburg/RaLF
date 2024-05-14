from pyboreas.utils.utils import yawPitchRollToRot
from pyboreas.utils.utils import load_lidar, get_transform
from pyboreas.utils.odometry import *

from utils.utils import *
from utils.config import boreas_dataroot


# Function to extract lidar poses and stamps from the sequence data

def get_lidar_poses(dataroot, seq_dir):
    poses = os.path.join(dataroot, seq_dir, 'applanix', 'lidar_poses.csv')
    lidar_poses = []
    stamps = []
    first_idx = True
    with open(poses, 'r') as f:
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
            # pose[0, 3] -= offset_x
            # pose[1, 3] -= offset_y 
            pose[2, 3] -= offset_z

            lidar_poses.append(pose)
            
    return lidar_poses, stamps


# Function for map construction

def create_map(dataroot, seq_dir, fwd_range, side_range):

    # Placeholder to store map array and intensity array
    seq_map = None
    i_map = None

    # Parameter to downsample point cloud to save memory and for faster map access
    downsample_k = 50

    # Get lidar poses and stamps
    lidar_poses, stamps = get_lidar_poses(dataroot, seq_dir, )

    for idx in tqdm(range(0, len(lidar_poses), 5)):
        gt_pose = lidar_poses[idx]
        lidar_file_path = f"{dataroot}/{seq_dir}/lidar/{str(stamps[idx])}.bin"        
    
        scan = load_lidar(lidar_file_path)
        z_points = scan[:, 2]
        indices = np.where((z_points > 0.2) & (z_points < 0.8))[0]
        scan = scan[indices, :]
    
        scan = gt_pose @ np.hstack((scan[:, 0:3], np.ones(scan.shape[0])[:, np.newaxis])).T
        scan = scan.T
        #print(scan.shape[0])
    
        x_points = scan[:, 0]
        y_points = scan[:, 1]
        z_points = scan[:, 2]
        i_points = scan[:, 3]
    
        # ipdb.set_trace()
        x, y = lidar_poses[idx][:2,3]
        f = (x + fwd_range[0], x + fwd_range[1])
        s = (y + side_range[0], y + side_range[1])
        # f = (fwd_range[0], fwd_range[1])
        # s = (side_range[0], side_range[1])
        # ipdb.set_trace()
    
        f_filt = np.logical_and((x_points > f[0]), (x_points < f[1]))
        s_filt = np.logical_and((y_points > s[0]), (y_points < s[1]))
        # i_filt = (i_points >= 10)
        ego_filt_f = np.logical_or((x_points < x-2.0), (x_points >  x+2.0))
        ego_filt_s = np.logical_or((y_points < y-2.0), (y_points > y+2.0))
        # z_filt = np.logical_and((z_points > -5.0), (z_points < 5.0))
        filter = np.logical_and(f_filt, s_filt)
        filter = np.logical_and(filter, ego_filt_f)
        filter = np.logical_and(filter, ego_filt_s)
        # filter = np.logical_and(filter, i_filt)
        indices = np.argwhere(filter).flatten()
    
        scan_subset = scan[indices, :]
        i_points = i_points[indices]
        #print(f"scan subset shape: {scan_subset.shape}")
    
    
        if seq_map is None:
            # seq_map = scan_subset[:, 0:3] 
            # i_map = i_points[:, np.newaxis]
            seq_map = scan_subset[0:scan_subset.shape[0]:downsample_k, 0:3]
            i_map = i_points[0:i_points.shape[0]:downsample_k, np.newaxis]
    
        else:
            # seq_map = np.vstack((seq_map, scan_subset[:, 0:3]))
            # i_map = np.vstack((i_map, i_points[:, np.newaxis]))
            i_map = np.vstack((i_map, i_points[0:i_points.shape[0]:downsample_k, np.newaxis]))
            seq_map = np.vstack((seq_map, scan_subset[0:scan_subset.shape[0]:downsample_k, 0:3]))
    
    return seq_map, i_map


if __name__ == '__main__':
    # Check the config.py file in the utils directory! Set the boreas_dataroot in the file
    dataroot = boreas_dataroot
    
    sequences = [s for s in os.listdir(dataroot) if osp.isdir(osp.join(dataroot, s))]

    side_range = (-100, 100)  # left-most to right-most
    fwd_range = (-100, 100)  # back-most to forward-most
    save_as_pcd = False

    for seq_dir in sequences:
        
        map_pts, i_map = create_map(dataroot, seq_dir, fwd_range, side_range)
        print(f"Final map shape: {map_pts.shape}, {i_map.shape}")

        map_with_intensity = np.hstack((map_pts, i_map))
        np.savez(f"{dataroot}/{seq_dir}/downsampled_map.npz", map_with_intensity)

        if save_as_pcd:            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(map_pts[:, 0:3])
            pcd.colors = o3d.utility.Vector3dVector(np.repeat(i_map, 3, axis=1))

            o3d.io.write_point_cloud(f"{dataroot}/{seq_dir}/map_intensity.pcd", pcd)
