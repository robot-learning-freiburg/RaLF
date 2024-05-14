################################################################################
#
# Copyright (c) 2017 University of Oxford
# Authors:
#  Dan Barnes (dbarnes@robots.ox.ac.uk)
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
###############################################################################

from typing import AnyStr
import numpy as np
import os
import cv2
import open3d as o3d
import ipdb

# Hard coded configuration to simplify parsing code
hdl32e_range_resolution = 0.002  # m / pixel
hdl32e_minimum_range = 1.0
hdl32e_elevations = np.array([-0.1862, -0.1628, -0.1396, -0.1164, -0.0930,
                              -0.0698, -0.0466, -0.0232, 0., 0.0232, 0.0466, 0.0698,
                              0.0930, 0.1164, 0.1396, 0.1628, 0.1862, 0.2094, 0.2327,
                              0.2560, 0.2793, 0.3025, 0.3259, 0.3491, 0.3723, 0.3957,
                              0.4189, 0.4421, 0.4655, 0.4887, 0.5119, 0.5353])[:, np.newaxis]
hdl32e_base_to_fire_height = 0.090805
hdl32e_cos_elevations = np.cos(hdl32e_elevations)
hdl32e_sin_elevations = np.sin(hdl32e_elevations)

def load_velodyne_binary(velodyne_bin_path: AnyStr):
    """Decode a binary Velodyne example (of the form '<timestamp>.bin')
    Args:
        example_path (AnyStr): Oxford Radar RobotCar Dataset binary Velodyne pointcloud example path
    Returns:
        ptcld (np.ndarray): XYZI pointcloud from the binary Velodyne data_utils Nx4
    Notes:
        - The pre computed points are *NOT* motion compensated.
        - Converting a raw velodyne scan to pointcloud can be done using the
            `velodyne_ranges_intensities_angles_to_pointcloud` function.
    """
    ext = os.path.splitext(velodyne_bin_path)[1]
    if ext != ".bin":
        raise RuntimeError("Velodyne binary pointcloud file should have `.bin` extension but had: {}".format(ext))
    if not os.path.isfile(velodyne_bin_path):
        raise FileNotFoundError("Could not find velodyne bin example: {}".format(velodyne_bin_path))
    data = np.fromfile(velodyne_bin_path, dtype=np.float32)
    ptcld = data.reshape((4, -1))
    return ptcld

def load_velodyne_raw(velodyne_raw_path: AnyStr):
    """Decode a raw Velodyne example. (of the form '<timestamp>.png')
    Args:
        example_path (AnyStr): Oxford Radar RobotCar Dataset raw Velodyne example path
    Returns:
        ranges (np.ndarray): Range of each measurement in meters where 0 == invalid, (32 x N)
        intensities (np.ndarray): Intensity of each measurement where 0 == invalid, (32 x N)
        angles (np.ndarray): Angle of each measurement in radians (1 x N)
        approximate_timestamps (np.ndarray): Approximate linearly interpolated timestamps of each mesaurement (1 x N).
            Approximate as we only receive timestamps for each packet. The timestamp of the next frame will was used to
            interpolate the last packet timestamps. If there was no next frame, the last packet timestamps was
            extrapolated. The original packet timestamps can be recovered with:
                approximate_timestamps(:, 1:12:end) (12 is the number of azimuth returns in each packet)
     Notes:
       Reference: https://velodynelidar.com/lidar/products/manual/63-9113%20HDL-32E%20manual_Rev%20E_NOV2012.pdf
    """
    ext = os.path.splitext(velodyne_raw_path)[1]
    if ext != ".png":
        raise RuntimeError("Velodyne raw file should have `.png` extension but had: {}".format(ext))
    if not os.path.isfile(velodyne_raw_path):
        raise FileNotFoundError("Could not find velodyne raw example: {}".format(velodyne_raw_path))
    example = cv2.imread(velodyne_raw_path, cv2.IMREAD_GRAYSCALE)
    intensities, ranges_raw, angles_raw, timestamps_raw = np.array_split(example, [32, 96, 98], 0)
    ranges = np.ascontiguousarray(ranges_raw.transpose()).view(np.uint16).transpose()
    ranges = ranges * hdl32e_range_resolution
    angles = np.ascontiguousarray(angles_raw.transpose()).view(np.uint16).transpose()
    angles = angles * (2. * np.pi) / 36000
    approximate_timestamps = np.ascontiguousarray(timestamps_raw.transpose()).view(np.int64).transpose()
    return ranges, intensities, angles, approximate_timestamps

def velodyne_raw_to_pointcloud(ranges: np.ndarray, intensities: np.ndarray, angles: np.ndarray):
    """ Convert raw Velodyne data_utils (from load_velodyne_raw) into a pointcloud
    Args:
        ranges (np.ndarray): Raw Velodyne range readings
        intensities (np.ndarray): Raw Velodyne intensity readings
        angles (np.ndarray): Raw Velodyne angles
    Returns:
        pointcloud (np.ndarray): XYZI pointcloud generated from the raw Velodyne data_utils Nx4

    Notes:
        - This implementation does *NOT* perform motion compensation on the generated pointcloud.
        - Accessing the pointclouds in binary form via `load_velodyne_pointcloud` is approximately 2x faster at the cost
            of 8x the storage space
    """
    valid = ranges > hdl32e_minimum_range
    z = hdl32e_sin_elevations * ranges - hdl32e_base_to_fire_height
    xy = hdl32e_cos_elevations * ranges
    x = np.sin(angles) * xy
    y = -np.cos(angles) * xy

    xf = x[valid].reshape(-1)
    yf = y[valid].reshape(-1)
    zf = z[valid].reshape(-1)
    intensityf = intensities[valid].reshape(-1).astype(np.float32)
    ptcld = np.stack((xf, yf, zf, intensityf), 0)
    return ptcld

def remove_ground_plane(ptcld):
    scan = ptcld.transpose()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scan[:, 0:3])
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.2,
                                             ransac_n=3,
                                             num_iterations=1000)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)

    i = 0
    while np.argmax(plane_model[:-1]) != 2:
        i += 1
        pcd = pcd.select_by_index(inliers, invert=True)
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.2,
                                                 ransac_n=3,
                                                 num_iterations=10000)

    outliers_index = set(range(scan.shape[0])) - set(inliers)
    outliers_index = list(outliers_index)
    no_ground_scan = scan[outliers_index]

    no_ground_scan = np.transpose(no_ground_scan)

    return no_ground_scan

def return_ptcld(vl_path, vr_path, side_range, fwd_range):
    ranges, intensities, angles, approximate_timestamps = load_velodyne_raw(vl_path)
    vl_ptcld = velodyne_raw_to_pointcloud(ranges, intensities, angles)

    ranges, intensities, angles, approximate_timestamps = load_velodyne_raw(vr_path)
    vr_ptcld = velodyne_raw_to_pointcloud(ranges, intensities, angles)

    # Add offsets according to dataset
    # vr_ptcld = vr_ptcld + np.array([[0.1], [-0.47], [0.0], [0.0]])
    # vl_ptcld = vl_ptcld + np.array([[0.1], [0.47], [0.0], [0.0]])

    ptcld = np.hstack((vl_ptcld, vr_ptcld))

    # send full point cloud
    vr_ptcld = remove_ground_plane(vr_ptcld)
    vl_ptcld = remove_ground_plane(vl_ptcld)

    # EXTRACT THE POINTS FOR EACH AXIS
    xr_points, yr_points, zr_points, ir_points = vr_ptcld[0], vr_ptcld[1], vr_ptcld[2], vr_ptcld[3]
    xl_points, yl_points, zl_points, il_points = vl_ptcld[0], vl_ptcld[1], vl_ptcld[2], vl_ptcld[3]

    # y_points = points[1]
    # z_points = points[2]
    # i_points = points[3]

    # FILTER - To return only indices of points within desired cube
    # Three filters for: Front-to-back, side-to-side, and height ranges
    f_filt = np.logical_and((xl_points > fwd_range[0]), (xl_points < fwd_range[1]))
    s_filt = np.logical_and((yl_points > side_range[0]), (yl_points < side_range[1]))
    i_filt = (il_points >= 10)
    ego_filt_f = np.logical_or((xl_points < -2.0), (xl_points > 2.0))
    ego_filt_s = np.logical_or((yl_points < -2.0), (yl_points > 2.0))
    z_filt = np.logical_and((zl_points > -5.0), (zl_points < 5.0))
    filter = np.logical_and(f_filt, s_filt, ego_filt_s)
    filter = np.logical_and(filter, ego_filt_f)
    filter = np.logical_and(filter, z_filt)
    filter = np.logical_and(filter, i_filt)
    l_indices = np.argwhere(filter).flatten()

    f_filt = np.logical_and((xr_points > fwd_range[0]), (xr_points < fwd_range[1]))
    s_filt = np.logical_and((yr_points > -side_range[1]), (yr_points < -side_range[0]))
    i_filt = (ir_points >= 10)
    ego_filt_f = np.logical_or((xr_points < -2.0), (xr_points > 2.0))
    ego_filt_s = np.logical_or((yr_points < -2.0), (yr_points > 2.0))

    z_filt = np.logical_and((zr_points > -5.0), (zr_points < 5.0))

    filter = np.logical_and(f_filt, s_filt, ego_filt_s)
    filter = np.logical_and(filter, ego_filt_f)
    filter = np.logical_and(filter, z_filt)
    filter = np.logical_and(filter, i_filt)
    r_indices = np.argwhere(filter).flatten()

    return vl_ptcld[:, l_indices], vr_ptcld[:, r_indices]

def post_process_lidar(vl_path, vr_path):
    side_range = (-60, 60)  # left-most to right-most
    fwd_range = (-60, 60)  # back-most to forward-most

    points = return_ptcld(vl_path, vr_path, side_range, fwd_range)
    ipdb.set_trace()
    x_points = points[0]
    y_points = points[1]
    z_points = points[2]
    i_points = points[3]

    res = 0.2
    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (-y_points / res).astype(np.int32)  # x axis is -y in LIDAR
    y_img = (x_points / res).astype(np.int32)  # y axis is -x in LIDAR

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor and ceil used to prevent anything being rounded to below 0 after shift
    x_img -= int(np.floor(side_range[0] / res))
    y_img += int(np.ceil(fwd_range[1] / res))

    i_points[i_points <= 10] = 0.0
    i_points[i_points > 10] = 255.0

    # CLIP
    # pixel_values = np.clip(i_points, a_min=50.0, a_max=150.0)

    def scale_to_255(a, min, max, dtype=np.uint8):
        """ Scales an array of values from specified min, max range to 0-255
            Optionally specify the data_utils type of the output (default is uint8)
        """
        return (((a - min) / float(max - min)) * 255).astype(dtype)

    # RESCALE THE HEIGHT VALUES - to be between the range 0-255
    pixel_values = i_points #scale_to_255(pixel_values, min=min(i_points), max=max(i_points))

    # INITIALIZE EMPTY ARRAY - of the dimensions we want
    x_max = 1 + int((side_range[1] - side_range[0]) / res)
    y_max = 1 + int((fwd_range[1] - fwd_range[0]) / res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)

    # FILL PIXEL VALUES IN IMAGE ARRAY
    im[y_img, x_img] = pixel_values

    return im

if __name__ == '__main__':
    file = '../data_utils/oxford/sample/velodyne_left/1547131141271541.bin'
    im_file = '../data_utils/oxford/sample/velodyne_left/1547131046260961.png'
    im2_file = '../data_utils/oxford/sample/velodyne_right/1547131046278808.png'

    im = post_process_lidar(im_file, im2_file)
    cv2.imshow("lidar", im)
    cv2.waitKey(0)
