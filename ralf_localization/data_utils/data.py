import json
import os
import os.path as osp
import open3d as o3d
from torch.utils.data import Dataset
import numpy as np
import copy
from PIL import Image

from data_utils.interpolate_poses import *


def extract_poses(odom_file, rtime_file):
    fp = open(rtime_file, 'r')
    timestamps = [int(time.strip().split(' ')[0]) for time in fp.readlines()]

    gt_poses = interpolate_vo_poses(odom_file, timestamps, timestamps[0])
    # gt_poses = [list(pose[0:3, 3]) for pose in gt_poses]
    
    return gt_poses


def generate_image(xc, yc, xmax, ymax, pixel_values):
    im = np.zeros([xmax, ymax], dtype=np.uint8)

    x_img = np.clip(xc, 0, xmax - 1)
    y_img = np.clip(yc, 0, ymax - 1)

    # FILL PIXEL VALUES IN IMAGE ARRAY
    im[x_img, y_img] = pixel_values[:, 0]

    im = Image.fromarray(np.uint8(im))

    return im


def get_good_indices(c_x, c_y, xmax, ymax):
    fx1 = c_x >= 0
    fx2 = c_x < xmax
    fy1 = c_y >= 0
    fy2 = c_y < ymax

    filter = fx1 * fx2 * fy1 * fy2

    return filter


def get_pixel_coordinates(points, fr, sr, res=0.2):
    x_points, y_points = points[:, 0], points[:, 1]

    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (x_points / res).astype(np.int32)
    y_img = (y_points / res).astype(np.int32)

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor and ceil used to prevent anything being rounded to below 0 after shift
    x_img -= int(np.floor(fr[0] / res))
    y_img -= int(np.floor(sr[0] / res))
    x_img = int((fr[1] - fr[0]) / res) - x_img

    return x_img, y_img


def post_process_lidar(points, fwd_range, side_range, angle, pos_diff, gt_pose, flow=False, return_gtim=False, res=0.5, img_size=256, radar_angle=0.0):
    x_points = points[:, 0]
    y_points = points[:, 1]

    points_copy = copy.deepcopy(points)
    points_copy = np.hstack((points_copy[:, 0:3], np.ones(x_points.shape)[:, np.newaxis]))

    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > side_range[0]), (y_points < side_range[1]))
    filter = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter).flatten()

    i_points = points[indices, 3][:, np.newaxis]

    extrinsics_dir = '/data/robotcar_dataset_sdk/extrinsics'

    with open(os.path.join(extrinsics_dir, 'radar.txt')) as extrinsics_file:
        extrinsics = next(extrinsics_file)
        gt_pose = np.matmul(np.linalg.inv(build_se3_transform([float(x) for x in extrinsics.split(' ')])), gt_pose)

    radar_trf = np.array([[np.cos(radar_angle), -np.sin(radar_angle), 0, 0.0],
                          [np.sin(radar_angle), np.cos(radar_angle), 0, 0.0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]], dtype=np.float64)
    
    rot_mat = np.array([[np.cos(angle), -np.sin(angle), 0, pos_diff[0]],
                        [np.sin(angle), np.cos(angle), 0, pos_diff[1]],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]], dtype=np.float64)

    temp = np.linalg.inv(gt_pose) @ points_copy[indices, :].T

    # Apply the same radar rotation
    temp = radar_trf @ temp

    # Now apply lidar only transformation
    m = rot_mat @ temp
    m = m.T
    points_no_noise = copy.deepcopy(temp.T)

    x_points = m[:, 0]
    y_points = m[:, 1]

    range_3d = img_size * res / 2.0
    fr = (-range_3d, range_3d)
    sr = (-range_3d, range_3d)

    f_filt = np.logical_and((x_points > fr[0]), (x_points < fr[1]))
    s_filt = np.logical_and((y_points > sr[0]), (y_points < sr[1]))
    filter = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter).flatten()

    x_points = m[indices, 0]
    y_points = m[indices, 1]
    i_points = i_points[indices]
    points_no_noise = np.array(points_no_noise[indices, :])
    points_noisy = np.array(m[indices, 0:3])

    gt_xc, gt_yc = get_pixel_coordinates(points_no_noise, fr, sr, res=res)
    n_xc, n_yc = get_pixel_coordinates(points_noisy, fr, sr, res=res)

    pixel_values = i_points * 255.0

    xmax, ymax = img_size, img_size

    if return_gtim:
        # Ground Truth Lidar Image
        gt_im = generate_image(gt_xc, gt_yc, xmax=xmax, ymax=ymax, pixel_values=pixel_values)
    
    # Noisy Lidar Image
    n_im = generate_image(n_xc, n_yc, xmax=xmax, ymax=ymax, pixel_values=pixel_values)

    if flow:
        good_idx_nc = get_good_indices(n_xc, n_yc, xmax, ymax)
        good_idx_c = get_good_indices(gt_xc, gt_yc, xmax, ymax)
        indices = good_idx_c * good_idx_nc
        n_xc, n_yc = n_xc[indices], n_yc[indices]
        gt_xc, gt_yc = gt_xc[indices], gt_yc[indices]

        noisy_coords = [res for res in zip(n_xc, n_yc)]
        gt_coords = [res for res in zip(gt_xc, gt_yc)]     

        if return_gtim:            
            return gt_im, n_im, gt_coords, noisy_coords, rot_mat
        
        return n_im, gt_coords, noisy_coords, rot_mat

    if return_gtim:
        return gt_im, n_im, rot_mat

    return n_im, rot_mat


class R2R_PR(Dataset):
    def __init__(self, dataroot, file_name='train_data.json', radar_transform=None, pos_d=5.0, train=True):
        super(R2R_PR, self).__init__()
        self.dataroot = dataroot
        
        with open(f"{dataroot}/{file_name}", 'r') as f:
            self.train_data = json.load(f)

        self.train_indices = list(self.train_data.keys())
        self.radar_times = np.array([int(t) for t in self.train_indices])

        self.poses = np.array([(self.train_data[k]['pose'][0][3], self.train_data[k]['pose'][1][3]) for k in self.train_data.keys()])
        self.gps_pos = np.array([self.train_data[k]['gps_pos'] for k in self.train_data.keys()])

        self.num_samples = len(self.train_indices)

        self.radar_transform = radar_transform

        # Extract positive samples within this distance
        self.pos_d = pos_d
        self.train = train

        self.result_root = "./RadarData"
        self.anchor_radar = f'{self.result_root}/anchor_radar'
        os.makedirs(self.anchor_radar, exist_ok=True)

    def __len__(self):
        return self.num_samples
    
    def get_pos_idx(self, idx):
        cur_pose = self.gps_pos[idx]
        pose_diff = np.linalg.norm(cur_pose - self.gps_pos, axis=1)

        pos_indices = list(np.where((pose_diff < self.pos_d))[0])

        pos_idx = np.random.choice(pos_indices, 1)[0]

        return pos_idx
    
    def get_radar_im(self, time, seq):
        radar_im_path = f"{self.dataroot}/{seq}/radar_bev/{time}.png"
        radar_img = Image.open(radar_im_path)

        return radar_img

    def __getitem__(self, idx):
        radar_time = self.radar_times[idx]
        anchor_pose = np.array(self.gps_pos[idx]).astype('float32')
        anchor_im = self.get_radar_im(radar_time, self.train_data[str(radar_time)]['seq'])

        if self.train == False:
            if self.radar_transform:
                anchor_im = self.radar_transform(anchor_im)
            return anchor_im, anchor_pose
        
        # Not training phase, get pos sample
        pos_idx = self.get_pos_idx(idx)        
        pos_pose = np.array(self.gps_pos[pos_idx]).astype('float32')
        pos_time = self.radar_times[pos_idx]

        pos_im = self.get_radar_im(pos_time, self.train_data[str(pos_time)]['seq'])

        if self.radar_transform:
            pos_im = self.radar_transform(pos_im)
            anchor_im = self.radar_transform(anchor_im)

        return anchor_im, pos_im, anchor_pose, pos_pose


class L2L_PR(Dataset):
    def __init__(self, dataroot, file_name='train_data.json', lidar_transform=None, img_size=256, res=0.5, pos_d=5., train=True):
        super(L2L_PR, self).__init__()
        self.dataroot = dataroot
        
        with open(f"{dataroot}/{file_name}", 'r') as f:
            self.train_data = json.load(f)

        self.train_indices = list(self.train_data.keys())
        self.radar_times = np.array([int(t) for t in self.train_indices])

        self.img_size = img_size
        self.res = res

        self.sequences = [item for item in sorted(os.listdir(dataroot)) if os.path.isdir(osp.join(dataroot, item))]
        self.maps = {}
        for seq in self.sequences:
            self.maps[seq] = np.load(f"{dataroot}/{seq}/downsampled_map.npz")['arr_0']

        self.poses = np.array([(self.train_data[k]['pose'][0][3], self.train_data[k]['pose'][1][3]) for k in self.train_data.keys()])
        self.gps_pos = np.array([self.train_data[k]['gps_pos'] for k in self.train_data.keys()])

        self.num_samples = len(self.train_indices)

        self.lidar_transform = lidar_transform

        self.pos_d = pos_d
        self.train = train

        self.result_root = "./L2LData"
        self.anchor_radar = f'{self.result_root}/anchor_radar'
        os.makedirs(self.anchor_radar, exist_ok=True)

    def __len__(self):
        return self.num_samples
    
    def get_pos_idx(self, idx):
        cur_pose = self.gps_pos[idx]
        pose_diff = np.linalg.norm(cur_pose - self.gps_pos, axis=1)

        pos_indices = list(np.where((pose_diff < self.pos_d))[0])

        pos_idx = np.random.choice(pos_indices, 1)[0]

        return pos_idx
    
    def __getitem__(self, idx):
        radar_time = self.radar_times[idx]

        # Get all information about the anchor sample
        ar_seq = self.train_data[str(radar_time)]['seq']
        ar_posex, ar_posey = self.poses[idx][0], self.poses[idx][1]

        ar_pose = np.array(self.train_data[str(radar_time)]['pose'])
        al, _ = post_process_lidar(self.maps[ar_seq], (ar_posex - 100, ar_posex + 100), (ar_posey - 100, ar_posey + 100),
                            0.0, [0, 0], ar_pose, flow=False, return_gtim=False, res=self.res, img_size=self.img_size)
        
        anchor_pose = np.array(self.gps_pos[idx]).astype('float32')

        if self.train == False:
            # In Validation mode, just need the anchor
            if self.lidar_transform:
                al_im = self.lidar_transform(al)
            
            return al_im, anchor_pose
        
        # Training Phase, need Pos sample
        pos_idx = self.get_pos_idx(idx)
        pos_time = self.radar_times[pos_idx]

        pr_seq = self.train_data[str(pos_time)]['seq']        
        pr_posex, pr_posey = self.poses[pos_idx][0], self.poses[pos_idx][1]

        
        pr_pose = np.array(self.train_data[str(pos_time)]['pose'])
        pl, _ = post_process_lidar(self.maps[pr_seq], (pr_posex - 100, pr_posex + 100), (pr_posey - 100, pr_posey + 100),
                                    0.0, [0, 0], pr_pose, flow=False, return_gtim=False, res=self.res, img_size=self.img_size)      

        pos_pose = np.array(self.gps_pos[pos_idx]).astype('float32')

        if self.lidar_transform:
            al_im = self.lidar_transform(al)
            pl_im = self.lidar_transform(pl)

        return al_im, pl_im, anchor_pose, pos_pose


class R2L_PR(Dataset):
    def __init__(self, dataroot, file_name='train_data.json', radar_transform=None, lidar_transform=None, img_size=256, res=0.5, pos_d=5.0, max_angle=np.pi/6.0, train=True):
        super(R2L_PR, self).__init__()
        self.dataroot = dataroot
        
        with open(f"{dataroot}/{file_name}", 'r') as f:
            self.train_data = json.load(f)

        self.train_indices = list(self.train_data.keys())
        self.radar_times = np.array([int(t) for t in self.train_indices])

        self.img_size = img_size
        self.res = res

        self.sequences = [item for item in sorted(os.listdir(dataroot)) if os.path.isdir(osp.join(dataroot, item))]
        self.maps = {}
        for seq in self.sequences:
            self.maps[seq] = np.load(f"{dataroot}/{seq}/downsampled_map.npz")['arr_0']

        self.poses = np.array([(self.train_data[k]['pose'][0][3], self.train_data[k]['pose'][1][3]) for k in self.train_data.keys()])
        self.gps_pos = np.array([self.train_data[k]['gps_pos'] for k in self.train_data.keys()])

        self.num_samples = len(self.train_indices)

        self.radar_transform = radar_transform
        self.lidar_transform = lidar_transform

        self.pos_d = pos_d
        self.max_angle = max_angle
        self.train = train

        self.result_root = "./R2LData"
        self.anchor_radar = f'{self.result_root}/anchor_radar'
        os.makedirs(self.anchor_radar, exist_ok=True)

    def __len__(self):
        return self.num_samples
    
    def get_pos_idx(self, idx):
        cur_pose = self.gps_pos[idx]
        pose_diff = np.linalg.norm(cur_pose - self.gps_pos, axis=1)

        pos_indices = list(np.where((pose_diff < self.pos_d))[0])

        pos_idx = np.random.choice(pos_indices, 1)[0]

        return pos_idx
    
    def get_radar_im(self, time, seq):
        radar_im_path = f"{self.dataroot}/{seq}/radar_bev/{time}.png"
        radar_img = Image.open(radar_im_path)

        return radar_img
    
    def __getitem__(self, idx):
        radar_time = self.radar_times[idx]
        pos_idx = self.get_pos_idx(idx)

        pos_time = self.radar_times[pos_idx]

        ar_seq = self.train_data[str(radar_time)]['seq']
        pr_seq = self.train_data[str(pos_time)]['seq']

        ar = self.get_radar_im(radar_time, ar_seq)
        pr = self.get_radar_im(pos_time, pr_seq)
        
        ar_posex, ar_posey = self.poses[idx][0], self.poses[idx][1] 
        pr_posex, pr_posey = self.poses[pos_idx][0], self.poses[pos_idx][1]

        ar_pose = np.array(self.train_data[str(radar_time)]['pose'])
        pr_pose = np.array(self.train_data[str(pos_time)]['pose'])

        if self.train:
            rot_angle = np.random.uniform(-self.max_angle, self.max_angle, 2)
        else:
            rot_angle = [0.0, 0.0]

        al, _ = post_process_lidar(self.maps[ar_seq], (ar_posex - 100, ar_posex + 100), (ar_posey - 100, ar_posey + 100),
                                    rot_angle[0], [0, 0], ar_pose, flow=False, return_gtim=False, res=self.res, img_size=self.img_size)
        pl, _ = post_process_lidar(self.maps[pr_seq], (pr_posex - 100, pr_posex + 100), (pr_posey - 100, pr_posey + 100),
                                    rot_angle[1], [0, 0], pr_pose, flow=False, return_gtim=False, res=self.res, img_size=self.img_size)      

        anchor_pose = np.array(self.gps_pos[idx]).astype('float32')
        pos_pose = np.array(self.gps_pos[pos_idx]).astype('float32')

        if self.radar_transform:
            ar = self.radar_transform(ar)
            pr = self.radar_transform(pr)

        if self.lidar_transform:
            al = self.lidar_transform(al)
            pl = self.lidar_transform(pl)

        return ar, al, pr, pl, anchor_pose, pos_pose


class R2L_Flow(Dataset):
    def __init__(self, dataroot, file_name='train_data.json', radar_transform=None, lidar_transform=None,
                 img_size=256, res=0.5, pos_d=5.0, max_angle=np.pi/6.0, max_trans=10.0, return_gtim=False, train=True):
        super(R2L_Flow, self).__init__()
        self.dataroot = dataroot
        
        with open(f"{dataroot}/{file_name}", 'r') as f:
            self.train_data = json.load(f)

        self.train_indices = list(self.train_data.keys())
        self.radar_times = np.array([int(t) for t in self.train_indices])

        self.img_size = img_size
        self.res = res

        self.sequences = [item for item in sorted(os.listdir(dataroot)) if os.path.isdir(osp.join(dataroot, item))]
        self.maps = {}
        for seq in self.sequences:
            self.maps[seq] = np.load(f"{dataroot}/{seq}/downsampled_map.npz")['arr_0']

        self.poses = np.array([(self.train_data[k]['pose'][0][3], self.train_data[k]['pose'][1][3]) for k in self.train_data.keys()])
        self.gps_pos = np.array([self.train_data[k]['gps_pos'] for k in self.train_data.keys()])

        self.num_samples = len(self.train_indices)

        self.radar_transform = radar_transform
        self.lidar_transform = lidar_transform

        self.pos_d = pos_d
        self.max_angle = max_angle
        self.max_trans = max_trans
        self.train = train

    def __len__(self):
        return self.num_samples

    def get_optical_flow(self, c, nc):
        flow = np.zeros((2, self.img_size, self.img_size))
        mask = np.zeros((self.img_size, self.img_size))

        c = np.array(c)
        nc = np.array(nc)

        f = c - nc

        flow[:, nc[:, 0], nc[:, 1]] = f.transpose()
        mask[nc[:, 0], nc[:, 1]] = 1
        
        return flow, mask
    
    def get_pos_idx(self, idx):
        cur_pose = self.gps_pos[idx]
        pose_diff = np.linalg.norm(cur_pose - self.gps_pos, axis=1)

        pos_indices = list(np.where((pose_diff < self.pos_d))[0])

        pos_idx = np.random.choice(pos_indices, 1)[0]

        return pos_idx
    
    def get_radar_im(self, time, seq):
        radar_im_path = f"{self.dataroot}/{seq}/radar_bev/{time}.png"
        radar_img = Image.open(radar_im_path)

        return radar_img
  
    def __getitem__(self, idx):
        radar_time = self.radar_times[idx]
        ar_seq = self.train_data[str(radar_time)]['seq']

        ar = self.get_radar_im(radar_time, ar_seq)

        anchor_pose = np.array(self.gps_pos[idx]).astype('float32')        
        ar_posex, ar_posey = self.poses[idx][0], self.poses[idx][1] 
        ar_pose = np.array(self.train_data[str(radar_time)]['pose'])

        # Radar image rotation
        radar_rot = np.random.uniform(-self.max_angle, self.max_angle, 2)
        ar_rot_rad = radar_rot[0]
        pr_rot_rad = radar_rot[1]

        ar_rot_theta = ar_rot_rad * 180.0 / np.pi
        pr_rot_theta = pr_rot_rad * 180.0 / np.pi

        ar = ar.rotate(-ar_rot_theta)
 
        rot_angle = np.random.uniform(-self.max_angle, self.max_angle, 2)
        trans = np.random.uniform(-self.max_trans, self.max_trans, 4)

        al, a_c, a_nc, a_trf = post_process_lidar(self.maps[ar_seq], (ar_posex - 100, ar_posex + 100), (ar_posey - 100, ar_posey + 100),
                                    rot_angle[0], trans[0:2], ar_pose, flow=True, return_gtim=False, res=self.res, img_size=self.img_size, radar_angle=ar_rot_rad)
        a_flow, a_mask = self.get_optical_flow(a_c, a_nc)

        if self.train == False:
            # Apply radar and lidar transforms
            if self.radar_transform:
                ar = self.radar_transform(ar)

            # Lidar Transform
            if self.lidar_transform:
                al = self.lidar_transform(al)

            return ar, al, a_flow, a_mask, a_trf, anchor_pose

        # Information about the positive sample
        pos_idx = self.get_pos_idx(idx)
        pos_time = self.radar_times[pos_idx]
        pr_seq = self.train_data[str(pos_time)]['seq']

        pr = self.get_radar_im(pos_time, pr_seq)
        pr = pr.rotate(-pr_rot_theta)
        
        pos_pose = np.array(self.gps_pos[pos_idx]).astype('float32')
        pr_posex, pr_posey = self.poses[pos_idx][0], self.poses[pos_idx][1]
        pr_pose = np.array(self.train_data[str(pos_time)]['pose'])

        pl, p_c, p_nc, p_trf = post_process_lidar(self.maps[pr_seq], (pr_posex - 100, pr_posex + 100), (pr_posey - 100, pr_posey + 100),
                                    rot_angle[1], trans[2:], pr_pose, flow=True, return_gtim=False, res=self.res, img_size=self.img_size, radar_angle=pr_rot_rad)
        p_flow, p_mask = self.get_optical_flow(p_c, p_nc)

        if self.radar_transform:
            ar = self.radar_transform(ar)
            pr = self.radar_transform(pr)

        if self.lidar_transform:
            al = self.lidar_transform(al)
            pl = self.lidar_transform(pl)

        return ar, al, pr, pl, a_flow, a_mask, p_flow, p_mask, anchor_pose, pos_pose

