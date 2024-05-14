import json
import time
from operator import neg
import os
import os.path as osp
from turtle import pos
import open3d as o3d
import cv2
import ipdb
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
# from torchvision.io import read_image
import matplotlib.pyplot as plt
import copy
import math

from utils.interpolate_poses import *
from utils.config import *
from utils.utils import *

kaist_lidar_to_base_init_se3 = np.array([[-1.0000, -0.0058, -0.0000, 1.7042],
                                    [0.0058, -1.0000, 0.0000, -0.0210],
                                    [-0.0000, 0.0000, 1.0000, 1.8047],
                                    [0, 0, 0, 1.0000]])
kaist_base_to_lidar = np.linalg.inv(kaist_lidar_to_base_init_se3)

kaist_radar_to_base_init_se3 = np.array([[ 0.99987663, -0.01570732, 0.0, 1.5],
                                    [ 0.01570732, 0.99987663, 0.0, -0.04],
                                    [ 0.0, 0.0, 1.0, 1.97],
                                    [ 0.0, 0.0, 0.0, 1.0]])
kaist_base_to_radar = np.linalg.inv(kaist_radar_to_base_init_se3)

def extract_poses(odom_file, rtime_file):
    fp = open(rtime_file, 'r')
    timestamps = [int(time.strip().split(' ')[0]) for time in fp.readlines()]

    gt_poses = interpolate_vo_poses(odom_file, timestamps, timestamps[0])
    
    return gt_poses


def post_process_lidar_oxford(points, fwd_range, side_range, angle, pos_diff, gt_pose, flow=False, return_gtim=False, img_res=0.5, img_size=256, radar_angle=0.0):
    x_points = points[:, 0]
    y_points = points[:, 1]

    points_copy = copy.deepcopy(points)
    points_copy = np.hstack((points_copy[:, 0:3], np.ones(x_points.shape)[:, np.newaxis]))

    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > side_range[0]), (y_points < side_range[1]))
    filter = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter).flatten()

    i_points = points[indices, 3][:, np.newaxis]

    extrinsics_dir = ox_extr_dir

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

    range_3d = img_size * img_res / 2.0
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

    gt_xc, gt_yc = get_pixel_coordinates(points_no_noise, fr, sr, img_res=img_res)
    n_xc, n_yc = get_pixel_coordinates(points_noisy, fr, sr, img_res=img_res)

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

        noisy_coords = [coord for coord in zip(n_xc, n_yc)]
        gt_coords = [coord for coord in zip(gt_xc, gt_yc)]

        if return_gtim:            
            return gt_im, n_im, gt_coords, noisy_coords, rot_mat
        
        return n_im, gt_coords, noisy_coords, rot_mat

    if return_gtim:
        return gt_im, n_im, rot_mat

    return n_im, rot_mat

def post_process_lidar_boreas(points, fwd_range, side_range, angle, pos_diff, gt_pose, flow=False, return_gtim=False, img_res=0.5, img_size=256, radar_angle=0.0):
    # print(points.shape)
    # ipdb.set_trace()
    x_points = points[:, 0]
    y_points = points[:, 1]

    points_copy = copy.deepcopy(points)
    points_copy = np.hstack((points_copy[:, 0:3], np.ones(x_points.shape)[:, np.newaxis]))

    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > side_range[0]), (y_points < side_range[1]))
    filter = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter).flatten()

    i_points = points[indices, 3][:, np.newaxis]

    T_lidar_to_radar = np.array([[6.861993198242921643e-01, 7.274135642622281406e-01, 0.000000000000000000e+00, 0.000000000000000000e+00],
                            [7.274135642622281406e-01, -6.861993198242921643e-01, 0.000000000000000000e+00, 0.000000000000000000e+00],
                            [0.000000000000000000e+00, 0.000000000000000000e+00, -1.000000000000000000e+00, 2.100000000000000000e-01],
                            [0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]])

    pi_by_4_rot = np.array([[np.cos(-np.pi / 4.0), -np.sin(-np.pi / 4.0), 0, 0],
                            [np.sin(-np.pi / 4.0), np.cos(-np.pi / 4.0), 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=np.float64)
    
    radar_trf = np.array([[np.cos(radar_angle), -np.sin(radar_angle), 0, 0.0],
                        [np.sin(radar_angle), np.cos(radar_angle), 0, 0.0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]], dtype=np.float64)
    
    # angle -= np.pi / 4.0
    rot_mat = np.array([[np.cos(angle), -np.sin(angle), 0, pos_diff[0]],
                        [np.sin(angle), np.cos(angle), 0, pos_diff[1]],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]]   , dtype=np.float64)
    # ipdb.set_trace()
    temp = np.linalg.inv(gt_pose) @ points_copy[indices, :].T
    # temp = pi_by_4_rot @ temp

    temp = T_lidar_to_radar @ temp
    
    # Apply the same radar rotation
    temp = radar_trf @ temp

    m = rot_mat @ temp
    m = m.T
    points_no_noise = copy.deepcopy(temp.T)

    x_points = m[:, 0]
    y_points = m[:, 1]

    range_3d = img_size * img_res / 2.0
    fr = (-range_3d, range_3d)
    sr = (-range_3d, range_3d)

    f_filt = np.logical_and((x_points > fr[0]), (x_points < fr[1]))
    s_filt = np.logical_and((y_points > sr[0]), (y_points < sr[1]))
    filter = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter).flatten()

    x_points = m[indices, 0]
    y_points = m[indices, 1]
    i_points = i_points[indices] #np.ones_like(x_points) * 255.0#i_points[indices]
    points_no_noise = np.array(points_no_noise[indices, :])
    points_noisy = np.array(m[indices, 0:3]) 

    gt_xc, gt_yc = get_pixel_coordinates(points_no_noise, fr, sr, img_res=img_res)
    n_xc, n_yc = get_pixel_coordinates(points_noisy, fr, sr, img_res=img_res)

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

        noisy_coords = [coord for coord in zip(n_xc, n_yc)]
        gt_coords = [coord for coord in zip(gt_xc, gt_yc)]     

        if return_gtim:            
            return gt_im, n_im, gt_coords, noisy_coords, rot_mat
        
        return n_im, gt_coords, noisy_coords, rot_mat

    if return_gtim:
        return gt_im, n_im, rot_mat

    return n_im, rot_mat

def post_process_lidar_kaist(points, fwd_range, side_range, angle, pos_diff, gt_pose, flow=False, return_gtim=False, img_res=0.5, img_size=256, radar_angle=0.0):
    # print(points.shape)
    x_points = points[:, 0]
    # ipdb.set_trace()

    points_copy = copy.deepcopy(points)
    points_copy = np.hstack((points_copy[:, 0:3], np.ones(x_points.shape)[:, np.newaxis]))

    points_copy = kaist_base_to_radar @ np.linalg.inv(gt_pose) @ points_copy.T
    points_copy = points_copy.T

    x_points = points_copy[:, 0]
    y_points = points_copy[:, 1]

    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > side_range[0]), (y_points < side_range[1]))
    filter = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter).flatten()

    i_points = points[indices, 3][:, np.newaxis]
    
    radar_trf = np.array([[np.cos(radar_angle), -np.sin(radar_angle), 0, 0.0],
                        [np.sin(radar_angle), np.cos(radar_angle), 0, 0.0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]], dtype=np.float64)
    
    # angle -= np.pi / 4.0
    rot_mat = np.array([[np.cos(angle), -np.sin(angle), 0, pos_diff[0]],
                        [np.sin(angle), np.cos(angle), 0, pos_diff[1]],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]]   , dtype=np.float64)
    # ipdb.set_trace()
    temp = points_copy[indices, :].T
    
    # Apply the same radar rotation
    temp = radar_trf @ temp

    m = rot_mat @ temp
    m = m.T
    points_no_noise = copy.deepcopy(temp.T)

    x_points = m[:, 0]
    y_points = m[:, 1]

    range_3d = img_size * img_res / 2.0
    fr = (-range_3d, range_3d)
    sr = (-range_3d, range_3d)

    f_filt = np.logical_and((x_points > fr[0]), (x_points < fr[1]))
    s_filt = np.logical_and((y_points > sr[0]), (y_points < sr[1]))
    filter = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter).flatten()

    x_points = m[indices, 0]
    y_points = m[indices, 1]
    i_points = i_points[indices] #np.ones_like(x_points) * 255.0#i_points[indices]

    rmax = np.max(i_points)
    rmin = np.min(i_points)
    tmax = 255.0
    tmin = 0.0

    i_points = (i_points - rmin) * 255.0 / (rmax - rmin) + tmin
    points_no_noise = np.array(points_no_noise[indices, :])
    points_noisy = np.array(m[indices, 0:3]) 

    gt_xc, gt_yc = get_pixel_coordinates(points_no_noise, fr, sr, img_res=img_res)
    n_xc, n_yc = get_pixel_coordinates(points_noisy, fr, sr, img_res=img_res)

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

        noisy_coords = [coord for coord in zip(n_xc, n_yc)]
        gt_coords = [coord for coord in zip(gt_xc, gt_yc)]     

        if return_gtim:            
            return gt_im, n_im, gt_coords, noisy_coords, rot_mat
        
        return n_im, gt_coords, noisy_coords, rot_mat

    if return_gtim:
        return gt_im, n_im, rot_mat

    return n_im, rot_mat


class R2L_oxford(Dataset):
    def __init__(self, dataroot, file_name='train_data.json', radar_transform=None, lidar_transform=None, img_size=256, img_res=0.5, pos_d=5.0, max_angle=np.pi/6.0, train=True):
        super(R2L_oxford, self).__init__()
        self.dataroot = dataroot
        
        with open(f"{dataroot}/{file_name}", 'r') as f:
            self.train_data = json.load(f)

        self.train_indices = list(self.train_data.keys())
        self.radar_times = np.array([int(t) for t in self.train_indices])

        self.img_size = img_size
        self.img_res = img_res

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
        # radar_im_path = f"{self.dataroot}/{seq}/radar_bev/{time}.png"
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

        al, _ = post_process_lidar_oxford(self.maps[ar_seq], (ar_posex - 100, ar_posex + 100), (ar_posey - 100, ar_posey + 100),
                                    rot_angle[0], [0, 0], ar_pose, flow=False, return_gtim=False, img_res=self.img_res, img_size=self.img_size)
        pl, _ = post_process_lidar_oxford(self.maps[pr_seq], (pr_posex - 100, pr_posex + 100), (pr_posey - 100, pr_posey + 100),
                                    rot_angle[1], [0, 0], pr_pose, flow=False, return_gtim=False, img_res=self.img_res, img_size=self.img_size)      

        anchor_pose = np.array(self.gps_pos[idx]).astype('float32')
        pos_pose = np.array(self.gps_pos[pos_idx]).astype('float32')

        if self.radar_transform:
            ar = self.radar_transform(ar)
            pr = self.radar_transform(pr)

        if self.lidar_transform:
            al = self.lidar_transform(al)
            pl = self.lidar_transform(pl)

        return ar, al, pr, pl, anchor_pose, pos_pose

class R2L_Flow_oxford(Dataset):
    def __init__(self, dataroot, file_name='train_data.json', radar_transform=None, lidar_transform=None,
                 img_size=256, img_res=0.5, pos_d=5.0, max_angle=np.pi/6.0, max_trans=10.0, return_gtim=False, train=True):
        super(R2L_Flow_oxford, self).__init__()
        self.dataroot = dataroot
        
        with open(f"{dataroot}/{file_name}", 'r') as f:
            self.train_data = json.load(f)

        self.train_indices = list(self.train_data.keys())
        self.radar_times = np.array([int(t) for t in self.train_indices])

        self.img_size = img_size
        self.img_res = img_res

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

        # radar_im_path = f"{self.dataroot}/{seq}/radar_bev/{time}.png"
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

        al, a_c, a_nc, a_trf = post_process_lidar_oxford(self.maps[ar_seq], (ar_posex - 100, ar_posex + 100), (ar_posey - 100, ar_posey + 100),
                                    rot_angle[0], trans[0:2], ar_pose, flow=True, return_gtim=False, img_res=self.img_res, img_size=self.img_size, radar_angle=ar_rot_rad)
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

        pl, p_c, p_nc, _ = post_process_lidar_oxford(self.maps[pr_seq], (pr_posex - 100, pr_posex + 100), (pr_posey - 100, pr_posey + 100),
                                    rot_angle[1], trans[2:], pr_pose, flow=True, return_gtim=False, img_res=self.img_res, img_size=self.img_size, radar_angle=pr_rot_rad)
        p_flow, p_mask = self.get_optical_flow(p_c, p_nc)

        if self.radar_transform:
            ar = self.radar_transform(ar)
            pr = self.radar_transform(pr)

        if self.lidar_transform:
            al = self.lidar_transform(al)
            pl = self.lidar_transform(pl)

        return ar, al, pr, pl, a_flow, a_mask, p_flow, p_mask, anchor_pose, pos_pose

class R2L_boreas(Dataset):
    def __init__(self, dataroot, file_name='train_data.json', radar_transform=None, lidar_transform=None, 
                 img_size=256, img_res=0.5, pos_d=5.0, max_angle=np.pi/6.0, max_trans=5.0, train=True):
        super(R2L_boreas, self).__init__()
        self.dataroot = dataroot
        
        with open(f"{dataroot}/{file_name}", 'r') as f:
            self.train_data = json.load(f)

        self.train_indices = list(self.train_data.keys())
        self.radar_times = np.array([int(t) for t in self.train_indices])

        self.img_size = img_size
        self.img_res = img_res

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

        al, _ = post_process_lidar_boreas(self.maps[ar_seq], (ar_posex - 100, ar_posex + 100), (ar_posey - 100, ar_posey + 100),
                                    rot_angle[0], [0, 0], ar_pose, flow=False, return_gtim=False, img_res=self.img_res, img_size=self.img_size)
        pl, _ = post_process_lidar_boreas(self.maps[pr_seq], (pr_posex - 100, pr_posex + 100), (pr_posey - 100, pr_posey + 100),
                                    rot_angle[1], [0, 0], pr_pose, flow=False, return_gtim=False, img_res=self.img_res, img_size=self.img_size)      

        anchor_pose = np.array(self.gps_pos[idx]).astype('float32')
        pos_pose = np.array(self.gps_pos[pos_idx]).astype('float32')

        if self.radar_transform:
            ar = self.radar_transform(ar)
            pr = self.radar_transform(pr)

        if self.lidar_transform:
            al = self.lidar_transform(al)
            pl = self.lidar_transform(pl)

        return ar, al, pr, pl, anchor_pose, pos_pose

class R2L_Flow_boreas(Dataset):
    def __init__(self, dataroot, file_name='train_times.json', radar_transform=None, lidar_transform=None,
                 img_size=256, img_res=0.5, pos_d=5.0, max_angle=np.pi/6.0, max_trans=10.0, return_gtim=False, train=True):
        super(R2L_Flow_boreas, self).__init__()
        self.dataroot = dataroot
        
        with open(f"{dataroot}/{file_name}", 'r') as f:
            self.train_data = json.load(f)

        self.train_indices = list(self.train_data.keys())
        self.radar_times = np.array([int(t) for t in self.train_indices])

        self.img_size = img_size
        self.img_res = img_res

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

        # ipdb.set_trace()
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
                
        # timestamps, azimuths, valid, fft_data, radar_resolution = load_radar(radar_im_path)
        # radar_img = radar_polar_to_cartesian(azimuths, fft_data, radar_resolution, 0.5, 256)

        return radar_img
  
    def __getitem__(self, idx):
        radar_time = self.radar_times[idx]
        ar_seq = self.train_data[str(radar_time)]['seq']

        ar = self.get_radar_im(radar_time, ar_seq)

        anchor_pose = np.array(self.gps_pos[idx]).astype('float32')        
        ar_posex, ar_posey = self.poses[idx][0], self.poses[idx][1] 
        ar_pose = np.array(self.train_data[str(radar_time)]['pose'])

        # Radar image rotation
        radar_rot = [0.0, 0.0] #np.random.uniform(-self.max_angle, self.max_angle, 2)
        ar_rot_rad = radar_rot[0]
        pr_rot_rad = radar_rot[1]

        ar_rot_theta = ar_rot_rad * 180.0 / np.pi
        pr_rot_theta = pr_rot_rad * 180.0 / np.pi

        ar = ar.rotate(-ar_rot_theta)
 
        rot_angle = np.random.uniform(-self.max_angle, self.max_angle, 2)
        trans = np.random.uniform(-self.max_trans, self.max_trans, 4)

        al, a_c, a_nc, a_trf = post_process_lidar_boreas(self.maps[ar_seq], (ar_posex - 100, ar_posex + 100), (ar_posey - 100, ar_posey + 100),
                                    rot_angle[0], trans[0:2], ar_pose, flow=True, return_gtim=False, img_res=self.img_res, img_size=self.img_size, radar_angle=ar_rot_rad)
        a_flow, a_mask = self.get_optical_flow(a_c, a_nc)

        if self.train == False:
            # Apply radar and lidar transforms
            if self.radar_transform:
                ar = self.radar_transform(ar)

            # Lidar Transform
            if self.lidar_transform:
                al = self.lidar_transform(al)
                # gt_al = self.lidar_transform(gt_al)

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

        pl, p_c, p_nc, p_trf = post_process_lidar_boreas(self.maps[pr_seq], (pr_posex - 100, pr_posex + 100), (pr_posey - 100, pr_posey + 100),
                                    rot_angle[1], trans[2:], pr_pose, flow=True, return_gtim=False, img_res=self.img_res, img_size=self.img_size, radar_angle=pr_rot_rad)
        p_flow, p_mask = self.get_optical_flow(p_c, p_nc)

        if self.radar_transform:
            ar = self.radar_transform(ar)
            pr = self.radar_transform(pr)

        if self.lidar_transform:
            al = self.lidar_transform(al)
            pl = self.lidar_transform(pl)

            # gt_al = self.lidar_transform(gt_al)
            # gt_pl = self.lidar_transform(gt_pl)

        return ar, al, pr, pl, a_flow, a_mask, p_flow, p_mask, anchor_pose, pos_pose

class R2L_kaist(Dataset):
    def __init__(self, dataroot, file_name='train_times.json', radar_transform=None, lidar_transform=None, 
                 img_size=256, img_res=0.5, pos_d=5.0, max_angle=np.pi/6.0, max_trans=5.0, train=True):
        super(R2L_kaist, self).__init__()
        self.dataroot = dataroot
        
        with open(f"{dataroot}/{file_name}", 'r') as f:
            self.train_data = json.load(f)

        self.train_indices = list(self.train_data.keys())
        self.radar_times = np.array([int(t) for t in self.train_indices])

        self.img_size = img_size
        self.img_res = img_res

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

        # self.result_root = "./R2LData"
        # self.anchor_radar = f'{self.result_root}/anchor_radar'
        # os.makedirs(self.anchor_radar, exist_ok=True)

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
                
        # timestamps, azimuths, valid, fft_data, radar_resolution = load_radar(radar_im_path)
        # radar_img = radar_polar_to_cartesian(azimuths, fft_data, radar_resolution, 0.5, 256)

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

        al, _ = post_process_lidar_kaist(self.maps[ar_seq], (-200, 200), (-200, 200),
                                    rot_angle[0], [0, 0], ar_pose, flow=False, return_gtim=False, img_res=self.img_res, img_size=self.img_size)
        pl, _ = post_process_lidar_kaist(self.maps[pr_seq], (-200, 200), (-200, 200),
                                    rot_angle[1], [0, 0], pr_pose, flow=False, return_gtim=False, img_res=self.img_res, img_size=self.img_size)      

        anchor_pose = np.array(self.gps_pos[idx]).astype('float32')
        pos_pose = np.array(self.gps_pos[pos_idx]).astype('float32')

        if self.radar_transform:
            ar = self.radar_transform(ar)
            pr = self.radar_transform(pr)

        if self.lidar_transform:
            al = self.lidar_transform(al)
            pl = self.lidar_transform(pl)

        return ar, al, pr, pl, anchor_pose, pos_pose

class R2L_Flow_kaist(Dataset):
    def __init__(self, dataroot, file_name='train_times.json', radar_transform=None, lidar_transform=None,
                 img_size=256, img_res=0.5, pos_d=5.0, max_angle=np.pi/6.0, max_trans=10.0, return_gtim=False, train=True):
        super(R2L_Flow_kaist, self).__init__()
        self.dataroot = dataroot
        
        with open(f"{dataroot}/{file_name}", 'r') as f:
            self.train_data = json.load(f)

        self.train_indices = list(self.train_data.keys())
        self.radar_times = np.array([int(t) for t in self.train_indices])

        self.img_size = img_size
        self.img_res = img_res

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

        # ipdb.set_trace()
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
                
        # timestamps, azimuths, valid, fft_data, radar_resolution = load_radar(radar_im_path)
        # radar_img = radar_polar_to_cartesian(azimuths, fft_data, radar_resolution, 0.5, 256)

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

        al, a_c, a_nc, a_trf = post_process_lidar_kaist(self.maps[ar_seq], (-100, 100), (-100, 100),
                                    rot_angle[0], trans[0:2], ar_pose, flow=True, return_gtim=False, img_res=self.img_res, img_size=self.img_size, radar_angle=-ar_rot_rad)
        a_flow, a_mask = self.get_optical_flow(a_c, a_nc)

        if self.train == False:
            # Apply radar and lidar transforms
            if self.radar_transform:
                ar = self.radar_transform(ar)

            # Lidar Transform
            if self.lidar_transform:
                al = self.lidar_transform(al)
                # gt_al = self.lidar_transform(gt_al)

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

        pl, p_c, p_nc, p_trf = post_process_lidar_kaist(self.maps[pr_seq], (-100, 100), (-100, 100),
                                    rot_angle[1], trans[2:], pr_pose, flow=True, return_gtim=False, img_res=self.img_res, img_size=self.img_size, radar_angle=-pr_rot_rad)
        p_flow, p_mask = self.get_optical_flow(p_c, p_nc)

        if self.radar_transform:
            ar = self.radar_transform(ar)
            pr = self.radar_transform(pr)

        if self.lidar_transform:
            al = self.lidar_transform(al)
            pl = self.lidar_transform(pl)

            # gt_al = self.lidar_transform(gt_al)
            # gt_pl = self.lidar_transform(gt_pl)

        return ar, al, pr, pl, a_flow, a_mask, p_flow, p_mask, anchor_pose, pos_pose


if __name__ == '__main__':
    test = 'R2L'
    start_time = time.process_time()

    if test == 'R2L':
        data_root = '/work/dlclarge2/nayaka-CML_Workspace/radar_robotcar_sequences'
        dataset = R2L(dataroot=data_root, file_name='train_times_standard_subset.json')

        for idx in range(3000, 5000, 100):
            data = dataset.__getitem__(idx)
            print("Fin")
            # ipdb.set_trace()
            continue

            al = np.asarray(al, np.float64)
            ar = np.asarray(ar, np.float64)
            pl = np.asarray(pl, np.float64)
            pr = np.asarray(pr, np.float64)

            al_3 = np.zeros((al.shape[0], al.shape[1], 3))
            ar_3 = np.zeros((ar.shape[0], ar.shape[1], 3))

            al_3[:, :, 2] = al
            ar_3[:, :, 0] = ar

            # ipdb.set_trace()

            sumImg = cv2.addWeighted(ar_3, 1.0, al_3, 0.5, 0)

            cv2.imwrite(os.path.join(dataset.combined_path, f'{idx}.jpg'), sumImg)
            cv2.imwrite(os.path.join(dataset.anchor_lidar, f'{idx}.jpg'), al)
            cv2.imwrite(os.path.join(dataset.anchor_radar, f'{idx}.jpg'), ar)
            cv2.imwrite(os.path.join(dataset.pos_lidar_path, f'{idx}.jpg'), pl)
            cv2.imwrite(os.path.join(dataset.pos_radar_path, f'{idx}.jpg'), pr)

            np.savez(f'{dataset.flow_data_path}/a_flow_{idx}.npz', a_flow)
            np.savez(f'{dataset.flow_data_path}/a_mask_{idx}.npz', a_mask)
            np.savez(f'{dataset.flow_data_path}/p_flow_{idx}.npz', p_flow)
            np.savez(f'{dataset.flow_data_path}/p_mask_{idx}.npz', p_mask)
            np.savez(f'{dataset.trans_path}/transformation_a_{idx}.npz', rot_mat_a)
            np.savez(f'{dataset.trans_path}/transformation_p_{idx}.npz', rot_mat_p)

            print("wrote image")
    elif test == 'RadarLidarWithGT':
        data_root = '/export/nayaka/radar_robotcar_sequences'
        dataset = RadarLidarDataset(dataroot=data_root, add_noise=True, flow=True, return_gtim=True)

        coords_dict = {}

        for idx in range(3000, 5000, 100):
            ar, gt_al, al, pr, gt_pl, pl, _, _, a_flow, a_mask, p_flow, p_mask = dataset.__getitem__(idx)
            # ipdb.set_trace()
            al = np.asarray(al, np.float64)
            ar = np.asarray(ar, np.float64)
            pl = np.asarray(pl, np.float64)
            pr = np.asarray(pr, np.float64)
            gt_al = np.asarray(gt_al, np.float64)
            gt_pl = np.asarray(gt_pl, np.float64)

            al_3 = np.zeros((al.shape[0], al.shape[1], 3))
            ar_3 = np.zeros((ar.shape[0], ar.shape[1], 3))

            al_3[:, :, 2] = al
            ar_3[:, :, 0] = ar

            # ipdb.set_trace()

            sumImg = cv2.addWeighted(ar_3, 1.0, al_3, 0.5, 0)

            cv2.imwrite(os.path.join(dataset.combined_path, f'{idx}.jpg'), sumImg)
            cv2.imwrite(os.path.join(dataset.anchor_lidar, f'{idx}.jpg'), al)
            cv2.imwrite(os.path.join(dataset.anchor_radar, f'{idx}.jpg'), ar)
            cv2.imwrite(os.path.join(dataset.pos_lidar_path, f'{idx}.jpg'), pl)
            cv2.imwrite(os.path.join(dataset.pos_radar_path, f'{idx}.jpg'), pr)
            cv2.imwrite(f"./gt_al_{idx}.jpg", gt_al)
            cv2.imwrite(f"./gt_pl_{idx}.jpg", gt_pl)

            np.savez(f'{dataset.flow_data_path}/a_flow_{idx}.npz', a_flow)
            np.savez(f'{dataset.flow_data_path}/a_mask_{idx}.npz', a_mask)
            np.savez(f'{dataset.flow_data_path}/p_flow_{idx}.npz', p_flow)
            np.savez(f'{dataset.flow_data_path}/p_mask_{idx}.npz', p_mask)

            print("wrote image")  
    elif test == 'RadarLidarFlowVal':
        data_root = '/export/nayaka/radar_robotcar_sequences'
        dataset = RadarLidarDatasetFlowVal(dataroot=data_root, add_noise=True, flow=True, return_gtim=True)

        for idx in range(3000, 5000, 100):
            ar, gt_al, al, _, a_flow, a_mask, rot_mat = dataset.__getitem__(idx)
            al = np.asarray(al, np.float64)
            gt_al = np.asarray(gt_al, np.float64)
            ar = np.asarray(ar, np.float64)

            al_3 = np.zeros((al.shape[0], al.shape[1], 3))
            ar_3 = np.zeros((ar.shape[0], ar.shape[1], 3))

            al_3[:, :, 2] = al
            ar_3[:, :, 0] = ar

            sumImg = cv2.addWeighted(ar_3, 1.0, al_3, 0.5, 0)

            cv2.imwrite(os.path.join(dataset.combined_path, f'{idx}.jpg'), sumImg)
            cv2.imwrite(os.path.join(dataset.anchor_lidar, f'{idx}.jpg'), al)
            cv2.imwrite(os.path.join(dataset.anchor_radar, f'{idx}.jpg'), ar)
            cv2.imwrite(os.path.join(dataset.anchor_lidar_gt, f'{idx}.jpg'), gt_al)

            np.savez(f'{dataset.flow_data_path}/a_flow_{idx}.npz', a_flow)
            np.savez(f'{dataset.flow_data_path}/a_mask_{idx}.npz', a_mask)
            np.savez(f'{dataset.trans_path}/a_trans_{idx}.npz', rot_mat)

            print("wrote image")
    elif test == 'RadarLidarDatasetFlowValBoreas':
        data_root = '/export/nayaka/boreas'
        dataset = RadarLidarDatasetFlowValBoreas(dataroot=data_root, add_noise=True, flow=True, return_gtim=True)

        for idx in range(1000, 3000, 100):
            ar, gt_al, al, _, a_flow, a_mask, rot_mat = dataset.__getitem__(idx)
            al = np.asarray(al, np.float64)
            gt_al = np.asarray(gt_al, np.float64)
            ar = np.asarray(ar, np.float64)

            al_3 = np.zeros((al.shape[0], al.shape[1], 3))
            ar_3 = np.zeros((ar.shape[0], ar.shape[1], 3))

            al_3[:, :, 2] = al
            ar_3[:, :, 0] = ar

            sumImg = cv2.addWeighted(ar_3, 1.0, al_3, 0.5, 0)

            cv2.imwrite(os.path.join(dataset.combined_path, f'{idx}.jpg'), sumImg)
            cv2.imwrite(os.path.join(dataset.anchor_lidar, f'{idx}.jpg'), al)
            cv2.imwrite(os.path.join(dataset.anchor_radar, f'{idx}.jpg'), ar)
            cv2.imwrite(os.path.join(dataset.anchor_lidar_gt, f'{idx}.jpg'), gt_al)

            np.savez(f'{dataset.flow_data_path}/a_flow_{idx}.npz', a_flow)
            np.savez(f'{dataset.flow_data_path}/a_mask_{idx}.npz', a_mask)
            np.savez(f'{dataset.trans_path}/a_trans_{idx}.npz', rot_mat)

            print("wrote image") 
    elif test == 'RadarLidarBoreas':
        data_root = '/export/nayaka/boreas'
        dataset = RadarLidarDatasetBoreas(dataroot=data_root, add_noise=True, flow=True)

        coords_dict = {}

        for idx in range(1000, 3000, 100):
            ar, al, pr, pl, _, _, a_flow, a_mask, p_flow, p_mask, rot_mat_a, rot_mat_p = dataset.__getitem__(idx)
            # ipdb.set_trace()
            al = np.asarray(al, np.float64)
            ar = np.asarray(ar, np.float64)
            pl = np.asarray(pl, np.float64)
            pr = np.asarray(pr, np.float64)

            al_3 = np.zeros((al.shape[0], al.shape[1], 3))
            ar_3 = np.zeros((ar.shape[0], ar.shape[1], 3))

            al_3[:, :, 2] = al
            ar_3[:, :, 0] = ar

            # ipdb.set_trace()
            # sumImg = np.zeros_like(al)

            sumImg = cv2.addWeighted(ar_3, 1.0, al_3, 0.5, 0)

            cv2.imwrite(os.path.join(dataset.combined_path, f'{idx}.jpg'), sumImg)
            cv2.imwrite(os.path.join(dataset.anchor_lidar, f'{idx}.jpg'), al)
            cv2.imwrite(os.path.join(dataset.anchor_radar, f'{idx}.jpg'), ar)
            cv2.imwrite(os.path.join(dataset.pos_lidar_path, f'{idx}.jpg'), pl)
            cv2.imwrite(os.path.join(dataset.pos_radar_path, f'{idx}.jpg'), pr)

            np.savez(f'{dataset.flow_data_path}/a_flow_{idx}.npz', a_flow)
            np.savez(f'{dataset.flow_data_path}/a_mask_{idx}.npz', a_mask)
            np.savez(f'{dataset.flow_data_path}/p_flow_{idx}.npz', p_flow)
            np.savez(f'{dataset.flow_data_path}/p_mask_{idx}.npz', p_mask)
            np.savez(f'{dataset.trans_path}/a_trans_{idx}.npz', rot_mat_a)
            np.savez(f'{dataset.trans_path}/p_trans_{idx}.npz', rot_mat_p)

            print("wrote image")  
    elif test == 'Radar':
        data_root = '/export/nayaka/radar_robotcar_sequences'
        dataset = RadarDataset(dataroot=data_root)

        for idx in range(3000, 5000, 100):
            ar = dataset.__getitem__(idx)
            ar = np.asarray(ar, np.float64)

            cv2.imwrite(os.path.join(dataset.anchor_radar, f'{idx}.jpg'), ar)

            print("wrote image")
    elif test == 'Lidar':
        data_root = '/export/nayaka/radar_robotcar_sequences'
        dataset = LidarDataset(dataroot=data_root, add_noise=False)

        for idx in range(3000, 5000, 100):
            al = dataset.__getitem__(idx)

            al = np.asarray(al, np.float64)

            cv2.imwrite(os.path.join(dataset.anchor_lidar, f'{idx}.jpg'), al)

            print("wrote image")
    elif test == 'RadarBoreas':
        data_root = '/export/nayaka/boreas'
        dataset = RadarDatasetBoreas(dataroot=data_root)

        for idx in range(1000, 3000, 100):
            ar = dataset.__getitem__(idx)
            ar = np.asarray(ar, np.float64)

            cv2.imwrite(os.path.join(dataset.anchor_radar, f'{idx}.jpg'), ar)

            print("wrote image")
    elif test == 'LidarBoreas':
        data_root = '/export/nayaka/boreas'
        dataset = LidarDatasetBoreas(dataroot=data_root, add_noise=False)

        for idx in range(1000, 3000, 100):
            al = dataset.__getitem__(idx)

            al = np.asarray(al, np.float64)

            cv2.imwrite(os.path.join(dataset.anchor_lidar, f'{idx}.jpg'), al)

            print("wrote image")
    elif test == "RadarDatasetTimeBasedFlow":
        data_root = '/export/nayaka/radar_robotcar_sequences'
        dataset = RadarDatasetTimeBasedFlow(dataroot=data_root, )

        for idx in range(0, 3000, 200):
            ar, al, flow, mask, trf = dataset.__getitem__(idx)

            ipdb.set_trace()

            al = np.asarray(al, np.float64)
            ar = np.asarray(ar, np.float64)

            al_3 = np.zeros((al.shape[0], al.shape[1], 3))
            ar_3 = np.zeros((ar.shape[0], ar.shape[1], 3))

            al_3[:, :, 2] = al
            ar_3[:, :, 0] = ar

            # ipdb.set_trace()

            sumImg = cv2.addWeighted(ar_3, 1.0, al_3, 0.5, 0)

            cv2.imwrite(os.path.join(dataset.combined_path, f'{idx}.jpg'), sumImg)
            cv2.imwrite(os.path.join(dataset.anchor_lidar, f'{idx}.jpg'), al)
            cv2.imwrite(os.path.join(dataset.anchor_radar, f'{idx}.jpg'), ar)

            np.savez(f'{dataset.flow_path}/a_flow_{idx}.npz', flow)
            np.savez(f'{dataset.mask_path}/a_mask_{idx}.npz', mask)
            np.savez(f'{dataset.trans_path}/trf_{idx}.npz', trf)

            print("wrote image")
    print(f"This took {time.process_time() - start_time} seconds")
