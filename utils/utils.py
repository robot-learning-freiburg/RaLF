import os
import os.path as osp
import numpy as np
import json
import math
import ipdb
import matplotlib.pyplot as plt
import pandas as pd
import csv
import cv2
import open3d as o3d
from tqdm import tqdm
from pathlib import Path
from PIL import Image



# Given a query, return the nearest value in the array
def find_nearest(array, query):
    array = np.asarray(array)
    idx = (np.abs(array - query)).argmin()
    return array[idx]


# Given a query, return index of nearest value in the array
def find_nearest_idx(array, query):
    array = np.asarray(array)
    idx = (np.abs(array - query)).argmin()
    return idx


def scale_to_255(a, min, max, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)


# Remove the ground plane using open3d. Helps create a cleaner map
def remove_ground_plane(ptcld, i_points):
    # print(ptcld.shape)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ptcld[:, 0:3])
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
                                                 num_iterations=1000)

    outliers_index = set(range(ptcld.shape[0])) - set(inliers)
    outliers_index = list(outliers_index)
    no_ground_scan = ptcld[outliers_index]
    i_pts = i_points[outliers_index]

    return no_ground_scan, i_pts



def get_validation_filter(pose_x, pose_y, xmin, xmax, ymin, ymax):
    fx1 = pose_x >= xmin
    fx2 = pose_x <= xmax
    fy1 = pose_y >= ymin 
    fy2 = pose_y <= ymax

    val_filter = fx1 * fx2 * fy1 * fy2

    return val_filter


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


def get_pixel_coordinates(points, fr, sr, img_res=0.2):
    x_points, y_points = points[:, 0], points[:, 1]

    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (x_points / img_res).astype(np.int32)  # x axis is -x in LIDAR
    y_img = (y_points / img_res).astype(np.int32)  # y axis is y in LIDAR

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor and ceil used to prevent anything being rounded to below 0 after shift
    x_img -= int(np.floor(fr[0] / img_res))
    y_img -= int(np.floor(sr[0] / img_res))
    x_img = int((fr[1] - fr[0]) / img_res) - x_img

    return x_img, y_img


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
