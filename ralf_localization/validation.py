import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from ralf_localization.options.parser import *
from ralf_localization.metrics import *

from ralf_localization.network_arcs.networks import *
from ralf_localization.data import *
from utils.config import *
from utils.utils import *

import flow_vis
import wandb
import open3d as o3d

import logging
import random
from scipy.spatial.transform import Rotation as R

MAX_FLOW = 300

def solve_transform(points_p, points_q):
    """
    Find transformation s.t. (R|t) @ p == q

    Arguments:
        points_p: (n, d) non-homogeneous
        points_q: (n, d) non-homogeneous
    Returns:
        align_transform: (d+1, d+1) homogeneous transformation
    """
    if len(points_p) == 0:
        logging.warning(f"solve_transform failed: empty set.")
        return np.eye(4)

    points_p = points_p[:, :3]
    points_q = points_q[:, :3]
    assert points_p.shape[1] == 3
    assert points_q.shape[1] == 3

    # compute mean translation
    p_mean = points_p.mean(axis=0)
    o_mean = points_q.mean(axis=0)

    # whiten
    points_x = points_p - p_mean
    points_y = points_q - o_mean

    s_matrix = (points_x.T @ points_y)

    u, _, v_T = np.linalg.svd(s_matrix)
    v = v_T.T
    det = np.linalg.det(v @ u.T)

    d_num = s_matrix.shape[0]
    Idt = np.eye(d_num)
    Idt[-1, -1] = det

    rot = (v @ Idt @ u.T)
    trans = o_mean - rot @ p_mean

    res = np.eye(d_num + 1)
    res[:d_num, :d_num] = rot
    res[:d_num, d_num] = trans

    if np.any(np.isnan(res)):
        # logging.warning(f"solve_transform failed, contains NaN values.")
        return np.eye(4)

    assert np.linalg.det(res) > 0, "res = {}".format(str(res))
    return res


def eval_fit(trf_estm, start_ptc, end_ptc):
    start_m = (trf_estm @ start_ptc[:, 0:4].T).T
    fit_qe = np.linalg.norm(start_m[:, :3] - end_ptc[:, :3], axis=1)
    return fit_qe


class Ransac:
    """
        Ransac class.
        Given a function that estimates a model that fits the given data robustly.

        Arguments:
            data_in: NxK, Contains N samples of input data (each sample is of dimension K).
            data_out: NxC, Contains N samples of output data (each sample is of dimension C).
            estimate_model_fct(data_in_ss, data_out_ss):
                Given data_in_ss in M'xK and data_out_ss in M'xC a model is estimated.
            score_model_fct(model, data_in_ss, data_out_ss):
                Calculates scores for each data point. Returns a vector of scores of length M'
    """
    def __init__(self, data_in, data_out, estimate_model_fct, score_model_fct,
                 thresh, num_pts_needed,
                 percentage_thresh=0.99, outlier_ratio=0.2):
        self.data_in = data_in
        self.data_out = data_out
        self.estimate = estimate_model_fct
        self.score = score_model_fct
        self.thresh = thresh
        self.num_pts_needed = num_pts_needed

        self.num_runs = np.ceil(np.log(1 - percentage_thresh) / np.log(1 - np.power(1 - outlier_ratio, num_pts_needed)))
        self.num_runs = int(self.num_runs)

    def run(self):
        """run ransac estimation"""
        n, inliers = -float('inf'), None
        for _ in range(self.num_runs):
            subset = random.sample(range(self.data_in.shape[0]), self.num_pts_needed)
            # ipdb.set_trace()
            model = self.estimate(self.data_in[subset, :3], self.data_out[subset, :3])
            scores = self.score(model, self.data_in, self.data_out)
            inliers = scores < self.thresh
            num_inliers = np.sum(inliers)

            if n < num_inliers:
                n = num_inliers
                inliers_best = inliers

        if inliers_best is None:
            return None

        # use best fit to calculate model from all inliers
        model_best = self.estimate(self.data_in[inliers_best], self.data_out[inliers_best])
        score_best = self.score(model_best, self.data_in, self.data_out)

        return score_best, model_best


def get_3d_points(x_img, y_img, fr, sr, res=0.2):
    # Use image size and resolution to convery image coordinates to 3d coordinates
    x_im = int((fr[1] - fr[0]) / res) - x_img
    x_im += int(np.floor(fr[0] / res))
    y_im = y_img + int(np.floor(sr[0] / res))

    x_points = x_im * res
    y_points = y_im * res
    
    return x_points, y_points


def get_image_coords(flow, mask):
    x, y = np.where(mask == 1)

    # Coordinates after noise augmentation
    nc = [(x_, y_) for (x_, y_) in zip(x, y)]
    nc = np.array(nc)
    
    f = flow[:, x, y].T

    # Coordinates before noise augmentation
    c = nc + f
    
    return c, nc


def compute_transformation(c, nc, fr, sr, res=0.2):
    c_x, c_y = c[:, 0], c[:, 1]
    nc_x, nc_y = nc[:, 0], nc[:, 1]
    
    x_p, y_p = get_3d_points(c_x, c_y, fr, sr, res)
    n_xp, n_yp = get_3d_points(nc_x, nc_y, fr, sr, res)

    start_pts = [(x_pt, y_pt, 0.0) for (x_pt, y_pt) in zip(n_xp, n_yp)]
    end_pts = [(x_pt, y_pt, 0.0) for (x_pt, y_pt) in zip(x_p, y_p)]

    start_pts = np.array(start_pts)
    end_pts = np.array(end_pts)
    
    start_pts = np.pad(start_pts, ((0, 0), (0, 1)), mode="constant", constant_values=1)
    end_pts = np.pad(end_pts, ((0, 0), (0, 1)), mode="constant", constant_values=1)
    
    ransac = Ransac(start_pts, end_pts, solve_transform, eval_fit, .1, 5)
    fit_q_pos, trf_est = ransac.run()
    
    return fit_q_pos, trf_est, start_pts, end_pts


def compute_trf_error(flow, mask, trf_gt, img_size, res):
    c, nc = get_image_coords(flow, mask)
    
    range_3d = img_size * res / 2
    fr = (-range_3d, range_3d)
    sr = (-range_3d, range_3d)
    
    fit_score = np.inf
    best_trf = np.ones((4, 4))

    fit_q, trf_est, start_pts, end_pts = compute_transformation(c, nc, fr, sr, res)
    fs = np.linalg.norm(fit_q)
    if fs < fit_score:
        fit_score = fs
        best_trf = np.linalg.inv(trf_est)
    
    R_est = best_trf[0:3, 0:3]
    t_est = best_trf[:, 3]
    
    R_gt = trf_gt[0:3, 0:3]
    t_gt = trf_gt[:, 3]
    
    t_error = np.linalg.norm(t_est - t_gt)

    angle = np.arccos((np.trace(R_est.T @ R_gt) - 1) / 2)
    angle = np.degrees(angle)

    x_err = np.abs(t_gt[0] - t_est[0])
    y_err = np.abs(t_gt[1] - t_est[1])

    R_error = angle
    
    return t_error, R_error, fit_score, x_err, y_err


def get_batch_trf_error(pred_flow, a_mask, trf_gt, img_size=256, img_res=0.5, use_wandb=False):
    # assert pred_flow.shape[0] == a_mask.shape[0] == trf_gt.shape[0]
    
    total_t_error, total_R_error, fit_q, x_errors, y_errors = [], [], [], [], []

    for idx in range(pred_flow.shape[0]):
        t_err, R_err, fit_score, x_err, y_err = compute_trf_error(pred_flow[idx].cpu().numpy(), a_mask[idx].cpu().numpy(), trf_gt[idx].numpy(), img_size=img_size, res=img_res)
        total_t_error.append(t_err)
        total_R_error.append(R_err)
        fit_q.append(fit_score)
        x_errors.append(x_err)
        y_errors.append(y_err)

        if use_wandb:            
            wandb.log({"Sample T Error": t_err, "Sample R Error": R_err, "Sample Fit Quality": fit_score,
                    "Sample X Error": x_err, "Sample Y Error": y_err}, commit=False)

    return np.mean(total_t_error), np.mean(total_R_error), np.mean(fit_q), np.mean(x_errors), np.mean(y_errors)


def val_metrics(flow_pred, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    # Compute end-point error between final predicted flow and ground truth flow  
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    i_weight = 1
    i_loss = (flow_pred - flow_gt).abs()
    flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_pred - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def validate_pr_with_flow_model(query_model, map_model, map_loader, query_loader, cfg={}):
    use_wandb = cfg['use_wandb']
    
    query_model.eval()
    map_model.eval()

    with torch.no_grad():

        feats_query = []
        feats_map = []

        query_poses = []
        map_poses = []

        # Get the query features
        for idx, data in enumerate(tqdm(query_loader)):
            ar, al, _, _, a_pose, _ = data
            query_poses.append(a_pose)

            ar = ar.cuda()

            lf, rf, _, _ = query_model(lidar_=al, radar_=ar, iters=cfg['iters'], test_mode=True)
            query_features = rf.reshape(rf.shape[0], -1)

            feats_query.append(query_features)
        
        # Get the map features
        for idx, data in enumerate(tqdm(map_loader)):
            ar, al, _, _, a_pose, _ = data
            map_poses.append(a_pose)

            al = al.cuda()

            lf, rf, _, _ = map_model(lidar_=al, radar_=ar, iters=cfg['iters'], test_mode=True)
            map_features = lf.reshape(lf.shape[0], -1)

            feats_map.append(map_features)

        # Concatenate all batch features for the map and query
        feats_query = torch.cat(feats_query).cpu()
        feats_map = torch.cat(feats_map).cpu()

        map_poses = torch.cat(map_poses).detach().cpu().numpy()
        query_poses = torch.cat(query_poses).detach().cpu().numpy()

        # Now you have the query features as well as the map features
        # Use FAISS to compute similarity scores between all of these samples
        
        feature_size = feats_query.shape[-1]

        # Create faiss index with the feature size and L2 distance
        index = faiss.IndexFlatL2(feature_size)

        # Add map features to the index
        index.add(feats_map.numpy())

        # Search for 10 most similar map features
        k = 10
        D, I = index.search(feats_query.numpy(), k)
        
        K = range(1, 11)
        recall1m, recall3m, recall5m, recall10m = [], [], [], []

        for k in K:
            r1, r3, r5, r10 = recall_at_k_standard(I[:, 0:10], k=k, query_poses=query_poses, map_poses=map_poses)
            recall1m.append(r1)
            recall3m.append(r3)
            recall5m.append(r5)
            recall10m.append(r10)

        if use_wandb:
            wandb.log({f"recall@1 (3m) {cfg['dataset']}": recall3m[0]})
        else:
            print(f"recall@1 (3m) {cfg['dataset']}: {recall3m[0]}")

        return recall1m, recall3m, recall5m, recall10m


def validate_flow(model, flow_loader, cfg, total_steps):
    model.eval()

    use_wandb = cfg['use_wandb']
    device = cfg['device']

    total_T_error = []
    total_R_error = []
    total_X_error = []
    total_Y_error = []
    fit_quality = []

    with torch.no_grad():

        for i_batch, data_blob in enumerate(tqdm(flow_loader)):
            ar, al, a_flow, a_mask, trf_gt, a_pose = [x for x in data_blob]
            
            ar, al = ar.cuda(), al.cuda()
            a_flow, a_mask = a_flow.cuda(), a_mask.cuda()

            _, _, coord_diff, flow_pred = model(lidar_=al, radar_=ar, iters=cfg['iters'], test_mode=True)

            # Optical Flow Loss
            loss, metrics = val_metrics(flow_pred, a_flow, a_mask, cfg['gamma'])

            T_error, R_error, fit_q, X_error, Y_error = get_batch_trf_error(flow_pred, a_mask, trf_gt, use_wandb=use_wandb, img_size=cfg['img_size'], img_res=cfg['img_res'])
            total_T_error.append(T_error)
            total_R_error.append(R_error)
            total_X_error.append(X_error)
            total_Y_error.append(Y_error)
            fit_quality.append(fit_q)

            if use_wandb:
                wandb.log({"Val Loss": loss, 'Val EPE': metrics['epe'],
                        "Val 1px": metrics['1px'], "Val 3px": metrics['3px'],
                        "Val 5px": metrics['5px']}, commit=False)
                
                # Also display flow images on wandb
                f_gt = a_flow[0].cpu().numpy()
                f_gt = np.moveaxis(f_gt, 0, -1)

                f_pr = flow_pred[0].cpu().numpy()
                f_pr = np.moveaxis(f_pr, 0, -1)

                # GT flow image
                gt_flow_im = flow_vis.flow_to_color(f_gt, convert_to_bgr=False)

                # Use flow_pred to create flow image
                pred_flow_im = flow_vis.flow_to_color(f_pr, convert_to_bgr=False)

                H, W, C = gt_flow_im.shape

                # Create a flow image array and store values
                flow_array = np.zeros((W, H * 3, C))
                flow_array[:, 0:H, :] = gt_flow_im
                flow_array[:, H:2*H, :] = pred_flow_im
                
                a_m = a_mask[0].cpu().numpy()[..., np.newaxis].repeat(3, axis=2)
                flow_array[:, 2*H:3*H, :] = pred_flow_im * a_m

                images = wandb.Image(flow_array, caption=f"Left: GT, Right: Pred, Step: {total_steps}")
                wandb.log({"Flow Predictions": images})

        if use_wandb:
            wandb.log({f"T Error mean {cfg['dataset']}": np.mean(total_T_error),
                       f"R Error mean {cfg['dataset']}": np.mean(total_R_error),
                       f"Fit Quality mean {cfg['dataset']}": np.mean(fit_quality),
                       f"X Error mean {cfg['dataset']}": np.mean(total_X_error),
                       f"Y Error mean {cfg['dataset']}": np.mean(total_Y_error)})
        else:
            print(f"T Error mean {cfg['dataset']}: {np.mean(total_T_error)}")
            print(f"R Error mean {cfg['dataset']}: {np.mean(total_R_error)}")
            print(f"Fit Quality mean {cfg['dataset']}: {np.mean(fit_quality)}")
            print(f"X Error mean {cfg['dataset']}: {np.mean(total_X_error)}")
            print(f"Y Error mean {cfg['dataset']}: {np.mean(total_Y_error)}")


if __name__ == '__main__':
    config_file = './ralf_localization/configs/test.json'
    cfg = load_cfg(config_file)

    # Create model
    model = nn.DataParallel(RaLF(cfg=cfg)).cuda()

    if cfg['load_model'] is not None:
        # Load model
        print(f"Loading Model: {cfg['load_model']}")
        model.load_state_dict(torch.load(cfg['load_model']))
        model.eval()

    rot_aug = cfg['rot_aug'] * np.pi / 180.0
    trans_aug = cfg['trans_aug']
    val_bs = cfg['val_batch_size']

    if cfg['use_wandb']:
        wandb.init(project=cfg['wandb_proj'])

    # Transforms
    val_transform = T.Compose([T.ToTensor()])

    # Query and Map Data loaders - For place recognition validation
    v_query = R2L_boreas(dataroot=boreas_dataroot,
                       file_name=boreas_query_file,
                       radar_transform=val_transform, lidar_transform=val_transform,
                       train=False, max_angle=0.0)

    v_map = R2L_boreas(dataroot=boreas_dataroot,
                       file_name=boreas_map_file,
                       radar_transform=val_transform, lidar_transform=val_transform,
                       train=False, max_angle=0.0)
    
    # Add dataset attribute to the config. This will be used in wandb
    cfg['dataset'] = 'boreas'
    
    vquery_loader = DataLoader(v_query, batch_size=val_bs, shuffle=False, num_workers=1)
    vmap_loader = DataLoader(v_map, batch_size=val_bs, shuffle=False, num_workers=1)

    # Flow Data loader - for Radar to lidar flow validation
    v_flow = R2L_Flow_boreas(dataroot=boreas_dataroot,
                      file_name=boreas_query_file,
                      radar_transform=val_transform, lidar_transform=val_transform,
                      train=False, max_angle=rot_aug, max_trans=trans_aug)
                       
    vflow_loader = DataLoader(v_flow, batch_size=val_bs, shuffle=False, num_workers=1)

    # Run Validation, results stored on wandb if parameter is set to true in the config file
    with torch.no_grad():
        validate_flow(model=model, flow_loader=vflow_loader, cfg=cfg, total_steps=0)
        r1, r3, r5, r10 = validate_pr_with_flow_model(query_model=model, map_model=model,
                                                    query_loader=vquery_loader, map_loader=vmap_loader, cfg=cfg)
