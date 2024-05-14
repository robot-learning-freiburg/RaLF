import os
from ralf_localization.network_arcs.networks import *
import torch
import faiss
import numpy as np
import ipdb
import argparse
import json


def save_configs(cfg, save_path):
    config = cfg
    with open(save_path, 'w') as outfile:
        json.dump(config, outfile)


def load_cfg(config_file):
    f = open(config_file)
    configs = json.load(f)
    return configs


def create_optimizer(cfg, params):
    if cfg['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params , lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
    elif cfg['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    elif cfg['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=cfg['lr'])
    else:
        raise NotImplementedError("Optimizer type not implemented!")
    
    return optimizer 


def create_scheduler(cfg, optimizer):
    if cfg['lr_policy'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['lr_step_size'], gamma=cfg['lr_gamma'])
    elif cfg['lr_policy'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    elif cfg['lr_policy'] == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, cfg['lr'], cfg['max_steps'],
                                                        pct_start=cfg['pct_start'], cycle_momentum=False, anneal_strategy='linear')
    elif cfg['lr_policy'] == 'none':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['num_epochs'], gamma=1.0)
    else:
        raise NotImplementedError("Scheduler not implemented!")
    
    return scheduler


def generate_positive_negative_matrix(poses, neg_dist=80.0):
    pos_mat = np.zeros((poses.shape[0], poses.shape[0]))
    neg_mat = np.zeros((poses.shape[0], poses.shape[0]))
    
    d = 2
    index = faiss.IndexFlatL2(d)
    D = faiss.pairwise_distances(poses, poses)

    k = poses.shape[0]
    
    for i in range(k//2):
        pos_mat[i, i + k // 2] = 1
        pos_mat[i + k // 2, i] = 1

    neg_mat[D > (neg_dist ** 2)] = 1

    return pos_mat, neg_mat
