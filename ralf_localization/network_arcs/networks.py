import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from ralf_localization.network_arcs.encoder import *
from ralf_localization.network_arcs.necks import *
from ralf_localization.network_arcs.netvlad import *
from ralf_localization.network_arcs.update import BasicUpdateBlock
from ralf_localization.network_arcs.corr import CorrBlock, AlternateCorrBlock    

from ralf_localization.network_utils.utils import bilinear_sampler, coords_grid, upflow8


def get_encoder(cfg):
    encoder = BasicEncoder(output_dim=cfg['output_dim'], norm_fn='instance', dropout=cfg['dropout'])
        
    return encoder


def get_neck(cfg):
    neck = CNNNeck(feature_size=cfg['neck_feature_size'])

    return neck


def get_context_network(cfg, hdim, cdim):
    cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=cfg['dropout'])

    return cnet


def get_update_block(cfg, hdim):
    updater = BasicUpdateBlock(cfg=cfg, hidden_dim=hdim)
    
    return updater


class RaLF(nn.Module):
    def __init__(self, cfg):
        super(RaLF, self).__init__()
        self.cfg = cfg
        
        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        self.cfg['corr_levels'] = 4
        self.cfg['corr_radius'] = 4

        if 'dropout' not in self.cfg:
            self.cfg['dropout'] = 0

        if 'alternate_corr' not in cfg:
            self.cfg['alternate_corr'] = False

        # Encoder Networks for Radar and Lidar images
        self.fnet_r = get_encoder(cfg)
        self.fnet_l = get_encoder(cfg)

        # Add a PR head for place recognition
        self.neck = get_neck(cfg)

        # Context Network
        self.cnet = get_context_network(cfg, hdim, cdim)

        # Update Block
        self.update_block = get_update_block(cfg, hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1


    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, lidar_, radar_, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        lidar_ = 2 * (lidar_) - 1.0
        radar_ = 2 * (radar_) - 1.0

        lidar_ = lidar_.contiguous()
        radar_ = radar_.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        fmapl, fmapr = self.fnet_l(lidar_), self.fnet_r(radar_)
                    
        fmapl = fmapl.float()
        fmapr = fmapr.float()

        if self.cfg['alternate_corr']:
            corr_fn = AlternateCorrBlock(fmapl, fmapr, radius=self.cfg['corr_radius'])
        else:
            corr_fn = CorrBlock(fmapl, fmapr, radius=self.cfg['corr_radius'])

        # run the context network
        cnet = self.cnet(lidar_)
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(lidar_)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up)

        fmapl = self.neck(fmapl)
        fmapr = self.neck(fmapr)

        if test_mode:
            return fmapl, fmapr, coords1 - coords0, flow_up
            
        return fmapl, fmapr, flow_predictions
