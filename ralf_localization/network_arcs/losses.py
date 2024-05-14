import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class triplet_loss(torch.nn.Module):

    def __init__(self, margin):
        super(triplet_loss, self).__init__()

        self.margin = margin
        self.torch_triplet_loss = nn.TripletMarginLoss(margin=self.margin, p=2, reduction='mean')
        self.pdist = nn.PairwiseDistance(p=2)

    def triplet_selector(self, pos_mat, neg_mat, e1=None, e2=None, selector_fn='random'):
        triplets = []
        pos_idx_x, pos_idx_y = np.where(pos_mat == 1)
        neg_idx_x, neg_idx_y = np.where(neg_mat == 1)

        fe1 = e1.contiguous().view(e1.shape[0], -1)
        fe2 = e2.contiguous().view(e2.shape[0], -1)
        
        with torch.no_grad():
            for a_idx in range(pos_mat.shape[0] // 2):
                # p_idx is a single index now
                p_idx = pos_idx_y[(pos_idx_x == a_idx) & (pos_idx_y != a_idx)][0]

                # Set of all possible negative indices
                n_indices = neg_idx_y[(neg_idx_x == a_idx) & (neg_idx_y != a_idx)] 

                if len(n_indices) == 0:
                    continue

                if selector_fn == 'all':
                    pos_dist = self.pdist(e1[a_idx], e2[p_idx])
                    neg_dist = self.pdist(e1[a_idx], e2[n_indices])

                    dist = pos_dist - neg_dist + self.margin
                    possible_neg_indices = np.where(dist.cpu().numpy() > 0.0)[0]

                    neg_indices_to_send = n_indices[possible_neg_indices]

                    for idx in neg_indices_to_send:
                        triplets.append([a_idx, p_idx, idx])

                    continue

                elif selector_fn == 'hard':
                    # Compute minimum distance. Negative index with minimum dist to anchor is the Hardest negative sample           
                    pos_dist = self.pdist(fe1[[a_idx]], fe2[[p_idx]])
                    neg_dist = self.pdist(fe1[[a_idx]], fe2[[n_indices]])

                    dist = pos_dist - neg_dist + self.margin
                    possible_neg_indices = np.where(dist.cpu().numpy() > 0.0)[0]

                    if len(possible_neg_indices) == 0:
                        # All samples have a negative distance. Choose random 
                        n_idx = np.random.choice(n_indices)
                    else:
                        n_idx = np.random.choice(n_indices[possible_neg_indices])

                elif selector_fn == 'random':
                    n_idx = np.random.choice(n_indices)

                else:
                    raise NotImplementedError('Selector function not implemented')

                triplets.append([a_idx, p_idx, n_idx])

        return torch.as_tensor(triplets)

    def triplet_loss_mean(self, anchors, positives, negatives):

        loss = (self.margin + self.pdist(anchors, positives) -\
             self.pdist(anchors, negatives))
        
        loss = loss.clamp(min=0.0)
        loss = torch.mean(loss)
        
        return loss

    def forward(self, pos_mat, neg_mat, radar_features, lidar_features, triplet_selector_fn='hard'):

        triplets = self.triplet_selector(pos_mat, neg_mat, radar_features, lidar_features, selector_fn=triplet_selector_fn)

        if len(triplets) == 0:
            return 0.0, None, None
        
        anchor_indices = triplets[:, 0]
        pos_indices = triplets[:, 1]
        neg_indices = triplets[:, 2]

        anchor_radar = radar_features[anchor_indices,...].reshape(anchor_indices.shape[0],-1)
        anchor_lidar = lidar_features[anchor_indices,...].reshape(anchor_indices.shape[0],-1)

        positive_radar = radar_features[pos_indices,...].reshape(anchor_indices.shape[0],-1)
        positive_lidar = lidar_features[pos_indices,...].reshape(anchor_indices.shape[0],-1)

        negative_radar = radar_features[neg_indices,...].reshape(anchor_indices.shape[0],-1)
        negative_lidar = lidar_features[neg_indices,...].reshape(anchor_indices.shape[0],-1)

        # 2*2*2 cases
        L1 = self.triplet_loss_mean(anchor_radar, positive_radar, negative_radar)
        L2 = self.triplet_loss_mean(anchor_radar, positive_radar, negative_lidar)
        L3 = self.triplet_loss_mean(anchor_radar, positive_lidar, negative_radar)
        L4 = self.triplet_loss_mean(anchor_radar, positive_lidar, negative_lidar)

        L5 = self.triplet_loss_mean(anchor_lidar, positive_radar, negative_radar)
        L6 = self.triplet_loss_mean(anchor_lidar, positive_radar, negative_lidar)
        L7 = self.triplet_loss_mean(anchor_lidar, positive_lidar, negative_radar)
        L8 = self.triplet_loss_mean(anchor_lidar, positive_lidar, negative_lidar)

        # print('[%s %f], [%s %f], [%s %f], [%s %f], [%s %f], [%s %f], [%s %f], [%s %f]'%\
        #     ('rrr', L1.detach().cpu().numpy(), 'rrl', L2.detach().cpu().numpy(), 'rlr', L3.detach().cpu().numpy(),\
        #          'rll', L4.detach().cpu().numpy(), 'lrr',L5.detach().cpu().numpy(), 'lrl', L6.detach().cpu().numpy(),\
        #               'llr', L7.detach().cpu().numpy(), 'lll', L8.detach().cpu().numpy()))

        loss = L1 + L2 + L3 + L4 + L5 + L6 + L7 + L8

        return loss, radar_features, lidar_features


def flow_sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=300):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics
