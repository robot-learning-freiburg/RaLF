import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class MLP(nn.Module):
    def __init__(self, in_feat=512, out_feat=512):
        super(MLP, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.mlp1 = nn.Linear(in_features=self.in_feat, out_features=self.out_feat)
        self.mlp2 = nn.Linear(in_features=self.out_feat, out_features=self.out_feat)
        self.flatten = nn.Flatten()

    def forward(self, x):
        out = F.relu(self.mlp1(self.flatten(x)))
        out = F.relu(self.mlp2(out))

        return out


class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, out_c=64, in_c=128, alpha=100.0,
                 normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = out_c
        self.dim = in_c
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(out_c, in_c))
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x):
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)

        # calculate residuals to each clusters
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
                   self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad

if __name__ == '__main__':
    net = NetVLAD(out_c=256, in_c=256)
    # net = MLP(in_feat=29 * 29 * 1024, out_feat=8192)
    print(summary(net.cuda(), (256, 29, 29)))
