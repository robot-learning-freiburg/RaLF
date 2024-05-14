from pip import main
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchsummary import summary
import ipdb

class NetVLADLoupe(nn.Module):
    def __init__(self, feature_size, cluster_size, output_dim,
                 gating=True, add_norm=True, is_training=True, normalization='batch'):
        super(NetVLADLoupe, self).__init__()
        self.feature_size = feature_size
        self.output_dim = output_dim
        self.is_training = is_training
        self.gating = gating
        self.add_batch_norm = add_norm
        self.cluster_size = cluster_size
        if normalization == 'instance':
            norm = lambda x: nn.LayerNorm(x)
        elif normalization == 'group':
            norm = lambda x: nn.GroupNorm(cfg.GROUP_NORM_NUM, x)
        else:
            norm = lambda x: nn.BatchNorm1d(x)
        self.softmax = nn.Softmax(dim=-1)
        self.cluster_weights = nn.Parameter(torch.randn(feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.cluster_weights2 = nn.Parameter(torch.randn(1, feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.hidden1_weights = nn.Parameter(
            torch.randn(cluster_size * feature_size, output_dim) * 1 / math.sqrt(feature_size))

        if add_norm:
            self.cluster_biases = None
            self.bn1 = norm(cluster_size)
        else:
            self.cluster_biases = nn.Parameter(torch.randn(cluster_size) * 1 / math.sqrt(feature_size))
            self.bn1 = None

        self.bn2 = norm(output_dim)

        if gating:
            self.context_gating = GatingContext(output_dim, add_norm=add_norm, normalization=normalization)

    def forward(self, x):
        #print("MAXX: ",x.max())
        x = x.transpose(1, 3).contiguous()
        batch_size = x.shape[0]
        feature_size = x.shape[-1]
        x = x.view((batch_size, -1, feature_size))
        max_samples = x.shape[1]
        activation = torch.matmul(x, self.cluster_weights)
        if self.add_batch_norm:
            # activation = activation.transpose(1,2).contiguous()
            activation = activation.view(-1, self.cluster_size)
            activation = self.bn1(activation)
            activation = activation.view(-1, max_samples, self.cluster_size)
            # activation = activation.transpose(1,2).contiguous()
        else:
            activation = activation + self.cluster_biases
        activation = self.softmax(activation)
        #activation = activation.view((-1, max_samples, self.cluster_size))

        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weights2

        activation = torch.transpose(activation, 2, 1)
        x = x.view((-1, max_samples, self.feature_size))
        vlad = torch.matmul(activation, x)
        vlad = torch.transpose(vlad, 2, 1)
        vlad0 = vlad - a

        vlad1 = F.normalize(vlad0, dim=1, p=2, eps=1e-6)
        vlad2 = vlad1.contiguous().view((-1, self.cluster_size * self.feature_size))
        vlad = F.normalize(vlad2, dim=1, p=2, eps=1e-6)

        vlad = torch.matmul(vlad, self.hidden1_weights)

        vlad = self.bn2(vlad)

        if self.gating:
            vlad = self.context_gating(vlad)

        return vlad

class GatingContext(nn.Module):
    def __init__(self, dim, add_norm=True, normalization='batch'):
        super(GatingContext, self).__init__()
        self.dim = dim
        self.add_norm = add_norm
        self.gating_weights = nn.Parameter(torch.randn(dim, dim) * 1 / math.sqrt(dim))
        self.sigmoid = nn.Sigmoid()

        if normalization == 'instance':
            norm = lambda x: nn.LayerNorm(x)
        elif normalization == 'group':
            norm = lambda x: nn.GroupNorm(cfg.GROUP_NORM_NUM, x)
        else:
            norm = lambda x: nn.BatchNorm1d(x)

        if add_norm:
            self.gating_biases = None
            self.bn1 = norm(dim)
        else:
            self.gating_biases = nn.Parameter(torch.randn(dim) * 1 / math.sqrt(dim))
            self.bn1 = None

    def forward(self,x):
        gates = torch.matmul(x, self.gating_weights)

        if self.add_norm:
            gates = self.bn1(gates)
        else:
            gates = gates + self.gating_biases

        gates = self.sigmoid(gates)

        activation = x * gates

        return activation

class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128, alpha=100.0,
                 normalize_input=True, output_dim = 256):
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
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self.output_fc = nn.Linear(num_clusters * dim, output_dim)
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1).data.clone()
        )
        self.conv.bias = nn.Parameter(
            (- self.alpha * self.centroids.norm(dim=1)).data.clone()
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
        vlad = self.output_fc(vlad)

        return vlad

class EmbedNet(nn.Module):
    def __init__(self, base_model, net_vlad):
        super(EmbedNet, self).__init__()
        self.base_model = base_model
        self.net_vlad = net_vlad

    def forward(self, x):
        x = self.base_model(x)
        embedded_x = self.net_vlad(x)
        return embedded_x

class TripletNet(nn.Module):
    def __init__(self, embed_net):
        super(TripletNet, self).__init__()
        self.embed_net = embed_net

    def forward(self, a, p, n):
        embedded_a = self.embed_net(a)
        embedded_p = self.embed_net(p)
        embedded_n = self.embed_net(n)
        return embedded_a, embedded_p, embedded_n

    def feature_extract(self, x):
        return self.embed_net(x)

class CNNNeck(nn.Module):
    def __init__(self, feature_size=256):
        super(CNNNeck, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(feature_size, 256, kernel_size=3, stride=1),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(),
                                  nn.Conv2d(256, 128, kernel_size=3, stride=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(),
                                  nn.Conv2d(128, 128, kernel_size=3, stride=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(),
                                  nn.Conv2d(128, 128, kernel_size=3, stride=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU())

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    model = NetVLADLoupe(feature_size=512, cluster_size=64, output_dim=128)
    summary(model.cuda(), (512, 30, 30))