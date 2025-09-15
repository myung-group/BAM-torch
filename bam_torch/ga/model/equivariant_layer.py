import torch
from torch import nn, Tensor as T
from torch.nn import functional as F

from .vnn.vn_dgcnn_util import get_graph_feature
from .vnn.vn_layers import VNLinearLeakyReLU, VNMaxPool, VNStdFeature, VNLinear
from bam_torch.ga.utils.utils import batched_gram_schmidt_3d
#from interface import PermutaionMatrixPenalty


class VNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=96, n_knn=4, dropout=0.0):
        super().__init__()
        self.dim = hidden_dim // 3
        self.pooled_dim = hidden_dim // 3
        self.input_dim = input_dim
        self.n_knn = n_knn
        # build equivariant layers
        self.conv1 = VNLinearLeakyReLU(input_dim*2, self.dim)
        self.conv2 = VNLinearLeakyReLU(self.dim*2, self.dim)
        self.conv3 = VNLinearLeakyReLU(self.dim*2, self.pooled_dim, dim=4, share_nonlinearity=True)
        self.pool1 = VNMaxPool(self.dim)
        self.pool2 = VNMaxPool(self.dim)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        # rotation invariant pooling
        self.std_feature = VNStdFeature(self.pooled_dim*2, dim=4, normalize_frame=False)
        self.head1 = nn.Linear(self.pooled_dim*6 + self.dim*6, 1)
        # permutation invariant pooling
        self.pool3 = VNMaxPool(self.dim)
        self.head2 = VNLinear(self.dim, 3)

    def _permutation_invariant_rotation_equivariant_pool(self, x):
        b, c, _, n = x.shape
        assert x.shape == (b, c, 3, n)
        x = self.pool3(x)
        _, d, _ = x.shape
        assert x.shape == (b, d, 3)
        x = self.head2(x)  # [b, c=3, 3]
        assert x.shape == (b, 3, 3)
        return x

    def forward(self, x):
        b, c, _, n = x.shape
        assert x.shape == (b, c, 3, n)
        # construct feature for graph convolution, (b, c', 3, n, k)
        x = get_graph_feature(x, k=self.n_knn)
        x = self.conv1(x)
        x1 = self.pool1(x)  # [b, c', 3, n]
        x = self.dropout1(x1)
        x = get_graph_feature(x1, k=self.n_knn)
        x = self.conv2(x)
        x2 = self.pool2(x)  # [b, c', 3, n]
        x2 = self.dropout2(x2)
        x12 = torch.cat((x1, x2), dim=1)
        x = self.conv3(x12)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        # Sn x O(3) equivariant until here
        # produce hidden features for group representations by pooling
        pseudo_ks = self._permutation_invariant_rotation_equivariant_pool(x1)
        assert pseudo_ks.shape == (b, 3, 3)
        return pseudo_ks


class EquivariantInterface(nn.Module):
    def __init__(
        self,
        symmetry='SO3',
        interface='prob',
        fixed_noise=False,
        noise_scale=0.1,
        tau=0.01,
        hard=True,
        vnn_hidden_dim=96,
        vnn_k_nearest_neighbors=4,
        vnn_dropout=0.1
    ):
        super().__init__()
        assert symmetry in ['SO3', 'O3']
        assert interface in ['prob', 'unif']
        self.symmetry = symmetry
        self.interface = interface
        self.fixed_noise = fixed_noise
        self.noise_scale = noise_scale
        self.tau = tau
        self.hard = hard
        self.vnn_interface = VNN(
            input_dim=2,
            hidden_dim=vnn_hidden_dim,
            n_knn=vnn_k_nearest_neighbors,
            dropout=vnn_dropout
        )
        # self.compute_entropy_loss = PermutaionMatrixPenalty(n=5)

    def _postprocess_rotation(self, pseudo_ks, eps=1e-6):
        """Obtain rotation component (b, k, 3, 3) from hidden representation"""
        # note: this assumes left equivariance, i.e., pseudo_ks: (b, k, 3, C=3)
        # pseudo_ks: GL(N)
        device = pseudo_ks.device
        b, k, _, _ = pseudo_ks.shape
        assert pseudo_ks.shape == (b, k, 3, 3)
        # add small noise to prevent rank collapse
        pseudo_ks = pseudo_ks + eps * torch.randn_like(pseudo_ks, device=device)
        pseudo_ks = pseudo_ks.view(b*k, 3, 3)
        # use gram-schmidt to obtain orthogonal matrix
        ks = batched_gram_schmidt_3d(pseudo_ks)  # O(3)
        assert ks.shape == (b*k, 3, 3)
        if self.symmetry in ('SnxSO3', 'SO3'):
            # SO(3) equivariant map that maps O(3) matrix to SO(3) matrix
            # determinant are +- 1
            deter_ks = torch.linalg.det(ks)
            assert deter_ks.shape == (b*k,)
            # multiply the first column
            sign_arr = torch.ones(b*k, 3, device=device)
            #sign_arr = sign_arr.clone()
            sign_arr[:, 0] = deter_ks
            #sign_arr = torch.cat([deter_ks.unsqueeze(1), sign_arr[:, 1:]], dim=1)
            sign_arr = sign_arr[:, None, :].expand(b*k, 3, 3)
            # elementwise multiplication
            ks = ks * sign_arr
        ks = ks.reshape(b, k, 3, 3)
        return ks

    def sample_invariant_noise(self, x, idx):
        _, k, n, _, d_node = x.shape
        if self.fixed_noise:
            zs = []
            for i in idx.tolist():
                seed = torch.seed()
                torch.manual_seed(i)
                z = torch.zeros(k, n, 3, d_node, device=x.device, dtype=x.dtype)
                z = z.normal_(0, self.noise_scale)
                zs.append(z)
                torch.manual_seed(seed)
            z = torch.stack(zs, dim=0)
        else:
            z = torch.zeros_like(x).normal_(0, self.noise_scale)
        return z

    def _forward_prob(self, node_features, idx, k: int):
        # k is the number of interface samples
        b, n, _, d_node = node_features.shape
        assert node_features.shape == (b, n, 3, d_node)
        # replicate input k times
        x = node_features[:, None, :, :, :].expand(b, k, n, 3, d_node)
        # add noise
        x = x + self.sample_invariant_noise(x, idx)
        x = x.view(b*k, n, 3, d_node)
        # SnxO(3) equivariant procedure that returns hidden representation
        x = x.permute(0, 3, 2, 1).contiguous()
        assert x.shape == (b*k, d_node, 3, n)
        pseudo_ks = self.vnn_interface(x)
        assert pseudo_ks.shape == (b*k, 3, 3)  # [b*k, c=3, 3]
        pseudo_ks = pseudo_ks.transpose(1, 2)  # [b*k, 3, c=3]
        pseudo_ks = pseudo_ks.reshape(b, k, 3, 3)
        # post-processing for permutation matrix
        # hs, entropy_loss = self._postprocess_permutation(pseudo_hs)
        # assert hs.shape == (b, k, n, n)
        # post-processing for SO(3) or O(3) matrix
        ks = self._postprocess_rotation(pseudo_ks)
        assert ks.shape == (b, k, 3, 3)
        return ks #entropy_loss

    def _forward_unif(self, node_features, idx, k: int):
        b, n, _, _ = node_features.shape
        device = node_features.device
        # sample Sn representation
        assert self.hard
        if self.fixed_noise:
            raise NotImplementedError
        indices = torch.randn(b*k, n, device=device).argsort(dim=-1)
        # sample O(3) or SO(3) representation
        if self.symmetry in ('O3'):
            ks = torch.randn(b*k, 3, 3, device=device)
            ks = batched_gram_schmidt_3d(ks)
            ks = ks.reshape(b, k, 3, 3)
        elif self.symmetry in ('SO3'):
            ks = torch.randn(b*k, 3, 3, device=device)
            ks = batched_gram_schmidt_3d(ks)
            # SO(3) equivariant map that maps O(3) matrix to SO(3) matrix
            # determinant are +- 1
            deter_ks = torch.linalg.det(ks)
            assert deter_ks.shape == (b*k,)
            # multiply the first column
            sign_arr = torch.ones(b*k, 3, device=device)
            #sign_arr = sign_arr.clone()
            sign_arr[:, 0] = deter_ks
            #sign_arr = torch.cat([deter_ks.unsqueeze(1), sign_arr[:, 1:]], dim=1)
            sign_arr = sign_arr[:, None, :].expand(b*k, 3, 3)
            ks = ks * sign_arr
            ks = ks.reshape(b, k, 3, 3)
        else:
            raise NotImplementedError
        ks
        return ks

    
    def forward(self, node_features, edge_features, idx, k):
        # k is the number of interface samples
        b, n, _, d_node = node_features.shape
        _, _, _, d_edge = edge_features.shape
        assert node_features.shape == (b, n, 3, d_node)
        assert edge_features.shape == (b, n, n, d_edge)
        assert idx.shape == (b,)
        # sample group representation
        if self.interface == 'prob':
            gs = self._forward_prob(node_features, idx, k)
            #if self.symmetry in ('O3', 'SO3'):
            #    # entropy loss is only for permutation involved groups
            #    entropy_loss = torch.tensor(0, device=node_features.device)
            return gs
        if self.interface == 'unif':
            gs = self._forward_unif(node_features, idx, k)
            return gs, torch.tensor(0, device=node_features.device)
        raise NotImplementedError

