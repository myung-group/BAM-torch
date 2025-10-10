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

    def _permutation_equivariant_rotation_invariant_pool(self, x12, x):
        b, c, _, n = x.shape
        assert x.shape == (b, c, 3, n)
        assert x12.shape == (b, c, 3, n)
        x, z0 = self.std_feature(x)
        assert x.shape == (b, c, 3, n)
        assert z0.shape == (b, 3, 3, n)
        # rotation invariant pooling
        x12 = torch.einsum('bijm,bjkm->bikm', x12, z0)
        assert x12.shape == (b, c, 3, n)
        x12 = x12.view(b, c*3, n)
        # max-pooling and broadcasting across nodes
        x = x.view(b, c*3, n)
        x = x.max(dim=-1, keepdim=True)[0].expand(x.size())
        # combine rotation invariant features
        x = torch.cat((x, x12), dim=1)
        # map to scalar
        x = self.head1(x.permute(0, 2, 1)).squeeze(-1)
        assert x.shape == (b, n)
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
        pseudo_hs = self._permutation_equivariant_rotation_invariant_pool(x12, x)
        pseudo_ks = self._permutation_invariant_rotation_equivariant_pool(x1)
        assert pseudo_hs.shape == (b, n) and pseudo_ks.shape == (b, 3, 3)
        return pseudo_hs, pseudo_ks


class PermutaionMatrixPenalty(nn.Module):
    def __init__(self): #, n):
        super().__init__()
        #self.n = n

    @staticmethod
    def _normalize(scores, axis, eps=1e-12):
        normalizer = torch.sum(scores, axis).clamp(min=eps)
        normalizer = normalizer.unsqueeze(axis)
        prob = torch.div(scores, normalizer)
        return prob

    @staticmethod
    def _entropy(prob, axis):
        return -torch.sum(prob * prob.log().clamp(min=-100), axis)

    def entropy(self, scores, eps=1e-12):
        b, k, n, _ = scores.shape
        # clamp min to avoid zero logarithm
        scores = scores.clamp(min=eps)
        # compute columnwise entropy
        col_prob = self._normalize(scores, axis=2)
        entropy_col = self._entropy(col_prob, axis=2)
        # compute rowwise entropy
        row_prob = self._normalize(scores, axis=3)
        entropy_row = self._entropy(row_prob, axis=3)
        # return entropy
        assert entropy_col.shape == entropy_row.shape == (b, k, n)
        return entropy_col, entropy_row

    def forward(self, perm_soft):
        b, k, n, _ = perm_soft.shape
        #assert n == self.n
        assert perm_soft.shape == (b, k, n, n)
        # compute entropy
        entropy_col, entropy_row = self.entropy(perm_soft)
        # compute mean over samples
        loss = entropy_col.mean(1) + entropy_row.mean(1)
        # compute mean over nodes
        loss = loss.mean(1)
        # compute mean over batch
        loss = loss.mean()
        return loss


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
        self.compute_entropy_loss = PermutaionMatrixPenalty()

    def _postprocess_permutation(self, pseudo_hs: T, sinkhorn_iter=20):
        """Obtain permutation component (b, k, n, n) from hidden representation"""
        scores = pseudo_hs
        b, k, n = scores.shape
        # add small noise to break ties
        # without this, models can exploit the ordering of tied scores
        scores = scores + torch.zeros_like(scores).uniform_(0, 1e-6)
        # normalize scores
        # this does not affect hard permutation, but affects
        # straight-through gradient from soft permutation matrix
        scores = F.normalize(scores, p=2.0, dim=-1)
        # sort scores, result is unique up to permutations of tied scores
        scores = scores[:, :, :, None]
        scores_sorted, indices = scores.sort(descending=True, dim=2)
        scores = scores.expand(b, k, n, n)
        scores_sorted = scores_sorted.transpose(2, 3).expand(b, k, n, n)
        # softsort + log sinkhorn operator for computing soft permutation matrix
        log_perm_soft = (scores - scores_sorted).abs().neg() / self.tau
        for _ in range(sinkhorn_iter):
            log_perm_soft = log_perm_soft - torch.logsumexp(log_perm_soft, dim=-1, keepdim=True)
            log_perm_soft = log_perm_soft - torch.logsumexp(log_perm_soft, dim=-2, keepdim=True)
        perm_soft = log_perm_soft.exp()
        hs = perm_soft
        if self.hard:
            # argsort for hard permutation matrix
            with torch.no_grad():
                perm_hard = torch.zeros_like(perm_soft).scatter(dim=-1, index=indices, value=1)
                perm_hard = perm_hard.transpose(2, 3)
                # (optional) test if perm_hard is a permutation matrix
                assert torch.allclose(perm_hard.sum(-1), torch.ones_like(perm_hard.sum(-1)))
                assert torch.allclose(perm_hard.sum(-2), torch.ones_like(perm_hard.sum(-2)))
            # differentiability with straight-through gradient
            # the estimated gradient is accurate if perm_soft is close to perm_hard
            # for this, entropy regularization is necessary
            perm_hard = (perm_hard - perm_soft).detach() + perm_soft
            hs = perm_hard
        # compute entropy loss
        entropy_loss = self.compute_entropy_loss(perm_soft)
        return hs, entropy_loss
        
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
        ###
        pseudo_hs, pseudo_ks = self.vnn_interface(x)
        assert pseudo_hs.shape == (b*k, n)
        assert pseudo_ks.shape == (b*k, 3, 3)  # [b*k, c=3, 3]
        pseudo_ks = pseudo_ks.transpose(1, 2)  # [b*k, 3, c=3]
        pseudo_ks = pseudo_ks.reshape(b, k, 3, 3)
        pseudo_hs = pseudo_hs.reshape(b, k, n)
        # post-processing for permutation matrix
        hs, entropy_loss = self._postprocess_permutation(pseudo_hs)
        assert hs.shape == (b, k, n, n)
        # post-processing for SO(3) or O(3) matrix
        ks = self._postprocess_rotation(pseudo_ks)
        assert ks.shape == (b, k, 3, 3)
        gs = (hs, ks)
        return gs ,entropy_loss

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
            gs, entropy_loss = self._forward_prob(node_features, idx, k)
            #if self.symmetry in ('O3', 'SO3'):
            #    # entropy loss is only for permutation involved groups
            #    entropy_loss = torch.tensor(0, device=node_features.device)
            return gs, entropy_loss
        if self.interface == 'unif':
            gs = self._forward_unif(node_features, idx, k)
            return gs, torch.tensor(0, device=node_features.device)
        raise NotImplementedError

