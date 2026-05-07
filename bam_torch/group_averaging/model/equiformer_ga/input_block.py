import torch
import torch.nn as nn
import copy

from .so3 import SO3_Embedding
from .radial_function import RadialFunction


class EdgeDegreeEmbedding_(torch.nn.Module):
    """

    Args:
        sphere_channels (int):      Number of spherical channels
        
        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution
        
        SO3_rotation (list:SO3_Rotation): Class to calculate Wigner-D matrices and rotate embeddings
        mappingReduced (CoefficientMappingModule): Class to convert l and m indices once node embedding is rotated
        
        max_num_elements (int):     Maximum number of atomic numbers
        edge_channels_list (list:int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].
                                        The last one will be used as hidden size when `use_atom_edge_embedding` is `True`.
        use_atom_edge_embedding (bool): Whether to use atomic embedding along with relative distance for edge scalar features

        rescale_factor (float):     Rescale the sum aggregation
    """

    def __init__(
        self,
        sphere_channels,
        
        lmax_list,
        mmax_list,
        
        #SO3_rotation,
        #mappingReduced,

        max_num_elements,
        edge_channels_list,
        use_atom_edge_embedding,
        
        rescale_factor
    ):
        super(EdgeDegreeEmbedding, self).__init__()
        self.sphere_channels = sphere_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(self.lmax_list)
        #self.SO3_rotation = SO3_rotation
        #self.mappingReduced = mappingReduced
        
        self.m_0_num_coefficients = self.mappingReduced.m_size[0] 
        self.m_all_num_coefficents = len(self.mappingReduced.l_harmonic)

        # Create edge scalar (invariant to rotations) features
        # Embedding function of the atomic numbers
        self.max_num_elements = max_num_elements
        self.edge_channels_list = copy.deepcopy(edge_channels_list)
        self.use_atom_edge_embedding = use_atom_edge_embedding

        if self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
            nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)
            self.edge_channels_list[0] = self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
        else:
            self.source_embedding, self.target_embedding = None, None

        # Embedding function of distance
        self.edge_channels_list.append(self.m_0_num_coefficients * self.sphere_channels)
        self.rad_func = RadialFunction(self.edge_channels_list)

        self.rescale_factor = rescale_factor


    def forward(
        self,
        atomic_numbers,
        edge_distance,
        edge_index
    ):    
        
        if self.use_atom_edge_embedding:
            source_element = atomic_numbers[edge_index[0]]  # Source atom atomic number
            target_element = atomic_numbers[edge_index[1]]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            x_edge = torch.cat((edge_distance, source_embedding, target_embedding), dim=1)
        else:
            x_edge = edge_distance

        x_edge_m_0 = self.rad_func(x_edge)
        x_edge_m_0 = x_edge_m_0.reshape(-1, self.m_0_num_coefficients, self.sphere_channels)
        x_edge_m_pad = torch.zeros((
            x_edge_m_0.shape[0], 
            (self.m_all_num_coefficents - self.m_0_num_coefficients), 
            self.sphere_channels), 
            device=x_edge_m_0.device)
        x_edge_m_all = torch.cat((x_edge_m_0, x_edge_m_pad), dim=1)

        x_edge_embedding = SO3_Embedding(
            0, 
            self.lmax_list.copy(), 
            self.sphere_channels, 
            device=x_edge_m_all.device, 
            dtype=x_edge_m_all.dtype
        )
        x_edge_embedding.set_embedding(x_edge_m_all)
        x_edge_embedding.set_lmax_mmax(self.lmax_list.copy(), self.mmax_list.copy())

        # Reshape the spherical harmonics based on l (degree)
        x_edge_embedding._l_primary(self.mappingReduced)

        # Rotate back the irreps
        x_edge_embedding._rotate_inv(self.SO3_rotation, self.mappingReduced)

        # Compute the sum of the incoming neighboring messages for each target node
        x_edge_embedding._reduce_edge(edge_index[1], atomic_numbers.shape[0])
        x_edge_embedding.embedding = x_edge_embedding.embedding / self.rescale_factor

        return x_edge_embedding


import copy
import torch
import torch.nn as nn

class EdgeDegreeEmbedding(nn.Module):
    def __init__(
        self,
        sphere_channels,
        lmax_list,
        max_num_elements,
        edge_channels_list,
        use_atom_edge_embedding,
        rescale_factor,
    ):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.lmax_list = lmax_list
        self.num_resolutions = len(lmax_list)

        # m=0 count = sum_l (1 per l)  -> (lmax+1) per resolution
        self.m0_num_coeff = sum(lmax + 1 for lmax in lmax_list)
        # total coeff = sum (lmax+1)^2 per resolution
        self.all_num_coeff = sum((lmax + 1) ** 2 for lmax in lmax_list)

        self.max_num_elements = max_num_elements
        self.edge_channels_list = copy.deepcopy(edge_channels_list)
        self.use_atom_edge_embedding = use_atom_edge_embedding

        if self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
            nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)
            self.edge_channels_list[0] = self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
        else:
            self.source_embedding, self.target_embedding = None, None

        # RadialFunction output: m0_num_coeff * sphere_channels
        self.edge_channels_list.append(self.m0_num_coeff * self.sphere_channels)
        self.rad_func = RadialFunction(self.edge_channels_list)

        self.rescale_factor = rescale_factor

    def forward(self, atomic_numbers, edge_distance, edge_index):
        """
        atomic_numbers: [N] (long)
        edge_distance:  [E, d]  (예: d=1 or RBF expanded distance 등)
        edge_index:     [2, E]  (source, target)
        """
        E = edge_index.shape[1]
        N = atomic_numbers.shape[0]
        device = edge_distance.device
        dtype = edge_distance.dtype

        # 1) edge scalar feature 만들기
        if self.use_atom_edge_embedding:
            src_Z = atomic_numbers[edge_index[0]]
            dst_Z = atomic_numbers[edge_index[1]]
            src_emb = self.source_embedding(src_Z)  # [E, emb]
            dst_emb = self.target_embedding(dst_Z)  # [E, emb]
            x_edge = torch.cat([edge_distance, src_emb, dst_emb], dim=1)
        else:
            x_edge = edge_distance

        # 2) RadialFunction -> [E, m0_num_coeff * sphere_channels]
        x = self.rad_func(x_edge)
        x = x.view(E, self.m0_num_coeff, self.sphere_channels)  # [E, m0, C]

        # 3) (equivariance 제거) 전체 coeff 슬롯을 만들되, m=0 부분만 채우고 나머지는 0
        x_all = torch.zeros(E, self.all_num_coeff, self.sphere_channels, device=device, dtype=dtype)
        x_all[:, :self.m0_num_coeff, :] = x

        # 4) target node로 sum aggregation
        out = torch.zeros(N, self.all_num_coeff, self.sphere_channels, device=device, dtype=dtype)
        out.index_add_(0, edge_index[1], x_all)
        out = out / self.rescale_factor

        return out  # [N, all_num_coeff, sphere_channels]