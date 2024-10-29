import paddle
from paddle import nn
import paddle.nn.functional as F
import math
import pgl
from rdkit.Chem import rdchem


def rdchem_enum_to_list(values):
    """values = {0: rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED, 
            1: rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW, 
            2: rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW, 
            3: rdkit.Chem.rdchem.ChiralType.CHI_OTHER}
    """
    return [values[i] for i in range(len(values))]

# allowable multiple choice node and edge features 
'''
allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)) + ['misc'],
    'possible_chirality_list' : rdchem_enum_to_list(rdchem.ChiralType.values),
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list' : rdchem_enum_to_list(rdchem.HybridizationType.values),
    'possible_is_aromatic_list': [0, 1],
    'possible_is_in_ring_list': [0, 1],
    'possible_bond_type_list' : rdchem_enum_to_list(rdchem.BondType.values),
    'possible_bond_stereo_list': rdchem_enum_to_list(rdchem.BondStereo.values), 
    'possible_is_conjugated_list': [0, 1],
}
'''
allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)) + ['misc'],
    'possible_chirality_list' : rdchem_enum_to_list(rdchem.ChiralType.values),
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list':  [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_explicit_valence_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list' : rdchem_enum_to_list(rdchem.HybridizationType.values),
    'possible_is_aromatic_list': [0, 1],
    'possible_is_in_ring_list': [0, 1],
    'possible_bond_dir_list': list(range(1, 50)), 
    'possible_bond_type_list' : rdchem_enum_to_list(rdchem.BondType.values),
    'possible_is_in_ring_list': [1,2,3,4,5,6,7], 
    #'possible_bond_length_list': [float],
}

def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1
# # miscellaneous case
# i = safe_index(allowable_features['possible_atomic_num_list'], 'asdf')
# assert allowable_features['possible_atomic_num_list'][i] == 'misc'
# # normal case
# i = safe_index(allowable_features['possible_atomic_num_list'], 2)
# assert allowable_features['possible_atomic_num_list'][i] == 2

# def atom_to_feature_vector(atom):
#     """
#     Converts rdkit atom object to feature list of indices
#     :param mol: rdkit atom object
#     :return: list
#     """
#     atom_feature = [
#             safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
#             safe_index(allowable_features['possible_chirality_list'], str(atom.GetChiralTag())),
#             safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
#             safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
#             safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
#             safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
#             safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
#             allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
#             allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
#             ]
#     return atom_feature
# from rdkit import Chem
# mol = Chem.MolFromSmiles('Cl[C@H](/C=C/C)Br')
# atom = mol.GetAtomWithIdx(1)  # chiral carbon
# atom_feature = atom_to_feature_vector(atom)
# assert atom_feature == [5, 2, 4, 5, 1, 0, 2, 0, 0]

# 0 for padding
def get_atom_feature_dims():
    return dict(
        atomic_num=len(allowable_features['possible_atomic_num_list']) + 1,
        chiral_tag=len(allowable_features['possible_chirality_list']) + 1,
        degree=len(allowable_features['possible_degree_list']) + 1,
        formal_charge=len(allowable_features['possible_formal_charge_list']) + 1,
        explicit_valence=len(allowable_features['possible_explicit_valence_list']) + 1,
        total_numHs=len(allowable_features['possible_numH_list']) + 1,
        num_radical_e=len(allowable_features['possible_number_radical_e_list']) + 1,
        hybridization=len(allowable_features['possible_hybridization_list']) + 1,
        is_aromatic=len(allowable_features['possible_is_aromatic_list']) + 1,
        atom_is_in_ring=len(allowable_features['possible_is_in_ring_list']) + 1,
    )

# def bond_to_feature_vector(bond):
#     """
#     Converts rdkit bond object to feature list of indices
#     :param mol: rdkit bond object
#     :return: list
#     """
#     bond_feature = [
#                 safe_index(allowable_features['possible_bond_type_list'], bond.GetBondType()),
#                 safe_index(allowable_features['possible_bond_stereo_list'], bond.GetStereo()),
#                 allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated()),
#             ]
#     return bond_feature
# uses same molecule as atom_to_feature_vector test
# bond = mol.GetBondWithIdx(2)  # double bond with stereochem
# bond_feature = bond_to_feature_vector(bond)
# assert bond_feature == [1, 2, 0]

# 0 for padding, N + 2 for self-loop
def get_bond_feature_dims():
    #return dict(
    #    bond_type=len(allowable_features['possible_bond_type_list']) + 3,
    #    bond_stereo=len(allowable_features['possible_bond_stereo_list']) + 3,
    #    is_conjugated=len(allowable_features['possible_is_conjugated_list']) + 3
    #)

    return dict(
            bond_dir=len(allowable_features['possible_bond_dir_list']) + 3,
            bond_type=len(allowable_features['possible_bond_type_list']) + 3,
            is_in_ring=len(allowable_features['possible_is_in_ring_list']) + 3,
            #bond_length=len(allowable_features['possible_bond_length_list']) + 3
        )

# def atom_feature_vector_to_dict(atom_feature):
#     [atomic_num_idx, 
#     chirality_idx,
#     degree_idx,
#     formal_charge_idx,
#     num_h_idx,
#     number_radical_e_idx,
#     hybridization_idx,
#     is_aromatic_idx,
#     is_in_ring_idx] = atom_feature

#     feature_dict = {
#         'atomic_num': allowable_features['possible_atomic_num_list'][atomic_num_idx],
#         'chiral_tag': allowable_features['possible_chirality_list'][chirality_idx],
#         'degree': allowable_features['possible_degree_list'][degree_idx],
#         'formal_charge': allowable_features['possible_formal_charge_list'][formal_charge_idx],
#         'total_numHs': allowable_features['possible_numH_list'][num_h_idx],
#         'num_radical_e': allowable_features['possible_number_radical_e_list'][number_radical_e_idx],
#         'hybridization': allowable_features['possible_hybridization_list'][hybridization_idx],
#         'is_aromatic': allowable_features['possible_is_aromatic_list'][is_aromatic_idx],
#         'atom_is_in_ring': allowable_features['possible_is_in_ring_list'][is_in_ring_idx]
#     }

#     return feature_dict
# # uses same atom_feature as atom_to_feature_vector test
# atom_feature_dict = atom_feature_vector_to_dict(atom_feature)
# assert atom_feature_dict['atomic_num'] == 6
# assert atom_feature_dict['chirality'] == 'CHI_TETRAHEDRAL_CCW'
# assert atom_feature_dict['degree'] == 4
# assert atom_feature_dict['formal_charge'] == 0
# assert atom_feature_dict['num_h'] == 1
# assert atom_feature_dict['num_rad_e'] == 0
# assert atom_feature_dict['hybridization'] == 'SP3'
# assert atom_feature_dict['is_aromatic'] == False
# assert atom_feature_dict['is_in_ring'] == False

# def bond_feature_vector_to_dict(bond_feature):
#     [bond_type_idx, 
#     bond_stereo_idx,
#     is_conjugated_idx] = bond_feature

#     feature_dict = {
#         'bond_type': allowable_features['possible_bond_type_list'][bond_type_idx],
#         'bond_stereo': allowable_features['possible_bond_stereo_list'][bond_stereo_idx],
#         'is_conjugated': allowable_features['possible_is_conjugated_list'][is_conjugated_idx]
#     }

#     return feature_dict
# # uses same bond as bond_to_feature_vector test
# bond_feature_dict = bond_feature_vector_to_dict(bond_feature)
# assert bond_feature_dict['bond_type'] == 'DOUBLE'
# assert bond_feature_dict['bond_stereo'] == 'STEREOE'
# assert bond_feature_dict['is_conjugated'] == False


full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()

class AtomEncoder(nn.Layer):

    def __init__(self, emb_dim):
        super().__init__()
        
        self.atom_embedding_list = nn.LayerDict()
        self.feature_list = [
            "atomic_num",
            "chiral_tag",
            "degree",
            "formal_charge",
            "explicit_valence",
            "total_numHs",
            "num_radical_e",
            "hybridization",
            "is_aromatic",
            "atom_is_in_ring"
        ]

        for key, dim in full_atom_feature_dims.items():
            emb = nn.Embedding(dim, emb_dim)
            nn.initializer.XavierUniform(emb.weight)
            self.atom_embedding_list[key] = emb

    def forward(self, node_feat: dict):
        x_embedding = 0
        import logging
        #logging.debug(print(node_feat))
        for key in node_feat.keys():
            if key in self.feature_list:
                x_embedding += self.atom_embedding_list[key](node_feat[key])

        return x_embedding


class BondEncoder(nn.Layer):
    
    def __init__(self, emb_dim):
        super().__init__()
        
        self.bond_embedding_list = nn.LayerDict()
        self.feature_list = [
           'bond_dir', 'bond_type', 'is_in_ring',
        ]

        for key, dim in full_bond_feature_dims.items():
            emb = nn.Embedding(dim, emb_dim)
            nn.initializer.XavierNormal(emb.weight)
            self.bond_embedding_list[key] = emb

    def forward(self, edge_feat: dict):
        bond_embedding = 0
       
        for key in edge_feat.keys():
            if key in self.feature_list:
                bond_embedding += self.bond_embedding_list[key](edge_feat[key])

        return bond_embedding  
    
    
class CosineCutoff(nn.Layer):

    def __init__(self, cutoff):
        super().__init__()

        self.cutoff = cutoff

    def forward(self, distances):
        cutoffs = 0.5 * (paddle.cos(distances * math.pi / self.cutoff) + 1.0)
        cutoffs = cutoffs * (distances < self.cutoff).astype('float32')
        return cutoffs


class ExpNormalSmearing(nn.Layer):
    def __init__(self, cutoff=5.0, num_rbf=50, trainable=True):
        super().__init__()
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(cutoff)
        self.alpha = 5.0 / cutoff

        means, betas = self._initial_params()
        if trainable:
            self.add_parameter("means", self.create_parameter(
                shape=means.shape,
                default_initializer=nn.initializer.Assign(means)
            ))
            self.add_parameter("betas", self.create_parameter(
                shape=means.shape,
                default_initializer=nn.initializer.Assign(betas)
            ))
            # self.register_parameter("means", nn.Parameter(means))
            # self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        start_value = paddle.exp(paddle.to_tensor(-self.cutoff))
        means = paddle.linspace(start_value, 1, self.num_rbf)
        betas = paddle.to_tensor([(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf)
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        paddle.assign(means, self.means)
        paddle.assign(betas, self.betas)
        # self.means.data.copy_(means)
        # self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * paddle.exp(-self.betas * (paddle.exp(self.alpha * (-dist)) - self.means) ** 2)


class GaussianSmearing(nn.Layer):
    def __init__(self, cutoff=5.0, num_rbf=50, trainable=True):
        super().__init__()
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.trainable = trainable

        offset, coeff = self._initial_params()
        if trainable:
            self.add_parameter("coeff", self.create_parameter(
                shape=coeff.shape,
                default_initializer=nn.initializer.Assign(coeff)
            ))
            self.add_parameter("offset", self.create_parameter(
                shape=offset.shape,
                default_initializer=nn.initializer.Assign(offset)
            ))
            # self.register_parameter("coeff", nn.Parameter(coeff))
            # self.register_parameter("offset", nn.Parameter(offset))
        else:
            self.register_buffer("coeff", coeff)
            self.register_buffer("offset", offset)

    def _initial_params(self):
        offset = paddle.linspace(0, self.cutoff, self.num_rbf)
        coeff = -0.5 / (offset[1] - offset[0]) ** 2
        return offset, coeff

    def reset_parameters(self):
        offset, coeff = self._initial_params()
        paddle.assign(offset, self.offset)
        paddle.assign(coeff, self.coeff)
        # self.offset.data.copy_(offset)
        # self.coeff.data.copy_(coeff)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) - self.offset
        return paddle.exp(self.coeff * paddle.pow(dist, 2))
    
rbf_class_mapping = {"gauss": GaussianSmearing, "expnorm": ExpNormalSmearing}


class Sphere(nn.Layer):

    def __init__(self, l=2):
        super(Sphere, self).__init__()
        self.l = l

    def forward(self, edge_vec):
        edge_sh = self._spherical_harmonics(self.l, edge_vec[..., 0], edge_vec[..., 1], edge_vec[..., 2])
        return edge_sh

    @staticmethod
    def _spherical_harmonics(lmax: int, x, y, z):

        sh_1_0, sh_1_1, sh_1_2 = x, y, z

        if lmax == 1:
            return paddle.stack([sh_1_0, sh_1_1, sh_1_2], axis=-1)

        sh_2_0 = math.sqrt(3.0) * x * z
        sh_2_1 = math.sqrt(3.0) * x * y
        y2 = y.pow(2)
        x2z2 = x.pow(2) + z.pow(2)
        sh_2_2 = y2 - 0.5 * x2z2
        sh_2_3 = math.sqrt(3.0) * y * z
        sh_2_4 = math.sqrt(3.0) / 2.0 * (z.pow(2) - x.pow(2))

        if lmax == 2:
            return paddle.stack([sh_1_0, sh_1_1, sh_1_2, sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4], axis=-1)
        
        
class VecLayerNorm(nn.Layer):
    def __init__(self, hidden_channels, trainable, norm_type="max_min"):
        super(VecLayerNorm, self).__init__()

        self.hidden_channels = hidden_channels
        self.eps = 1e-12

        weight = paddle.ones(self.hidden_channels)
        if trainable:
            self.add_parameter("weight", self.create_parameter(
                shape=weight.shape,
                default_initializer=nn.initializer.Assign(weight)
            ))
            # self.register_parameter("weight", nn.Parameter(weight))
        else:
            self.register_buffer("weight", weight)

        if norm_type == "rms":
            self.norm = self.rms_norm
        elif norm_type == "max_min":
            self.norm = self.max_min_norm
        else:
            self.norm = self.none_norm

        self.reset_parameters()

    def reset_parameters(self):
        weight = paddle.ones(self.hidden_channels)
        paddle.assign(weight, self.weight)
        # self.weight.data.copy_(weight)

    def none_norm(self, vec):
        return vec

    def rms_norm(self, vec):
        # vec: (num_atoms, 3 or 5, hidden_channels)
        dist = paddle.norm(vec, axis=1)

        if (dist == 0).all():
            return paddle.zeros_like(vec)

        # dist = dist.clamp(min=self.eps)
        dist = paddle.clip(dist, min=self.eps)
        dist = paddle.sqrt(paddle.mean(dist ** 2, axis=-1))
        return vec / F.relu(dist).unsqueeze(-1).unsqueeze(-1)

    def max_min_norm(self, vec):
        # vec: (num_atoms, 3 or 5, hidden_channels)
        dist = paddle.norm(vec, axis=1, keepdim=True)

        if (dist == 0).all():
            return paddle.zeros_like(vec)

        # dist = dist.clamp(min=self.eps)
        dist = paddle.clip(dist, min=self.eps)
        direct = vec / dist

        max_val, _ = paddle.max(dist, axis=-1)
        min_val, _ = paddle.min(dist, axis=-1)
        delta = (max_val - min_val).view(-1)
        delta = paddle.where(delta == 0, paddle.ones_like(delta), delta)
        dist = (dist - min_val.view(-1, 1, 1)) / delta.view(-1, 1, 1)

        return F.relu(dist) * direct

    def forward(self, vec):
        # vec: (num_atoms, 3 or 8, hidden_channels)
        if vec.shape[1] == 3:
            vec = self.norm(vec)
            return vec * self.weight.unsqueeze(0).unsqueeze(0)
        elif vec.shape[1] == 8:
            vec1, vec2 = paddle.split(vec, [3, 5], axis=1)
            vec1 = self.norm(vec1)
            vec2 = self.norm(vec2)
            vec = paddle.concat([vec1, vec2], axis=1)
            return vec * self.weight.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError("VecLayerNorm only support 3 or 8 channels")