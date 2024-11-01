import numpy as np
import pgl

def calc_parameter_size(parameter_list):
    """Calculate the total size of `parameter_list`"""
    count = 0
    for param in parameter_list:
        count += np.prod(param.shape)
    return count

import hashlib
from copy import deepcopy
def md5_hash(string):
    """tbd"""
    md5 = hashlib.md5(string.encode('utf-8')).hexdigest()
    return int(md5, 16)

def mask_context_of_geognn_graph(
        g, 
        superedge_g,
        target_atom_indices=None, 
        mask_ratio=None, 
        mask_value=0, 
        subgraph_num=None,
        version='gem'):
    """tbd"""
    def get_subgraph_str(g, atom_index, nei_atom_indices, nei_bond_indices):
        """tbd"""
        atomic_num = g.node_feat['atomic_num'].flatten()
        bond_type = g.edge_feat['bond_type'].flatten()
        subgraph_str = 'A' + str(atomic_num[atom_index])
        subgraph_str += 'N' + ':'.join([str(x) for x in np.sort(atomic_num[nei_atom_indices])])
        subgraph_str += 'E' + ':'.join([str(x) for x in np.sort(bond_type[nei_bond_indices])])
        return subgraph_str

    g = deepcopy(g)
    N = g.num_nodes
    E = g.num_edges
    full_atom_indices = np.arange(N)
    full_bond_indices = np.arange(E)

    if target_atom_indices is None:
        masked_size = max(1, int(N * mask_ratio))   # at least 1 atom will be selected.
        target_atom_indices = np.random.choice(full_atom_indices, size=masked_size, replace=False)
    target_labels = []
    Cm_node_i = []
    masked_bond_indices = []
    for atom_index in target_atom_indices:
        left_nei_bond_indices = full_bond_indices[g.edges[:, 0] == atom_index]
        right_nei_bond_indices = full_bond_indices[g.edges[:, 1] == atom_index]
        nei_bond_indices = np.append(left_nei_bond_indices, right_nei_bond_indices)
        left_nei_atom_indices = g.edges[left_nei_bond_indices, 1]
        right_nei_atom_indices = g.edges[right_nei_bond_indices, 0]
        nei_atom_indices = np.append(left_nei_atom_indices, right_nei_atom_indices)

        if version == 'gem':
            subgraph_str = get_subgraph_str(g, atom_index, nei_atom_indices, nei_bond_indices)
            subgraph_id = md5_hash(subgraph_str) % subgraph_num
            target_label = subgraph_id
        else:
            raise ValueError(version)
        
        target_labels.append(target_label)
        Cm_node_i.append([atom_index])
        Cm_node_i.append(nei_atom_indices)
        masked_bond_indices.append(nei_bond_indices)
    
    target_atom_indices = np.array(target_atom_indices)
    target_labels = np.array(target_labels)
    Cm_node_i = np.concatenate(Cm_node_i, 0)
    masked_bond_indices = np.concatenate(masked_bond_indices, 0)
    for name in g.node_feat:
        g.node_feat[name][Cm_node_i] = mask_value
    for name in g.edge_feat:
        g.edge_feat[name][masked_bond_indices] = mask_value

    # mask superedge_g
    full_superedge_indices = np.arange(superedge_g.num_edges)
    masked_superedge_indices = []
    for bond_index in masked_bond_indices:
        left_indices = full_superedge_indices[superedge_g.edges[:, 0] == bond_index]
        right_indices = full_superedge_indices[superedge_g.edges[:, 1] == bond_index]
        masked_superedge_indices.append(np.append(left_indices, right_indices))
    masked_superedge_indices = np.concatenate(masked_superedge_indices, 0)
    for name in superedge_g.edge_feat:
        superedge_g.edge_feat[name][masked_superedge_indices] = mask_value
    return [g, superedge_g, target_atom_indices, target_labels]

class DownstreamCollateFn(object):
    def __init__(self, task_type='regr', is_inference=True):
        atom_names = ["atomic_num", "formal_charge", "degree", "chiral_tag", "total_numHs", "is_aromatic",
                      "hybridization","atom_pos","explicit_valence"]
        bond_names = ["bond_dir", "bond_type", "is_in_ring"]
        bond_float_names = ["bond_length"]
        bond_angle_float_names = ["bond_angle"]

        self.atom_names = atom_names
        self.bond_names = bond_names
        self.bond_float_names = bond_float_names
        self.bond_angle_float_names = bond_angle_float_names
        self.task_type = task_type
        self.is_inference = is_inference

    def _flat_shapes(self, d):
        for name in d:
            d[name] = d[name].reshape([-1])

    def __call__(self, data_list):
        atom_bond_graph_list = []
        bond_angle_graph_list = []
        compound_class_list = []
        is_extrapolated_list = []
        for data in data_list:
            compound_class_list.append(data['Label'])
            data = data['Graph']
            ab_g = pgl.Graph(
                num_nodes=len(data[self.atom_names[0]]),
                edges=data['edges'],
                node_feat={name: data[name].reshape([-1, 1]) for name in self.atom_names},
                edge_feat={name: data[name].reshape([-1, 1]) for name in self.bond_names + self.bond_float_names})
            ba_g = pgl.Graph(
                num_nodes=len(data['edges']),
                edges=data['BondAngleGraph_edges'],
                node_feat={},
                edge_feat={name: data[name].reshape([-1, 1]) for name in self.bond_angle_float_names})
            atom_bond_graph_list.append(ab_g)
            bond_angle_graph_list.append(ba_g)

        atom_bond_graph = pgl.Graph.batch(atom_bond_graph_list)
        bond_angle_graph = pgl.Graph.batch(bond_angle_graph_list)

        self._flat_shapes(atom_bond_graph.node_feat)
        self._flat_shapes(atom_bond_graph.edge_feat)
        self._flat_shapes(bond_angle_graph.node_feat)
        self._flat_shapes(bond_angle_graph.edge_feat)

        return atom_bond_graph, bond_angle_graph, np.array(compound_class_list, dtype=np.float32)




class ContrastiveLearningCollateFn(object):
    def __init__(self, task_type='regr', is_inference=True):
        atom_names = ["atomic_num", "formal_charge", "degree", "chiral_tag", "total_numHs", "is_aromatic",
                      "hybridization"]
        bond_names = ["bond_dir", "bond_type", "is_in_ring"]
        bond_float_names = ["bond_length"]
        bond_angle_float_names = ["bond_angle"]

        self.atom_names = atom_names
        self.bond_names = bond_names
        self.bond_float_names = bond_float_names
        self.bond_angle_float_names = bond_angle_float_names
        self.task_type = task_type
        self.is_inference = is_inference

    def _flat_shapes(self, d):
        for name in d:
            d[name] = d[name].reshape([-1])

    def data_to_gs(self, data_item):
        data = data_item['Graph']
        ab_g = pgl.Graph(
            num_nodes=len(data[self.atom_names[0]]),
            edges=data['edges'],
            node_feat={name: data[name].reshape([-1, 1]) for name in self.atom_names},
            edge_feat={name: data[name].reshape([-1, 1]) for name in self.bond_names + self.bond_float_names})
        ba_g = pgl.Graph(
            num_nodes=len(data['edges']),
            edges=data['BondAngleGraph_edges'],
            node_feat={},
            edge_feat={name: data[name].reshape([-1, 1]) for name in self.bond_angle_float_names})
        return ab_g, ba_g

    def __call__(self, data_list):
        anchor_atom_bond_graph_list = []
        anchor_bond_angle_graph_list = []
        positive_atom_bond_graph_list = []
        positive_bond_angle_graph_list = []
        negative_atom_bond_graph_list = []
        negative_bond_angle_graph_list = []
        compound_class_list = []
        for data in data_list:
            anchor = data[0]
            positive = data[1]
            negative = data[2]
            compound_class_list.append(anchor['Label'])

            anchor_ab_g, anchor_ba_g = self.data_to_gs(anchor)
            positive_ab_g, positive_ba_g = self.data_to_gs(positive)
            negative_ab_g, negative_ba_g = self.data_to_gs(negative)

            anchor_atom_bond_graph_list.append(anchor_ab_g)
            anchor_bond_angle_graph_list.append(anchor_ba_g)
            positive_atom_bond_graph_list.append(positive_ab_g)
            positive_bond_angle_graph_list.append(positive_ba_g)
            negative_atom_bond_graph_list.append(negative_ab_g)
            negative_bond_angle_graph_list.append(negative_ba_g)

        anchor_atom_bond_graph = pgl.Graph.batch(anchor_atom_bond_graph_list)
        anchor_bond_angle_graph = pgl.Graph.batch(anchor_bond_angle_graph_list)

        positive_atom_bond_graph = pgl.Graph.batch(positive_atom_bond_graph_list)
        positive_bond_angle_graph = pgl.Graph.batch(positive_bond_angle_graph_list)

        negative_atom_bond_graph = pgl.Graph.batch(negative_atom_bond_graph_list)
        negative_bond_angle_graph = pgl.Graph.batch(negative_bond_angle_graph_list)

        self._flat_shapes(anchor_atom_bond_graph.node_feat)
        self._flat_shapes(anchor_atom_bond_graph.edge_feat)
        self._flat_shapes(anchor_bond_angle_graph.node_feat)
        self._flat_shapes(anchor_bond_angle_graph.edge_feat)

        self._flat_shapes(positive_atom_bond_graph.node_feat)
        self._flat_shapes(positive_atom_bond_graph.edge_feat)
        self._flat_shapes(positive_bond_angle_graph.node_feat)
        self._flat_shapes(positive_bond_angle_graph.edge_feat)

        self._flat_shapes(negative_atom_bond_graph.node_feat)
        self._flat_shapes(negative_atom_bond_graph.edge_feat)
        self._flat_shapes(negative_bond_angle_graph.node_feat)
        self._flat_shapes(negative_bond_angle_graph.edge_feat)

        return anchor_atom_bond_graph, anchor_bond_angle_graph, \
            positive_atom_bond_graph, positive_bond_angle_graph,\
            negative_atom_bond_graph, negative_bond_angle_graph,\
            np.array(compound_class_list, dtype=np.float32)

