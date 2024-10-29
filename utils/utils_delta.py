import numpy as np
import pgl

def calc_parameter_size(parameter_list):
    """Calculate the total size of `parameter_list`"""
    count = 0
    for param in parameter_list:
        count += np.prod(param.shape)
    return count
class DownstreamCollateFn(object):
    def __init__(self, task_type='regr', is_inference=True):
        atom_names = ["atomic_num", "formal_charge", "degree", "chiral_tag", "total_numHs", "is_aromatic",
                      "hybridization","atom_pos"]
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
        u0_list = []
        u_list = []
        for data in data_list:
            compound_class_list.append(data['Label'])
            u0_list.append(data['u0'])
            u_list.append(data['u'])
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

        return atom_bond_graph, bond_angle_graph, np.array(compound_class_list, dtype=np.float32), np.array(u0_list, dtype=np.float32), np.array(u_list, dtype=np.float32)



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

