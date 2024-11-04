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
            import logging
            #logging.debug(print(md5_hash(subgraph_str)% subgraph_num))
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
    def __init__(self, task_type='regr', is_inference=True,
           ):
        atom_names = ["atomic_num", "formal_charge", "degree", "chiral_tag", "explicit_valence", "total_numHs", "is_aromatic",
                      "hybridization","atom_pos"]
        bond_names = ["bond_dir", "bond_type", "is_in_ring"]
        bond_float_names = ["bond_length"]
        bond_angle_float_names = ["bond_angle"]
        pretrain_tasks=['Bar']
        mask_ratio = 0.1
        Cm_vocab = 1000
        self.atom_names = atom_names
        self.bond_names = bond_names
        self.bond_float_names = bond_float_names
        self.bond_angle_float_names = bond_angle_float_names
        self.task_type = task_type
        self.is_inference = is_inference
        self.pretrain_tasks = pretrain_tasks
        self.mask_ratio = mask_ratio
        self.Cm_vocab = Cm_vocab

    def _flat_shapes(self, d):
        for name in d:
            d[name] = d[name].reshape([-1])


    def get_pretrain_bond_angle(self, edges, atom_poses):
        """tbd"""
        def _get_angle(vec1, vec2):
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0
            vec1 = vec1 / (norm1 + 1e-5)    # 1e-5: prevent numerical errors
            vec2 = vec2 / (norm2 + 1e-5)
            angle = np.arccos(np.dot(vec1, vec2))
            return angle
        def _add_item(
                node_i_indices, node_j_indices, node_k_indices, bond_angles, 
                node_i_index, node_j_index, node_k_index):
            node_i_indices += [node_i_index, node_k_index]
            node_j_indices += [node_j_index, node_j_index]
            node_k_indices += [node_k_index, node_i_index]
            pos_i = atom_poses[node_i_index]
            pos_j = atom_poses[node_j_index]
            pos_k = atom_poses[node_k_index]
            angle = _get_angle(pos_i - pos_j, pos_k - pos_j)
            bond_angles += [angle, angle]

        E = len(edges)
        node_i_indices = []
        node_j_indices = []
        node_k_indices = []
        bond_angles = []
        for edge_i in range(E - 1):
            for edge_j in range(edge_i + 1, E):
                a0, a1 = edges[edge_i]
                b0, b1 = edges[edge_j]
                if a0 == b0 and a1 == b1:
                    continue
                if a0 == b1 and a1 == b0:
                    continue
                if a0 == b0:
                    _add_item(
                            node_i_indices, node_j_indices, node_k_indices, bond_angles,
                            a1, a0, b1)
                if a0 == b1:
                    _add_item(
                            node_i_indices, node_j_indices, node_k_indices, bond_angles,
                            a1, a0, b0)
                if a1 == b0:
                    _add_item(
                            node_i_indices, node_j_indices, node_k_indices, bond_angles,
                            a0, a1, b1)
                if a1 == b1:
                    _add_item(
                            node_i_indices, node_j_indices, node_k_indices, bond_angles,
                            a0, a1, b0)
        node_ijk = np.array([node_i_indices, node_j_indices, node_k_indices])
        uniq_node_ijk, uniq_index = np.unique(node_ijk, return_index=True, axis=1)
        node_i_indices, node_j_indices, node_k_indices = uniq_node_ijk
        bond_angles = np.array(bond_angles)[uniq_index]
        return [node_i_indices, node_j_indices, node_k_indices, bond_angles]

    def __call__(self, data_list):
        atom_bond_graph_list = []
        bond_angle_graph_list = []
        compound_class_list = []
        masked_atom_bond_graph_list = []
        masked_bond_angle_graph_list = []
        Cm_context_id = []
        Cm_node_i = []
        u0_list = []
        u_list = []
        Bl_node_i = []
        Bl_node_j = []
        Bl_bond_length = []
        Ba_node_i = []
        Ba_node_j = []
        Ba_node_k = []
        Ba_bond_angle = []
        atomic_charge = []
        node_count = 0
        for data in data_list:
            #compound_class_list.append(data['Label'])
            #u0_list.append(data['u0'])
            #u_list.append(data['u'])
            import logging
           
            data = data['Graph']


            #data['Bl_node_i'] = np.array(data['edges'][:, 0])
            #data['Bl_node_j'] = np.array(data['edges'][:, 1])
            #data['Bl_bond_length'] = np.array(data['bond_length'])

            node_i, node_j, node_k, bond_angles = \
                self.get_pretrain_bond_angle(data['edges'], data['atom_pos'])
            data['Ba_node_i'] = node_i
            data['Ba_node_j'] = node_j
            data['Ba_node_k'] = node_k
            data['Ba_bond_angle'] = bond_angles
            

            N = len(data[self.atom_names[0]])
            E = len(data['edges'])
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
            
            masked_ab_g, masked_ba_g, mask_node_i, context_id = mask_context_of_geognn_graph(
                    ab_g, ba_g, mask_ratio=self.mask_ratio, subgraph_num=self.Cm_vocab)
            
            
            atom_bond_graph_list.append(ab_g)
            bond_angle_graph_list.append(ba_g)
            masked_atom_bond_graph_list.append(masked_ab_g)
            masked_bond_angle_graph_list.append(masked_ba_g)
            #Cm_node_i.append(mask_node_i + node_count)
            #Cm_context_id.append(context_id)

            #Bl_node_i.append(data['Bl_node_i'] + node_count)
            #Bl_node_j.append(data['Bl_node_j'] + node_count)
            #Bl_bond_length.append(data['Bl_bond_length'])

            Ba_node_i.append(data['Ba_node_i'] + node_count)
            Ba_node_j.append(data['Ba_node_j'] + node_count)
            Ba_node_k.append(data['Ba_node_k'] + node_count)
            Ba_bond_angle.append(data['Ba_bond_angle'])

            #atomic_charge.append(data['atomic_charge'])

            node_count += N
        graph_dict = {}    
        feed_dict = {}
        atom_bond_graph = pgl.Graph.batch(atom_bond_graph_list)
        bond_angle_graph = pgl.Graph.batch(bond_angle_graph_list)
        masked_atom_bond_graph = pgl.Graph.batch(masked_atom_bond_graph_list)
        masked_bond_angle_graph = pgl.Graph.batch(masked_bond_angle_graph_list)
        self._flat_shapes(atom_bond_graph.node_feat)
        self._flat_shapes(atom_bond_graph.edge_feat)
        self._flat_shapes(bond_angle_graph.node_feat)
        self._flat_shapes(bond_angle_graph.edge_feat)
        self._flat_shapes(masked_atom_bond_graph.node_feat)
        self._flat_shapes(masked_atom_bond_graph.edge_feat)
        self._flat_shapes(masked_bond_angle_graph.node_feat)
        self._flat_shapes(masked_bond_angle_graph.edge_feat) 

        #graph_dict['atom_bond_graph'] = atom_bond_graph
        #graph_dict['bond_angle_graph'] = bond_angle_graph
        #graph_dict['masked_atom_bond_graph'] = masked_atom_bond_graph
        #graph_dict['masked_bond_angle_graph'] = masked_bond_angle_graph
    
        #feed_dict['Cm_node_i'] = np.concatenate(Cm_node_i, 0).reshape(-1).astype('int64')
        #feed_dict['Cm_context_id'] = np.concatenate(Cm_context_id, 0).reshape(-1, 1).astype('int64')

        #feed_dict['Bl_node_i'] = np.concatenate(Bl_node_i, 0).reshape(-1).astype('int64')
        #feed_dict['Bl_node_j'] = np.concatenate(Bl_node_j, 0).reshape(-1).astype('int64')
        #feed_dict['Bl_bond_length'] = np.concatenate(Bl_bond_length, 0).reshape(-1, 1).astype('float32')

        feed_dict['Ba_node_i'] = np.concatenate(Ba_node_i, 0).reshape(-1).astype('int64')
        feed_dict['Ba_node_j'] = np.concatenate(Ba_node_j, 0).reshape(-1).astype('int64')
        feed_dict['Ba_node_k'] = np.concatenate(Ba_node_k, 0).reshape(-1).astype('int64')
        feed_dict['Ba_bond_angle'] = np.concatenate(Ba_bond_angle, 0).reshape(-1, 1).astype('float32')

        #feed_dict['atomic_charge'] = np.concatenate(atomic_charge, 0).reshape(-1).astype('float32')
        
        #return atom_bond_graph, bond_angle_graph, np.array(compound_class_list, dtype=np.float32),# np.array(u0_list, dtype=np.float32), np.array(u_list, dtype=np.float32)
        return atom_bond_graph, masked_atom_bond_graph, feed_dict



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

