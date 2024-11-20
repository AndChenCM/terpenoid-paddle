from utils.utils import DownstreamCollateFn
from utils.inmemory_dataset import InMemoryDataset
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
from rdkit import rdBase
rdBase.DisableLog('rdApp.warning')
from rdkit import Chem

import warnings
import paddle
import paddle.nn as nn
import pandas as pd
from utils.to_graph import transfer_mol_to_graph
warnings.filterwarnings('ignore')

from visnet import visnet
from visnet import visnet_output_modules
import random


class GEMData_mmff(object):
    def __init__(self, path, label_name=None):
        self.df = pd.read_csv(path)
        self.smiles = self.df['smiles'].values
        self.mols = Chem.SDMolSupplier(test_sdf_path)

        if label_name is not None:
            self.label = self.df[label_name].values
        else:
            self.label = np.zeros(len(self.df))
    
    def to_data_list(self, num_worker=4):

        smiles_to_graph_dict = transfer_mol_to_graph(self.mols, self.smiles, num_worker)

        data_list = []
        for i in range(len(self.smiles)):
            data_item = {}
            data_item['Smiles'] = self.smiles[i]
            data_item['Graph'] = smiles_to_graph_dict[self.smiles[i]]
            data_item['Label'] = self.label[i]
            data_list.append(data_item)
        return data_list

def get_data_loader(mode, batch_size=256):
    collate_fn = DownstreamCollateFn()
    if mode == 'test':
        data_list = GEMData_mmff(test_csv_path).to_data_list(num_worker=4)

        print(f'len test is {len(data_list)}')

        test = InMemoryDataset(data_list)
        test_dl = test.get_data_loader(batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        return test_dl

def predict(seed=42):
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    test_dl = get_data_loader(mode='test', batch_size=128)
    representation_model = visnet.ViSNetBlock(lmax=2,
        vecnorm_type='none',
        trainable_vecnorm=False,
        num_heads=8,
        num_layers=6,
        hidden_channels=80,
        num_rbf=32,
        rbf_type="expnorm",
        trainable_rbf=False,
        activation="silu",
        attn_activation="silu",
        cutoff=5.0,
        max_num_neighbors=32,)

    output_model =  visnet_output_modules.EquivariantScalar(
        hidden_channels=80, out_channels=1
    )

    model = visnet.ViSNet(
        representation_model,
        output_model,
        reduce_op="sum",
        mean=None,
        std=None,
    )
    ckpt = paddle.load("/home/chenmingan/workplace/paddle/terpenoid-paddle/weight/visnet_hs80_l6_rbf32_lm2_bs32_lr1e-4_IDopt_smL1loss_rop_yood5%.pkl")
    new_state_dict = {}
    for key, value in ckpt.items():
        new_key = key.replace('daylight_fg_counts', 'daylight_fg')
        new_state_dict[new_key] = value
    
    model.set_state_dict(ckpt)

    model.eval()
    y_pred = np.array([])
    for (atom_bond_graph, bond_angle_graph, _) in test_dl:
        output = model(atom_bond_graph.tensor())
        y_pred = np.concatenate((y_pred, output[:, 0].cpu().numpy()))

    test_df = pd.read_csv(test_csv_path)
    test_df['pred'] = y_pred
    test_df.to_csv(result_csv_path, index=False)


if __name__ == "__main__":
    import os
    import sys
    test_csv_path = sys.argv[1]
    test_sdf_path = sys.argv[2]
    result_csv_path = sys.argv[3]

    predict()
    