from utils.utils import DownstreamCollateFn, calc_parameter_size
from utils.inmemory_dataset import InMemoryDataset
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import random 
import warnings
import paddle
from paddle import optimizer
from sklearn import metrics
import paddle.nn as nn
import pandas as pd
from rdkit import Chem
from utils.to_graph import transfer_smiles_to_graph, transfer_mol_to_graph
import pgl
from visnet import visnet
from visnet import visnet_output_modules
import sys
import os
sys.path.insert(0, '/home/chenmingan/workplace/paddle/terpenoid-paddle/env')
import pahelix
warnings.filterwarnings('ignore')


class GEMData_mmff(object):
    def __init__(self, path, sdf_path, label_name=None):
        self.df = pd.read_csv(path)
        self.smiles = self.df['smiles'].values
        self.mols = Chem.SDMolSupplier(sdf_path)

        if label_name is not None:
            self.label = self.df[label_name].values
        else:
            self.label = np.zeros(len(self.df))

    
    def to_data_list(self, save_name, num_worker=4):
        smiles_to_graph_dict = transfer_mol_to_graph(self.mols, self.smiles, num_worker)
        data_list = []
        for i in range(len(self.smiles)):
            data_item = {}
            data_item['Smiles'] = self.smiles[i]
            data_item['Graph'] = smiles_to_graph_dict[self.smiles[i]]
            
            data_item['Label'] = self.label[i] 
            data_list.append(data_item)

        save_path = f'work/{save_name}.pkl'
        pickle.dump(data_list, open(save_path, 'wb'))

    def get_label_stat(self):
        return {
            'mean': np.mean(self.label),
            'std': np.std(self.label),
            'N': len(self.label)
        }

data = GEMData_mmff('data/train.csv', 'data/train.sdf', label_name='label')
# label_stat = data.get_label_stat()
# print(f'label_stat: {label_stat}')
# label_mean = label_stat['mean']
# label_std = label_stat['std']
data.to_data_list(save_name='train_semi_from_mol_exclude2D', num_worker=40)