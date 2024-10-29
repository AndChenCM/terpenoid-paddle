from utils.utils import DownstreamCollateFn
from utils.inmemory_dataset import InMemoryDataset
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import random 
import warnings
import json
import paddle
from paddle import optimizer
from sklearn import metrics
from utils.gem_model import GeoGNNModel
import paddle.nn as nn
import pandas as pd
from utils.to_graph import transfer_smiles_to_graph
warnings.filterwarnings('ignore')

import visnet

class GEMData_mmff(object):
    def __init__(self, path, label_name=None):
        self.df = pd.read_csv(path)
        self.smiles = self.df['smiles'].values

        if label_name is not None:
            self.label = self.df[label_name].values
        else:
            self.label = np.zeros(len(self.df))

    def get_label_stat(self):
        return {
            'mean': np.mean(self.label),
            'std': np.std(self.label),
            'N': len(self.label)
        }
    
    def to_data_list(self, save_name, num_worker=20):
        # 将smiles转化为graph， num_worker为线程数
        smiles_to_graph_dict = transfer_smiles_to_graph(self.smiles, num_worker)
        # 这获取了MMFF下的最低能量构象，用rdkit采样了十次，取能量最低的那个
        data_list = []
        for i in range(len(self.smiles)):
            data_item = {}
            data_item['Smiles'] = self.smiles[i]
            data_item['Graph'] = smiles_to_graph_dict[self.smiles[i]]
            data_item['Label'] = self.label[i]
            data_list.append(data_item)

        save_path = f'work/{save_name}.pkl'
        pickle.dump(data_list, open(save_path, 'wb'))

#data = GEMData_mmff('data/data285818/test.csv', label_name=None)
#label_stat = data.get_label_stat()
#print(f'label_stat: {label_stat}')
#label_mean = label_stat['mean']
#label_std = label_stat['std']

#data.to_data_list(save_name='test_2D_', num_worker=20)

def get_data_loader(mode, batch_size=256):
    collate_fn = DownstreamCollateFn()
    if mode == 'train':
        data_list = pickle.load(open("work/train.pkl", 'rb'))

        train, valid = train_test_split(data_list, random_state=42, test_size=0.1)
        train, valid = InMemoryDataset(train), InMemoryDataset(valid)

        print(f'len train is {len(train)}, len valid is {len(valid)}')

        train_dl = train.get_data_loader(batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        valid_dl = valid.get_data_loader(batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        return train_dl, valid_dl
    elif mode == 'test':
        data_list = pickle.load(open("work/test_2D_.pkl", 'rb'))

        print(f'len test is {len(data_list)}')

        test = InMemoryDataset(data_list)
        test_dl = test.get_data_loader(batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        return test_dl

class GetLoss(nn.Layer):
    def __init__(self, encoder, label_mean=0.0, label_std=1.0):
        super(GetLoss, self).__init__()
        self.encoder = encoder
        self.label_mean = paddle.to_tensor(label_mean)
        self.label_std = paddle.to_tensor(label_std)
    def _get_scaled_label(self, x):
        return (x - self.label_mean) / (self.label_std + 1e-5)

    def _get_unscaled_pred(self, x):
        return x * (self.label_std + 1e-5) + self.label_mean
    
    def forward(self, atom_bond_graph, bond_angle_graph):

        x = self.encoder(atom_bond_graph.tensor())
        #pred = self._get_unscaled_pred(x)
        return x

test_dl = get_data_loader(mode='test', batch_size=256)

representation_model = visnet.ViSNetBlock(lmax=2,
        vecnorm_type='none',
        trainable_vecnorm=False,
        num_heads=8,
        num_layers=6,
        hidden_channels=128,
        num_rbf=32,
        rbf_type="expnorm",
        trainable_rbf=False,
        activation="silu",
        attn_activation="silu",
        cutoff=5.0,
        max_num_neighbors=32,)
   
   
import visnet_output_modules
output_model =  visnet_output_modules.EquivariantScalar(
    hidden_channels=128, out_channels=1
)

visnet_model = visnet.ViSNet(
    representation_model,
    output_model,
    reduce_op="sum",
    mean=None,
    std=None,
)
#compound_encoder_config = json.load(open('model_configs/geognn_l8.json', 'r'))
#encoder = GeoGNNModel(compound_encoder_config)
#encoder.set_state_dict(paddle.load(f"weight/regr.pdparams"))
model = GetLoss(encoder=visnet_model, label_mean=0, label_std=1)
model.set_state_dict(paddle.load('/home/chenmingan/projects/paddle/prop_regr_jiangxinyu/weight/model_visnet_pre_zinc_50w_pre_train6.pkl'))

model.eval()
y_pred = np.array([])
for (atom_bond_graph, bond_angle_graph, compound_class) in test_dl:
    output = model(atom_bond_graph, bond_angle_graph)
    y_pred = np.concatenate((y_pred, output[:, 0].cpu().numpy()))

test_df = pd.read_csv('data/data285818/test.csv')
test_df['pred'] = y_pred
test_df.to_csv('test_model_visnet_pre_zinc_50w_pre_train6.csv', index=False)

    