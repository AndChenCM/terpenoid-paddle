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
from rdkit import rdBase
rdBase.DisableLog('rdApp.warning')
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
data.to_data_list(save_name='train_semi_from_mol_exhaust', num_worker=40)

def get_data_loader(mode, batch_size=256):
    collate_fn = DownstreamCollateFn()
    if mode == 'train':
        data_list = pickle.load(open("work/train_semi_from_mol_exhaust.pkl", 'rb'))
        train, valid = train_test_split(data_list, random_state=42, test_size=0.1)
        train, valid = InMemoryDataset(train), InMemoryDataset(valid)

        print(f'len train is {len(train)}, len valid is {len(valid)}')

        train_dl = train.get_data_loader(batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        valid_dl = valid.get_data_loader(batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        #collate_fn 把数据转换成atom_bond_graph, bond_angle_graph, np.array(compound_class_list, dtype=np.float32)

        return train_dl, valid_dl
    elif mode == 'test':
        # data_list = pickle.load(open("work/test_rdkit_3Dplus_2D_wH.pkl", 'rb'))

        # print(f'len test is {len(data_list)}')

        # test = InMemoryDataset(data_list)
        # test_dl = test.get_data_loader(batch_size=batch_size, shuffle=False)
        # return test_dl
        return ValueError("This is a training script")
    
def evaluate(model, dataloader):
    """评估模型"""
    model.eval()

    y_pred = np.array([])
    y_true = np.array([])
    for (atom_bond_graph, bond_angle_graph, label_true) in dataloader:
        output = model(atom_bond_graph.tensor())
        y_pred = np.concatenate((y_pred, output[:, 0].cpu().numpy()))
        y_true = np.concatenate((y_true, label_true))
    r2 = metrics.r2_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))

    metric_eval = {
        'r2': round(r2, 4),
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4)
    }

    return metric_eval

def trial(model_version, batch_size, lr,  tmax, weight_decay, max_bearable_epoch, max_epoch):

    train_data_loader, valid_data_loader = get_data_loader(mode='train', batch_size=batch_size)   
    
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
        max_num_neighbors=32)
    
    output_model = visnet_output_modules.EquivariantScalar(
       hidden_channels=80,out_channels=1
    )

    model = visnet.ViSNet(
        representation_model,
        output_model,
        reduce_op="sum",
        mean=None,
        std=None,
    )

    print("parameter size:", calc_parameter_size(model.parameters()), flush=True)

    # model_path = 'pretrain_weight/visnet_hs80_l6_rbf32_lm2_pt_on_train_mol_exhaustvisnet_hs80_l6_rbf32_lm2_pt_on_train_mol_exhaust1.pkl'
    # model.set_state_dict(paddle.load(model_path))
    # print('Load state_dict from %s' % model_path, flush=True)

    lr = optimizer.lr.CosineAnnealingDecay(learning_rate=lr, T_max=tmax)
    opt = optimizer.Adam(lr, parameters=model.parameters(), weight_decay=weight_decay)
   
    best_score = 1e9
    best_epoch = 0
    best_train_metric = {}
    best_valid_metric = {}
    criterion = nn.MSELoss()
    for epoch in range(max_epoch):
        model.train()
        for batch_idx,(atom_bond_graph, bond_angle_graph, label_true) in enumerate(train_data_loader):
           
            output = model(atom_bond_graph.tensor())
            label_true = paddle.to_tensor(label_true, dtype=paddle.float32).unsqueeze(axis=1)
            loss = criterion(output, label_true)
            print(f"Epoch [{epoch + 1}/{max_epoch}], Batch [{batch_idx}], Loss: {loss.numpy()}")
            loss.backward() 
            opt.step()
            opt.clear_grad()
            
        lr.step() 

        # 评估模型在训练集、验证集的表现
        train_metric = evaluate(model, train_data_loader)
        valid_metric = evaluate(model, valid_data_loader)

        score = valid_metric['RMSE']

        if score < best_score:
            # 保存score最大时的模型权重
            paddle.save(model.state_dict(), "weight/"+model_version+".pkl")
            best_score = score
            best_epoch = epoch
            best_train_metric = train_metric
            best_valid_metric = valid_metric

        print('epoch', epoch, flush = True)
        print('train', train_metric, flush = True)
        print('valid', valid_metric, flush = True)
        print(f'current_best_score: {best_score:.4f}, best_epoch: {best_epoch}', flush = True)
        print('=================================================')

        if epoch > best_epoch + max_bearable_epoch or epoch == max_epoch - 1:
            print(f"model_{model_version} is Done!!", flush = True)
            print('train', flush = True)
            print(best_train_metric)
            print('valid', flush = True)
            print(best_valid_metric)
            print(f'current_best_score: {best_score:.4f}, best_epoch: {best_epoch}', flush = True)
            break




# 开始训练
# 固定随机种子
SEED = 42
paddle.seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

batch_size = 32           
lr = 1e-4
tmax = 15
weight_decay = 1e-5
max_bearable_epoch = 50
max_epoch = 1000

trial('visnet_hs80_l6_rbf32_lm2_bs32_lr1e-4_mol_exhaust_scratch', batch_size, lr, tmax, weight_decay, max_bearable_epoch, max_epoch)



