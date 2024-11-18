from utils.utils import DownstreamCollateFn, calc_parameter_size
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
from utils.to_graph import transfer_smiles_to_graph, transfer_mol_to_graph
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

# data = GEMData_mmff('data/cleaned_train_success_opt.csv', 'data/cleaned_train_success_opt.sdf', label_name='label')
# label_stat = data.get_label_stat()
# print(f'label_stat: {label_stat}')
# label_mean = label_stat['mean']
# label_std = label_stat['std']
# data.to_data_list(save_name='train_semi_success_opt', num_worker=40)

# def get_data_loader(mode, batch_size=256):
#     collate_fn = DownstreamCollateFn()
#     if mode == 'train':
#         data_list = pickle.load(open("work/train_semi_success_opt.pkl", 'rb'))
#         train, valid = train_test_split(data_list, random_state=42, test_size=0.1)
#         train, valid = InMemoryDataset(train), InMemoryDataset(valid)

#         print(f'len train is {len(train)}, len valid is {len(valid)}')

#         train_dl = train.get_data_loader(batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
#         valid_dl = valid.get_data_loader(batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
#         #collate_fn 把数据转换成atom_bond_graph, bond_angle_graph, np.array(compound_class_list, dtype=np.float32)

#         return train_dl, valid_dl
#     elif mode == 'test':
#         # data_list = pickle.load(open("work/test_rdkit_3Dplus_2D_wH.pkl", 'rb'))

#         # print(f'len test is {len(data_list)}')

#         # test = InMemoryDataset(data_list)
#         # test_dl = test.get_data_loader(batch_size=batch_size, shuffle=False)
#         # return test_dl
#         return ValueError("This is a training script")

def get_data_loader(mode, batch_size=256, split_seed=42, use_fg=False):
    collate_fn = DownstreamCollateFn(use_fg=use_fg)
    
    if mode == 'train':
        # data_list0 = pickle.load(open("work/train_semi_final_from_mol_clean_fg.pkl", 'rb'))
        # data_list1 = pickle.load(open("work/train_semi_from_smiles_2_fg_clean.pkl", 'rb'))
        # data_list = data_list0 + data_list1
        data_list = pickle.load(open("/home/chenmingan/workplace/paddle/terpenoid-paddle/data/train.pkl", 'rb'))
        # data_list = pickle.load(open("/home/chenmingan/workplace/paddle/terpenoid-paddle/data/train.pkl", 'rb'))
        train, valid = train_test_split(data_list, random_state=42, test_size=0.1)
        # from collections import defaultdict
        # grouped_data = defaultdict(list)
        # for item in data_list:
        #     grouped_data[item['Smiles']].append(item)

        # # 分组抽样
        # train_keys, valid_keys = train_test_split(list(grouped_data.keys()), random_state=split_seed, test_size=0.1)

        # # 从分组中提取训练集和验证集
        # train = [item for key in train_keys for item in grouped_data[key]]
        # valid = [item for key in valid_keys for item in grouped_data[key]]
        
        print(f'len train is {len(train)}, len valid is {len(valid)}', flush=True)
        train, valid = InMemoryDataset(train), InMemoryDataset(valid)
        train_dl = train.get_data_loader(batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        valid_dl = valid.get_data_loader(batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        return train_dl, valid_dl



class DownStreamModel(nn.Layer):
    def __init__(self, encoder):
        super(DownStreamModel, self).__init__()
        self.encoder = encoder
        self.mlp = nn.Sequential(
            nn.Linear(32, 32, weight_attr=nn.initializer.KaimingNormal()),
            nn.LeakyReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, atom_bond_graph, bond_angle_graph):
        node_repr, edge_repr, graph_repr = self.encoder(atom_bond_graph.tensor(), bond_angle_graph.tensor())
        x = self.mlp(graph_repr)
        return x

def trial(model_version, batch_size, lr, tmax, weight_decay, max_bearable_epoch, max_epoch, seed, dropout_rate):
    train_data_loader, valid_data_loader = get_data_loader(mode='train', batch_size=batch_size, random_state=seed)   

    compound_encoder_config = json.load(open('configs/geognn_l8.json', 'r'))
    compound_encoder_config['dropout_rate'] = dropout_rate
    encoder = GeoGNNModel(compound_encoder_config)
    print("encoder parameter size:", calc_parameter_size(encoder.parameters()), flush=True)
    encoder.set_state_dict(paddle.load(f"weight/regr.pdparams"))
    # encoder.set_state_dict(paddle.load(f"/home/chenmingan/workplace/paddle/prop_regr/weight/model_emb128_seed42_dp0_3Dplus2D_from_scratch_qm9pt.pkl"))
    print("Loading pretrain model")
    model = DownStreamModel(encoder=encoder)
    print("DownStreamModel parameter size:", calc_parameter_size(model.parameters()), flush=True)

    criterion = nn.MSELoss()
    lr = optimizer.lr.CosineAnnealingDecay(learning_rate=lr, T_max=tmax)
    opt = optimizer.Adam(lr, parameters=model.parameters(), weight_decay=weight_decay)

    best_score = 1e9
    best_epoch = 0
    best_train_metric = {}
    best_valid_metric = {}

    for epoch in range(max_epoch):
        model.train()
        for (atom_bond_graph, bond_angle_graph, label_true) in train_data_loader:
            output = model(atom_bond_graph, bond_angle_graph)
            label_true = paddle.to_tensor(label_true, dtype=paddle.float32).unsqueeze(axis=1)
            loss = criterion(output, label_true)
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

        print('epoch', epoch, flush=True)
        print('train', train_metric, flush=True)
        print('valid', valid_metric, flush=True)
        print(f'current_best_score: {best_score:.4f}, best_epoch: {best_epoch}', flush=True)
        print('=================================================', flush=True)

        if epoch > best_epoch + max_bearable_epoch or epoch == max_epoch - 1:
            print(f"model_{model_version} is Done!!", flush=True)
            print('train', flush=True)
            print(best_train_metric, flush=True)
            print('valid', flush=True)
            print(best_valid_metric, flush=True)
            print(f'current_best_score: {best_score:.4f}, best_epoch: {best_epoch}', flush=True)
            break


def evaluate(model, dataloader):
    """评估模型"""
    model.eval()
    y_pred = np.array([])
    y_true = np.array([])
    for (atom_bond_graph, bond_angle_graph, compound_class) in dataloader:
        output = model(atom_bond_graph, bond_angle_graph)
        y_pred = np.concatenate((y_pred, output[:, 0].cpu().numpy()))
        y_true = np.concatenate((y_true, compound_class))

    r2 = metrics.r2_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))

    metric_eval = {
        'r2': round(r2, 4),
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4)
    }

    return metric_eval

# 开始训练
# 固定随机种子
SEED = 42
paddle.seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
seed_data = 42
# seed: 42 666 3407 1234 2024

# batch_size = 32
batch_size = 32          
lr = 1e-3
tmax = 15
weight_decay = 1e-5
max_bearable_epoch = 200
max_epoch = 10000
dropout_rate = 0.3
model_version = f'GEM_emb128_seed42_dp0_3Dplus2D_ft_from_trainpt'

trial(model_version, batch_size, lr, tmax, weight_decay, max_bearable_epoch, max_epoch, seed_data, dropout_rate)