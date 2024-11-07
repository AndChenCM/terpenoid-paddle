from utils.pt_utils import DownstreamCollateFn, calc_parameter_size
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
import paddle.nn as nn
import pandas as pd
from utils.to_graph import transfer_smiles_to_graph
import pgl
from visnet import visnet
from visnet import visnet_output_modules
import gc
import os
warnings.filterwarnings('ignore')


class GEMData_mmff(object):
    def __init__(self, path, label_name=None):
        self.df = pd.read_csv(path)
        self.smiles = self.df['smiles'].values

        if label_name is not None:
            self.label = self.df[label_name].values
        else:
            self.label = np.zeros(len(self.df))

    def to_data_list(self, save_name, num_worker=20):
        smiles_to_graph_dict = transfer_smiles_to_graph(self.smiles, num_worker)
        data_list = []
        for i in range(len(self.smiles)):
            data_item = {}
            data_item['Smiles'] = self.smiles[i]
            data_item['Graph'] = smiles_to_graph_dict[self.smiles[i]]
            data_item['Label'] = self.label[i]


            #data_item['Label'] = self.label[i] 
            data_list.append(data_item)

        save_path = f'work/{save_name}.pkl'
        pickle.dump(data_list, open(save_path, 'wb'))

    def get_label_stat(self):
        return {
            'mean': np.mean(self.label),
            'std': np.std(self.label),
            'N': len(self.label)
        }

#data = GEMData_mmff('/home/chenmingan/projects/paddle/prop_regr_jiangxinyu/pre_visnet/data/data285818/zinc_sub_train.csv', label_name=None)
#label_stat = data.get_label_stat()
#print(f'label_stat: {label_stat}')
#label_mean = label_stat['mean']
#label_std = label_stat['std']
#data.to_data_list(save_name='zinc_sub_train', num_worker=40)

def get_data_loader(mode, batch_size=128):
    collate_fn = DownstreamCollateFn()
    if mode == 'train':
        data_list0 = pickle.load(open("work/train_semi_from_mol_clean.pkl", 'rb'))
        data_list1 = pickle.load(open("work/train_semi_from_smiles_clean.pkl", 'rb'))
        data_list = data_list0 + data_list1
        train, valid = train_test_split(data_list, random_state=42, test_size=0.1)
        train, valid = InMemoryDataset(train), InMemoryDataset(valid)

        print(f'len train is {len(train)}, len valid is {len(valid)}', flush=True)

        train_dl = train.get_data_loader(batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        valid_dl = valid.get_data_loader(batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        #collate_fn 把数据转换成atom_bond_graph, bond_angle_graph, np.array(compound_class_list, dtype=np.float32)

        return train_dl, valid_dl
    elif mode == 'test':
        return ValueError('This is a pretrain script')

class DownstreamVisNet(nn.Layer):
    def __init__(
            self,
            representation_model,
            output_model
    ):
        super().__init__()
        self.representation_model = representation_model
        self.output_model = output_model
        self.reset_parameters()

    def reset_parameters(self):
        self.representation_model.reset_parameters()
        self.output_model.reset_parameters()

    def forward(self, atom_bond_graph: pgl.Graph, feed_dict):

        x_orig, v_orig = self.representation_model(
                atom_bond_graph)

        loss = self.output_model.pre_reduce(x_orig,v_orig,feed_dict)
        
        return loss
    
def trial(model_version, batch_size, lr, tmax, weight_decay, max_bearable_epoch, max_epoch):
    train_data_loader, valid_data_loader = get_data_loader(mode='train', batch_size=batch_size) 
    os.makedirs(f"pretrain_weight/{model_version}", exist_ok=True)
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
   
   
    output_model =  visnet_output_modules.Pretrain_Output(
       hidden_channels=240,out_channels=1
    )

    model = DownstreamVisNet(
        representation_model,
        output_model
    )

    print("parameter size:", calc_parameter_size(model.parameters()), flush=True)

    #model_path = '/home/chenmingan/projects/paddle/prop_regr_jiangxinyu/pre_visnet/weight/model_pre_zinc_50w0.pkl'
    #model.set_state_dict(paddle.load(model_path))
    #print('Load state_dict from %s' % model_path)

    lr = optimizer.lr.CosineAnnealingDecay(learning_rate=lr, T_max=tmax)
    opt = optimizer.Adam(lr, parameters=model.parameters(), weight_decay=weight_decay)
   
    best_score = 1e9
    best_epoch = 0
    best_train_metric = {}
    best_valid_metric = {}

    for epoch in range(max_epoch):
        model.train()
        for batch_idx, (atom_bond_graph, _, feed_dict) in enumerate(train_data_loader):
            
            loss = model(atom_bond_graph.tensor(), feed_dict)
            # print(loss, flush=True)
            #  每隔100个批次打印一次损失
            if batch_idx % 100 == 0:
               print(f"Epoch [{epoch + 1}/{max_epoch}], Batch [{batch_idx}], Loss: {loss.numpy()}", flush=True)
            loss.backward() 

            opt.step()
            opt.clear_grad()
            
        lr.step() 

        # 评估模型在训练集、验证集的表现
        train_metric = evaluate(model, train_data_loader)
        valid_metric = evaluate(model, valid_data_loader)

        score = valid_metric

        
        paddle.save(model.state_dict(), f"pretrain_weight/{model_version}/"+model_version+f'{epoch}'+".pkl")
        if score < best_score:
            # 保存score最大时的模型权重
            paddle.save(model.state_dict(), f"pretrain_weight/{model_version}/"+model_version+"best.pkl")
            best_score = score
            best_epoch = epoch
            best_train_metric = train_metric
            best_valid_metric = valid_metric

        print('epoch', epoch, flush=True)
        print('train', train_metric, flush=True)
        print('valid', valid_metric, flush=True)
        print(f'current_best_score: {best_score:.4f}, best_epoch: {best_epoch}', flush=True)
        print('=================================================', flush=True)
        gc.collect()
        if epoch > best_epoch + max_bearable_epoch or epoch == max_epoch - 1:
            print(f"model_{model_version} is Done!!")
            print('train')
            print(best_train_metric)
            print('valid')
            print(best_valid_metric)
            print(f'current_best_score: {best_score:.4f}, best_epoch: {best_epoch}')
            break


def evaluate(model, dataloader):
    """评估模型"""
    model.eval()
    loss_list=[]

    for (atom_bond_graph, _, feed_dict) in dataloader:
        loss = model(atom_bond_graph.tensor(), feed_dict)

        loss = loss.cpu().numpy().flatten().astype(np.float32)
        loss_list.append(loss)
       
    return np.mean(loss_list)
# 开始训练
# 固定随机种子
SEED = 42
paddle.seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

batch_size = 32          
lr = 1e-3
tmax = 15
weight_decay = 1e-5
max_bearable_epoch = 5
max_epoch = 50

trial('visnet_hs80_l6_rbf32_lm2_pt_on_train_mol+smiles', batch_size, lr, tmax, weight_decay, max_bearable_epoch, max_epoch)



