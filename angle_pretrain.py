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
import paddle.nn as nn
import pandas as pd
from utils.to_graph import transfer_smiles_to_graph
import pgl
from visnet import visnet
from visnet import visnet_output_modules
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

def get_data_loader(mode, batch_size=256):
    collate_fn = DownstreamCollateFn()
    if mode == 'train':
        data_list = pickle.load(open("/home/chenmingan/projects/paddle/prop_regr_jiangxinyu/work/train_2D.pkl", 'rb'))
        train, valid = train_test_split(data_list, random_state=42, test_size=0.1)
        train, valid = InMemoryDataset(train), InMemoryDataset(valid)

        print(f'len train is {len(train)}, len valid is {len(valid)}')

        train_dl = train.get_data_loader(batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        valid_dl = valid.get_data_loader(batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        #collate_fn 把数据转换成atom_bond_graph, bond_angle_graph, np.array(compound_class_list, dtype=np.float32)

        return train_dl, valid_dl
    elif mode == 'test':
        data_list = pickle.load(open("work/test_2D.pkl", 'rb'))

        print(f'len test is {len(data_list)}')

        test = InMemoryDataset(data_list)
        test_dl = test.get_data_loader(batch_size=batch_size, shuffle=False)
        return test_dl

class GetLoss(nn.Layer):
    def __init__(self, encoder, label_mean=0.0, label_std=1.0):
        super(GetLoss, self).__init__()
        self.encoder = encoder
        self.criterion = nn.CrossEntropyLoss()#nn.MSELoss()
        self.label_mean = paddle.to_tensor(label_mean)
        self.label_std = paddle.to_tensor(label_std)
    
    def _get_scaled_label(self, x):
        return (x - self.label_mean) / (self.label_std + 1e-5)

    def _get_unscaled_pred(self, x):
        return x * (self.label_std + 1e-5) + self.label_mean
    
    def forward(self, atom_bond_graph, masked_atom_bond_graph, feed_dict):
        #label_true = paddle.to_tensor(label_true, dtype=paddle.int32, place=paddle.CUDAPlace(0)).unsqueeze(axis=1)
        x = self.encoder(atom_bond_graph.tensor(), masked_atom_bond_graph.tensor(), feed_dict)

        #else:
            # 没有掩码的情况下，正常计算损失
        #    loss = self.criterion(x, label_true)

        return x

    '''def forward(self, atom_bond_graph, bond_angle_graph, label_true,):
        label_true = paddle.to_tensor(label_true, dtype=paddle.float32, place=paddle.CUDAPlace(0)).unsqueeze(axis=1)
        x = self.encoder(atom_bond_graph.tensor())
        #x = self.mlp(graph_repr)

        if self.label_mean != 0 or self.label_std != 1:
            scaled_label = self._get_scaled_label(label_true)
            loss = self.criterion(x, scaled_label)
            pred = self._get_unscaled_pred(x)
        else:
            loss = self.criterion(x, label_true)
            pred = x
        
        return pred, loss'''
    
def trial(model_version, batch_size, lr, tmax, weight_decay, max_bearable_epoch, max_epoch,label_mean,label_std):
    train_data_loader, valid_data_loader = get_data_loader(mode='train', batch_size=batch_size) 
    
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
   
   
    output_model =  visnet_output_modules.Pretrain_Output(
       hidden_channels=384,out_channels=1
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
    model = GetLoss(encoder=visnet_model,label_mean=label_mean,label_std=label_std)
    print("parameter size:", calc_parameter_size(model.parameters()))

    #model_path = '/home/chenmingan/projects/paddle/prop_regr_jiangxinyu/pre_visnet/weight/model_pre_zinc_50w0.pkl'
    #model.set_state_dict(paddle.load(model_path))
    #print('Load state_dict from %s' % model_path)

    
    lr = optimizer.lr.CosineAnnealingDecay(learning_rate=lr, T_max=tmax)
    opt = optimizer.Adam(lr, parameters=model.parameters(), weight_decay=weight_decay)
   
    best_score = 1e9
    best_epoch = 0
    best_train_metric = {}
    best_valid_metric = {}
    import csv
    csv_file = "training_metrics_pre_zinc_20w.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'train_loss', 'val_loss'])  # 写入表头
    import logging
    from tqdm import tqdm
    for epoch in range(max_epoch):
        model.train()
        for (atom_bond_graph, masked_atom_bond_graph, feed_dict) in tqdm(train_data_loader):
            
            loss,_,_ = model(atom_bond_graph, masked_atom_bond_graph, feed_dict)
            #print(loss)
             # 每隔100个批次打印一次损失
            #if batch_idx % 100 == 0:
            #    print(f"Epoch [{epoch + 1}/{max_epoch}], Batch [{batch_idx}], Loss: {loss.numpy()}")
            loss.backward() 

            opt.step()
            opt.clear_grad()
            
        lr.step() 

        # 评估模型在训练集、验证集的表现
        train_metric = evaluate(model, train_data_loader,label_mean,label_std)
        valid_metric = evaluate(model, valid_data_loader,label_mean,label_std)

        score = valid_metric
        
        paddle.save(model.state_dict(), "weight/"+model_version+f'{epoch}'+".pkl")
        if score < best_score:
            # 保存score最大时的模型权重
            paddle.save(model.state_dict(), "weight/"+model_version+".pkl")
            best_score = score
            best_epoch = epoch
            best_train_metric = train_metric
            best_valid_metric = valid_metric
            
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, train_metric, valid_metric])

        print('epoch', epoch)
        print('train', train_metric)
        print('valid', valid_metric)
        print(f'current_best_score: {best_score:.4f}, best_epoch: {best_epoch}')
        print('=================================================')

        if epoch > best_epoch + max_bearable_epoch or epoch == max_epoch - 1:
            print(f"model_{model_version} is Done!!")
            print('train')
            print(best_train_metric)
            print('valid')
            print(best_valid_metric)
            print(f'current_best_score: {best_score:.4f}, best_epoch: {best_epoch}')
            break


def evaluate(model, dataloader,label_mean,label_std):
    """评估模型"""
    model.eval()
    pred_list=[]
    label_list=[]
    for (atom_bond_graph, masked_atom_bond_graph, feed_dict) in dataloader:
        _,pred,label= model(atom_bond_graph, masked_atom_bond_graph, feed_dict)

        preds_cpu = pred.cpu().numpy().flatten().astype(np.float32)
        labels_cpu = label.cpu().numpy().flatten().astype(np.float32)

        pred_list.append(preds_cpu)
        label_list.append(labels_cpu)
    #print(pred_list)
    rmse = np.sqrt(metrics.mean_squared_error(np.concatenate(pred_list),np.concatenate(label_list)))
    #r2 = metrics.r2_score(np.concatenate(pred_list),np.concatenate(label_list))
        
    return np.mean(rmse)
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
max_bearable_epoch = 100
max_epoch = 1000

trial('model_pre_zinc_50w_pre_train', batch_size, lr, tmax, weight_decay, max_bearable_epoch, max_epoch,0,1)



