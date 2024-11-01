#!/usr/bin/python                                                                                  
#-*-coding:utf-8-*- 
#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
VisNet pretrain using improved GEM methods
"""

import os
from os.path import join, exists, basename
import sys
import argparse
import time
import numpy as np
from glob import glob
import logging
from sklearn.model_selection import train_test_split
import paddle
import paddle.distributed as dist

from pahelix.datasets.inmemory_dataset import InMemoryDataset
from pahelix.utils import load_json_config
from utils.gem_featurizer import GeoPredTransformFn, GeoPredCollateFn
from utils.gem_model import GeoGNNModel, GeoPredModel
from visnet import visnet
from visnet import visnet_output_modules
import pickle
import warnings
warnings.filterwarnings('ignore')

def exempt_parameters(src_list, ref_list):
    """Remove element from src_list that is in ref_list"""
    res = []
    for x in src_list:
        flag = True
        for y in ref_list:
            if x is y:
                flag = False
                break
        if flag:
            res.append(x)
    return res

def train(args, model, optimizer, data_gen):
    """tbd"""
    model.train()
    
    # steps = get_steps_per_epoch(args)
    # step = 0
    list_loss = []
    for graph_dict, feed_dict in data_gen:
        # print('rank:%s step:%s' % (dist.get_rank(), step))
        # if dist.get_rank() == 1:
        #     time.sleep(100000)
        for k in graph_dict:
            graph_dict[k] = graph_dict[k].tensor()
        for k in feed_dict:
            feed_dict[k] = paddle.to_tensor(feed_dict[k])
        train_loss = model(graph_dict, feed_dict)
        train_loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        list_loss.append(train_loss.numpy().mean())
        # step += 1
        # if step > steps:
        #     print("jumpping out")
        #     break         
    return np.mean(list_loss)


@paddle.no_grad()
def evaluate(args, model, test_dataset, collate_fn):
    """tbd"""
    model.eval()
    data_gen = test_dataset.get_data_loader(
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            shuffle=True, 
            collate_fn=collate_fn)

    dict_loss = {'loss': []}
    for graph_dict, feed_dict in data_gen:
        for k in graph_dict:
            graph_dict[k] = graph_dict[k].tensor()
        for k in feed_dict:
            feed_dict[k] = paddle.to_tensor(feed_dict[k])
        loss, sub_losses = model(graph_dict, feed_dict, return_subloss=True)

        for name in sub_losses:
            if not name in dict_loss:
                dict_loss[name] = []
            v_np = sub_losses[name].numpy()
            dict_loss[name].append(v_np)
        dict_loss['loss'] = loss.numpy()
    dict_loss = {name: np.mean(dict_loss[name]) for name in dict_loss}
    return dict_loss


# def get_steps_per_epoch(args):
#     """tbd"""
#     # add as argument
#     if args.dataset == 'zinc':
#         train_num = int(20000000 * (1 - args.test_ratio))
#     else:
#         raise ValueError(args.dataset)
#     if args.DEBUG:
#         train_num = 100
#     steps_per_epoch = int(train_num / args.batch_size)
#     if args.distributed:
#         steps_per_epoch = int(steps_per_epoch / dist.get_world_size())
#     return steps_per_epoch


def load_smiles_to_dataset(data_path):
    """tbd"""
    files = sorted(glob('%s/*' % data_path))
    data_list = []
    for file in files:
        with open(file, 'r') as f:
            tmp_data_list = [line.strip() for line in f.readlines()]
        data_list.extend(tmp_data_list)
    dataset = InMemoryDataset(data_list=data_list)
    return dataset


def main(args):
    """tbd"""
    model_config = load_json_config(args.model_config)
    if not args.dropout_rate is None:
        model_config['dropout_rate'] = args.dropout_rate

    compound_encoder = visnet.ViSNetBlock(lmax=2,
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
    
    model = GeoPredModel(model_config, compound_encoder)
    # if args.distributed:
    #     model = paddle.DataParallel(model)
    opt = paddle.optimizer.Adam(learning_rate=args.lr, parameters=model.parameters())
    print('Total param num: %s' % (len(model.parameters())))
    for i, param in enumerate(model.named_parameters()):
        print(i, param[0], param[1].name)

    if not args.init_model is None and not args.init_model == "":
        compound_encoder.set_state_dict(paddle.load(args.init_model))
        print('Load state_dict from %s' % args.init_model)
    
    # get dataset
    dataset = load_smiles_to_dataset(args.data_path)
    print('Total size:%s' % (len(dataset)))
    save_path = f'work/pretrain_data_trainset_wH.pkl'
    if not exists(save_path):
        
        # print('Dataset smiles min/max/avg length: %s/%s/%s' % (
        #         np.min(smiles_lens), np.max(smiles_lens), np.mean(smiles_lens)))
        transform_fn = GeoPredTransformFn(model_config['pretrain_tasks'], model_config['mask_ratio'])
        # this step will be time consuming due to rdkit 3d calculation
        dataset.transform(transform_fn, num_workers=args.num_workers)
        pickle.dump(dataset, open(save_path, 'wb'))

    data_list = pickle.load(open("work/pretrain_data_trainset_wH.pkl", 'rb'))
    train_dataset, test_dataset = train_test_split(data_list, random_state=42, test_size=0.1)
    train_dataset, test_dataset = InMemoryDataset(train_dataset), InMemoryDataset(test_dataset)

    collate_fn = GeoPredCollateFn(
            atom_names=["atomic_num", "formal_charge", "degree", "chiral_tag", "total_numHs", "is_aromatic",
                      "hybridization","atom_pos","explicit_valence"],
            bond_names=["bond_dir", "bond_type", "is_in_ring"], 
            bond_float_names=["bond_length"],
            bond_angle_float_names=["bond_angle"],
            pretrain_tasks=model_config['pretrain_tasks'],
            mask_ratio=model_config['mask_ratio'],
            Cm_vocab=model_config['Cm_vocab'])

    train_data_gen = train_dataset.get_data_loader(
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            shuffle=True, 
            collate_fn=collate_fn)
    
    list_test_loss = []
    for epoch_id in range(args.max_epoch):
        s = time.time()
        train_loss = train(args, model, opt, train_data_gen)
        test_loss = evaluate(args, model, test_dataset, collate_fn)
        # if not args.distributed or dist.get_rank() == 0:
        paddle.save(compound_encoder.state_dict(), 
            '%s/epoch%d.pdparams' % (args.model_dir, epoch_id))
        list_test_loss.append(test_loss['loss'])
        print("epoch:%d train/loss:%s" % (epoch_id, train_loss))
        print("epoch:%d test/loss:%s" % (epoch_id, test_loss))
        print("Time used:%ss" % (time.time() - s))
    
    # if not args.distributed or dist.get_rank() == 0:
    print('Best epoch id:%s' % np.argmin(list_test_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--save_pickle_path", type=str, default=None)
    parser.add_argument("--removeHs", action='store_true', default=False, help="Remove hydrogen atoms")
    parser.add_argument("--test_ratio", type=float, default=0.1)
    # parser.add_argument("--compound_encoder_config", type=str)
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--init_model", type=str)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    args = parser.parse_args()

    main(args)