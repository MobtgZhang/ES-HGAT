import os
import yaml
import argparse
import torch
from znlp.model import *

class Config:
    def __init__(self,load_file_name=None):
        self.get_yaml_file(load_file_name)
    def get_yaml_file(self,load_file_name):
        if load_file_name is None:
            return 
        with open(load_file_name,'r',encoding='utf8') as rfp:
            data_dict=yaml.safe_load(rfp)
        if data_dict is None:
            raise ValueError("The yaml file is empty!")
        else:
            for key in data_dict:
                setattr(self,key,data_dict[key])
    def add_attr(self,name,value):
        setattr(self,name,value)
    def delete_attr(self,name):
        if hasattr(self,name):
            delattr(self,name)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',default='./data',type=str)
    parser.add_argument('--log-dir',default='./log',type=str)
    parser.add_argument('--save-attention',action='store_true')
    parser.add_argument('--config-dir',default='./config',type=str)
    parser.add_argument('--config-file',default=None,type=str)
    parser.add_argument('--pretrain-path',default=None,type=str)
    parser.add_argument('--optim',default="AdamW",type=str)
    parser.add_argument('--optim-step',default=1,type=int)
    parser.add_argument('--result-dir',default='./result',type=str)
    parser.add_argument('--model-name',default='ESHGAT',type=str)
    parser.add_argument('--epoches',default=40,type=int)
    parser.add_argument('--checkpoint-file',default=None,type=str)
    parser.add_argument('--batch-size',default=32,type=int)
    parser.add_argument('--dataset',default='CLUEEmotion2020',type=str)
    parser.add_argument('--learning-rate',default=2e-3,type=float)
    parser.add_argument('--gamma',default=0.85,type=float)
    parser.add_argument('--cuda',action='store_false')
    parser.add_argument('--random',action='store_true')
    parser.add_argument('--train-evaluate',action='store_true')
    parser.add_argument('--device-id',type=int,default=0)
    parser.add_argument('--mat-type',default="entropy",type=str)
    parser.add_argument('--window',default=4,type=int)
    args = parser.parse_args()
    return args

def check_args(args):
    result_dir = os.path.join(args.result_dir,args.dataset)
    model_dir = os.path.join(args.log_dir,args.dataset,args.model_name)
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    assert os.path.exists(result_dir)
    config_file = os.path.join(args.config_dir,args.model_name+".yaml")
    assert ((args.config_file and os.path.exists(args.config_file)) or os.path.exists(config_file))
    if not args.config_file or not os.path.exists(args.config_file):
        args.config_file = config_file
    assert args.dataset in os.listdir(args.result_dir)
    if args.pretrain_path is None:
        args.pretrain_path = "bert-base-chinese"
def get_models(args,config=None,**kwargs):
    model_dict = {
        "TextCNN":TextCNN,
        "TextCNNPre":TextCNNPre,
        "CapsuleNet":CapsuleNet,
        "CapsuleNetPre":CapsuleNetPre,
        "HyperGAT":HyperGAT,
        "HyperGATPre":HyperGATPre,
        "TextGCN":TextGCN,
        "TextGCNPre":TextGCNPre,
        "ESHGAT":ESHGAT,
        "GraphESHGAT":GraphESHGAT,
        "PretrainingModel":PretrainingModel,
    }
    assert args.model_name in model_dict
    config = config or Config(args.config_file)
    for key in kwargs:
        config.add_attr(key,kwargs[key])
    if not args.checkpoint_file:
        model = model_dict[args.model_name](config)
    else:
        model  = model_dict[args.model_name](config)
        model.load_state_dict(torch.load(args.checkpoint_file), strict=True)
    return model

    