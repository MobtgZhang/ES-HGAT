import os
import yaml
import argparse
import numpy as np
import torch

from znlp.models import *
from znlp.data import Dictionary
from znlp.models.textcnn import TextCNNWords
class Config:
    def __init__(self,load_file_name=None):
        self.get_yaml_file(load_file_name)
    def get_yaml_file(self,load_file_name):
        if load_file_name is None:
            return 
        with open(load_file_name,'r',encoding='utf8') as rfp:
            data_dict=yaml.safe_load(rfp)
        for key in data_dict:
            setattr(self,key,data_dict[key])
    def add_attr(self,name,value):
        setattr(self,name,value)
    def delete_attr(self,name):
        if hasattr(self,name):
            delattr(self,name)
def get_config(args,config = None):
    config = config or Config(args.config_file)
    if args.dataset == "AiChallenger2018":
        config.add_attr("single",False)
        config.add_attr("label_size",20)
        config.add_attr("class_size",4)
        args.lang = 'zh'
    elif args.dataset in ["wrime-ver1","wrime-ver2"]:
        config.add_attr("single",False)
        config.add_attr("label_size",8)
        config.add_attr("class_size",4)
        args.lang = 'jp'
    elif args.dataset == "CarsReviews":
        config.add_attr("single",False)
        config.add_attr("label_size",10)
        config.add_attr("class_size",4)
        args.lang = 'zh'
    elif args.dataset == "CLUEEmotion2020":
        config.add_attr("single",True)
        config.add_attr("class_size",7)
        args.lang = 'zh'
    elif args.dataset == "SimplifyWeibo4Moods":
        config.add_attr("single",True)
        config.add_attr("class_size",4)
        args.lang = 'zh'
    elif args.dataset == "TouTiaoNews":
        config.add_attr("single",True)
        config.add_attr("class_size",15)
        args.lang = 'zh'
    elif args.dataset == "NLPCC2018Task1":
        config.add_attr("single",True)
        config.add_attr("class_size",5)
        args.lang = 'zh'
    else:
        raise ValueError("The doesn't exist %s dataset."%args.dataset)
    return config
def get_models(args):
    model_dict = {
        "TextCNNWords":TextCNNWords,
        "TextCNNChars":TextCNNChars,
        "TextCNNBert":PretrainCNNModel,
        "TextCNNXlNet":PretrainCNNModel,
        "TextCNNAlbert":PretrainCNNModel,
        "TextCNNRoberta":PretrainCNNModel,
        "TextRCNN":TextRCNN,
        "CapsuleNetWords":CapsuleNetWords,
        "CapsuleNetChars":CapsuleNetChars,
        "CapsuleNetBert":PrtrainCapsModel,
        "CapsuleNetXlNet":PrtrainCapsModel,
        "CapsuleNetAlBert":PrtrainCapsModel,
        "CapsuleNetRobeta":PrtrainCapsModel,
        "HyperGAT":HyperGAT,
        "HyperGATBert":PtrainGATModel,
        "HyperGATXlNet":PtrainGATModel,
        "HyperGATAlBert":PtrainGATModel,
        "HyperGATRobeta":PtrainGATModel,
        "TextING":TextING,
        "TextGCN":TextGCN,
        "TextGCNBert":PretrainGCNModel,
        "TextGCNXlNet":PretrainGCNModel,
        "TextGCNAlBert":PretrainGCNModel,
        "TextGCNRoberta":PretrainGCNModel,
        "FGGNN-Bert":FGGNN, #
        "FGGNN-XLNet":FGGNN, #
        "FGGNN-AlBert":FGGNN, #
        "FGGNN-Roberta":FGGNN, #
        "FGGNN1":FGGNN,
        "FGGNN2":FGGNN,
        "FGGNN4":FGGNN,
        "FGGNN5":FGGNN,
        "GraphFGGNN":GraphFGGNN,
        "Bert":PretrainingModel, #
        "XLNet":PretrainingModel, #
        "AlBert":PretrainingModel, #
        "Roberta":PretrainingModel #
    }
    assert args.model_name in model_dict
    config = get_config(args)
    emd_dict = {
        "w_embedding":None,
        "c_embedding":None,
    }
    if not args.random:
        words_embedding_file = os.path.join(args.result_dir,args.dataset,"words-embedding.npy")
        chars_embedding_file = os.path.join(args.result_dir,args.dataset,"chars-embedding.npy")
        
        if os.path.exists(words_embedding_file):
            emd_dict["w_embedding"] = torch.tensor(np.load(words_embedding_file),dtype=torch.float)
        if os.path.exists(chars_embedding_file):
            emd_dict["c_embedding"] = torch.tensor(np.load(chars_embedding_file),dtype=torch.float)
    else:
        chars_dict_file = os.path.join(args.result_dir,args.dataset,"chars-dict.json")
        words_dict_file = os.path.join(args.result_dir,args.dataset,"words-dict.json")
        chars_dict = Dictionary.load(chars_dict_file)
        words_dict = Dictionary.load(words_dict_file)
        config.num_words = len(words_dict)
        config.num_chars = len(chars_dict)
    if not args.checkpoint_file:
        model = model_dict[args.model_name](config,**emd_dict)
    else:
        model  = model_dict[args.model_name](config)
        model.load_state_dict(torch.load(args.checkpoint_file), strict=True)
    return model
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
    if args.dataset in ["AiChallenger2018","CLUEEmotion2020","CarsReviews","SimplifyWeibo4Moods","TouTiaoNews","NLPCC2018Task1"]:
        args.lang = 'zh'
    elif args.dataset in ["wrime-ver1","wrime-ver2"]:
        args.lang = 'jp'
    else:
        raise ValueError("The doesn't exist %s dataset."%args.dataset)
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',default='./data',type=str)
    parser.add_argument('--log-dir',default='./log',type=str)
    parser.add_argument('--config-dir',default='./config',type=str)
    parser.add_argument('--config-file',default=None,type=str)
    parser.add_argument('--optim',default="AdamW",type=str)
    parser.add_argument('--optim-step',default=1,type=int)
    parser.add_argument('--result-dir',default='./result',type=str)
    parser.add_argument('--model-name',default='FGGNN',type=str)
    parser.add_argument('--epoches',default=40,type=int)
    parser.add_argument('--checkpoint-file',default=None,type=str)
    parser.add_argument('--batch-size',default=40,type=int)
    parser.add_argument('--dataset',default='CLUEEmotion2020',type=str)
    parser.add_argument('--learning-rate',default=2e-3,type=float)
    parser.add_argument('--gamma',default=0.85,type=float)
    parser.add_argument('--cuda',action='store_false')
    parser.add_argument('--random',action='store_true')
    parser.add_argument('--device-id',type=int,default=0)
    args = parser.parse_args()
    return args  


