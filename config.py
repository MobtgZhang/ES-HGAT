import os
import yaml
import argparse
import torch

from znlp.models import *
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
        config.add_attr("label_size",20)
        config.add_attr("class_size",4)
        args.lang = 'zh'
    elif args.dataset in ["wrime-ver1","wrime-ver2"]:
        config.add_attr("label_size",8)
        config.add_attr("class_size",4)
        args.lang = 'jp'
    elif args.dataset == "CarsReviews":
        config.add_attr("label_size",10)
        config.add_attr("class_size",4)
        args.lang = 'zh'
    elif args.dataset == "CLUEEmotion2020":
        config.add_attr("class_size",7)
        args.lang = 'zh'
    elif args.dataset == "SimplifyWeibo4Moods":
        config.add_attr("class_size",4)
        args.lang = 'zh'
    else:
        raise ValueError("The doesn't exist %s dataset."%args.dataset)
    return config
def get_models(args):
    groupA_list = ["TextCNN","TextRCNN","CapsuleNet","BiLSTMAtt","CharLSTM"]
    groupB_list = ["HyperGAT","TextING","TextGCN"]
    groupC_list = ["BERT","XLNET"]
    all_list = groupA_list + groupB_list + groupC_list
    assert args.model_name in all_list
    model_dict = {
        "TextCNN":TextCNN,
        "TextRCNN":TextRCNN,
        "CapsuleNet":CapsuleNet,
        "BiLSTMAtt":BiLSTMAtt,
        "CharLSTM":CharLSTM,
        "HyperGAT":HyperGAT,
        "TextING":TextING,
        "TextGCN":TextGCN,
        "FGGNN":FGGNN
    }
    config = get_config(args)
    words_embedding_file = os.path.join(args.result_dir,args.dataset,"words-embedding.npy")
    config.add_attr("words_embedding_file",words_embedding_file)
    if args.lang not in ["en"]:
        chars_embedding_file = os.path.join(args.result_dir,args.dataset,"chars-embedding.npy")
    else:
        chars_embedding_file = words_embedding_file
    config.add_attr("chars_embedding_file",chars_embedding_file)
    if not args.checkpoint_file:
        model  = model_dict[args.model_name].from_pretrain(config)
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
    assert (os.path.exists(args.config_file) or os.path.exists(config_file))
    if not os.path.exists(args.config_file):
        args.config_file = config_file
    if args.dataset in ["AiChallenger2018","CLUEEmotion2020","CarsReviews","SimplifyWeibo4Moods"]:
        args.lang = 'zh'
        if args.dataset in ["AiChallenger2018","CarsReviews"]:
            args.multiple = True
        elif args.dataset in ["SimplifyWeibo4Moods","CLUEEmotion2020"]:
            args.multiple = False
        else:
            raise ValueError("Unknow dataset %s"%str(args.dataset))
    elif args.dataset in ["wrime-ver1","wrime-ver2"]:
        args.lang = 'jp'
    else:
        raise ValueError("The doesn't exist %s dataset."%args.dataset)
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',default='./data',type=str)
    parser.add_argument('--log-dir',default='./log',type=str)
    parser.add_argument('--config-dir',default='./config',type=str)
    parser.add_argument('--config-file',default="",type=str)
    parser.add_argument('--optim',default="AdamW",type=str)
    parser.add_argument('--optim-step',default=1,type=int)
    parser.add_argument('--result-dir',default='./result',type=str)
    parser.add_argument('--model-name',default='FeatureAttGraph',type=str)
    parser.add_argument('--epoches',default=40,type=int)
    parser.add_argument('--checkpoint-file',default=None,type=str)
    parser.add_argument('--batch-size',default=40,type=int)
    parser.add_argument('--dataset',default='AiChallenger2018',type=str)
    parser.add_argument('--learning-rate',default=2e-4,type=float)
    parser.add_argument('--gamma',default=0.7,type=float)
    parser.add_argument('--cuda',action='store_false')
    args = parser.parse_args()
    return args  


