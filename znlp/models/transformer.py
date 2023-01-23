
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel,BertTokenizer
from transformers import XLNetModel,XLNetTokenizer
from transformers import AlbertModel,AlbertTokenizer
from transformers import RobertaModel,RobertaTokenizer
from ..layers import MatchSimpleNet

class PretrainingModel(nn.Module):
    def __init__(self,config,**kwargs) -> None:
        super(PretrainingModel,self).__init__()
        """
        The selected model:
        bert-base-chinese
        clue/roberta_chinese_base
        voidful/albert_chinese_base
        hfl/chinese-xlnet-base
        """
        if "bert" in config.pretrain_path:
            self.pretrain = BertModel.from_pretrained(config.pretrain_path)
            self.tokenizer = BertTokenizer.from_pretrained(config.pretrain_path)
        elif "roberta" in config.pretrain_path:
            self.pretrain = RobertaModel.from_pretrained(config.pretrain_path)
            self.tokenizer = RobertaTokenizer.from_pretrained(config.pretrain_path)
        elif "albert" in config.pretrain_path:
            self.pretrain = AlbertModel.from_pretrained(config.pretrain_path)
            self.tokenizer = AlbertTokenizer.from_pretrained(config.pretrain_path)
        elif "xlnet" in config.pretrain_path:
            self.pretrain = XLNetModel.from_pretrained(config.pretrain_path)
            self.tokenizer = XLNetTokenizer.from_pretrained(config.pretrain_path)
        else:
            raise ValueError("Unknown model name: %s"%config.pretrain_path)
        self.hid_dim = self.pretrain.config.hidden_size
        self.single = config.single
        # output layer
        self.class_size = config.class_size
        self.pred = nn.Linear(self.hid_dim,self.class_size)
    def forward(self,content,**kwargs):
        device = next(self.parameters()).device
        outputs = self.tokenizer(content,return_tensors="pt",padding=True,truncation=True)
        outputs = outputs.to(device)
        pooled_out = self.pretrain(**outputs)
        pooled_out = pooled_out['last_hidden_state']
        # output layer
        output = self.pred(pooled_out.sum(dim=1))
        logits = F.log_softmax(output,dim=-1)
        return logits
