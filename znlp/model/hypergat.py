import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel

from ..layers import HyperGraphAttentionLayerSparse
from ..layers import MutiAttentionLayer,SFU

class HyperGAT(nn.Module):
    def __init__(self):
        super(HyperGAT,self).__init__()
    def forward(self):
        pass
class HyperGATPre(nn.Module):
    def __init__(self,config):
        super(HyperGATPre,self).__init__()
        self.hid_dim = config.n_hid
        self.out_dim = config.out_dim
        self.dropout = config.dropout
        
        self.pretrain = AutoModel.from_pretrained(config.pretrain_path)
        self.graph = HyperGraphAttentionLayerSparse(self.w_embedding_dim,self.w_embedding_dim,self.dropout,alpha=0.1,transfer=True)
        # funsion layer
        self.att = MutiAttentionLayer(self.w_embedding_dim,self.hid_dim,self.hid_dim)
        self.sfu = SFU(self.hid_dim,self.hid_dim)
        self.encode = nn.Sequential(
            nn.Linear(self.pretrain.config.hidden_size,self.hid_dim),
            nn.GELU()
        )
        self.class_size = config.class_size
        self.pred = nn.Linear(self.out_dim,self.class_size)
    def forward(self,input_ids,attention_mask,adjmat,att_output=False,**kwargs):
        pooled_out = self.pretrain(input_ids=input_ids,attention_mask=attention_mask)
        dc_hid = pooled_out['last_hidden_state']
        dc_hid = self.encode(dc_hid)
        gw_hid = self.graph(gw_hid,adjmat)
        # fusion layer
        datt_hid,datts = self.att(dc_hid,gw_hid)
        dsfu_hid = self.sfu(datt_hid,dc_hid)
        output = self.pred(dsfu_hid.sum(dim=1))
        logits = F.log_softmax(output,dim=-1)
        if att_output:
            return logits,datts
        else:
            return logits
    