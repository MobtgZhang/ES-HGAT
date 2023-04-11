import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from ..layers import VGAE,GraphConvolution,HyperGraphAttentionLayerSparse
from ..layers import SFU,MutiAttentionLayer

class GraphESHGAT(nn.Module):
    def __init__(self,config):
        super(GraphESHGAT,self).__init__()
        self.num_layers = config.num_layers
        self.hid_dim = config.hid_dim
        self.out_dim = config.out_dim
        self.dropout = config.dropout
        # chars embedding
        
        # graph processing
        self.graph_list = nn.ModuleList()
        self.vae_list = nn.ModuleList()
        for k in range(self.num_layers):
            if k==0:
                self.vae_list.append(VGAE(self.w_embedding_dim,self.hid_dim,self.hid_dim,self.dropout))
                self.graph_list.append(GraphConvolution(self.w_embedding_dim,self.hid_dim))
            else:
                self.vae_list.append(VGAE(self.hid_dim,self.hid_dim,self.hid_dim,self.dropout))
                self.graph_list.append(HyperGraphAttentionLayerSparse(self.hid_dim,self.hid_dim,self.dropout,
                                                                      alpha=0.1,transfer=True))
        # output layer
        self.class_size = config.class_size
        self.pred = nn.Linear(self.hid_dim,self.class_size)
    def forward(self,input_ids,attention_mask,adjmat,**kwargs):
        pr_outs = self.pretrain(input_ids=input_ids,attention_mask=attention_mask)
        print(pr_outs)
        exit()
        outs = pr_outs["last_hidden_state"]
class ESHGAT(nn.Module):
    def __init__(self,config):
        super(ESHGAT,self).__init__()
        self.num_layers = config.num_layers
        self.hid_dim = config.hid_dim
        self.out_dim = config.out_dim
        self.dropout = config.dropout

        self.pretrain = AutoModel.from_pretrained(config.pretrain_path)
        # graph processing
        self.graph_list = nn.ModuleList()
        self.vae_list = nn.ModuleList()
        self.encode = nn.Sequential(
            nn.Linear(self.pretrain.config.hidden_size,self.hid_dim),
            nn.GELU()
        )
        for k in range(self.num_layers):
            if k==0:
                self.vae_list.append(VGAE(self.pretrain.config.hidden_size,self.hid_dim,self.hid_dim,self.dropout))
                self.graph_list.append(GraphConvolution(self.pretrain.config.hidden_size,self.hid_dim))
            else:
                self.vae_list.append(VGAE(self.hid_dim,self.hid_dim,self.hid_dim,self.dropout))
                self.graph_list.append(HyperGraphAttentionLayerSparse(self.hid_dim,self.hid_dim,self.dropout,
                                                                      alpha=0.1,transfer=True))
        # funsion layer
        self.att = MutiAttentionLayer(self.hid_dim,self.hid_dim,self.hid_dim)
        self.sfu = SFU(self.hid_dim,self.hid_dim)
        # output layer
        self.class_size = config.class_size
        self.pred = nn.Linear(self.hid_dim,self.class_size)

    def forward(self,input_ids,attention_mask,adjmat,att_output=False,**kwargs):
        """
        params: input_ids -->(b,s)
        params: attention_mask -->(b,s)
        params: adjmat -->(b,s,s)
        """
        pr_outs = self.pretrain(input_ids=input_ids,attention_mask=attention_mask)
        outs = pr_outs["last_hidden_state"] # (b,s,d)
        w_encode = self.encode(outs)
        gw_hid = outs
        for k in range(self.num_layers):
            A_hid = self.vae_list[k](gw_hid,adjmat)
            gw_hid = self.graph_list[k](gw_hid,A_hid)
        # fusion layer
        datt_hid,datts = self.att(w_encode,gw_hid)
        dcaps_hid = self.sfu(datt_hid,w_encode)
        # output layer
        output = self.pred(dcaps_hid.sum(dim=1))
        logits = F.log_softmax(output,dim=-1)
        if att_output:
            return logits,datts
        else:
            return logits

