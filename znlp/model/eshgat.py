import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from ..layers import VGAE,GraphConvolution,HyperGraphAttentionLayerSparse
from ..layers import SFU,MutiAttentionLayer
from ..layers import MultiCNNLayer
from ..layers import CapsuleLayer

class CapsESHGAT(nn.Module):
    def __init__(self,config):
        super(CapsESHGAT,self).__init__()
        self.config = config
        self.num_layers = config.num_layers
        self.hid_dim = config.hid_dim
        self.out_dim = config.out_dim
        self.dropout = config.dropout
        self.num_routing = config.num_routing
        self.num_capsules = config.num_capsules

        self.pretrain = AutoModel.from_pretrained(config.pretrain_path)
        self.caps = CapsuleLayer(self.num_capsules,self.pretrain.config.hidden_size,self.out_dim,self.num_routing)
        # output layer
        self.class_size = config.class_size
        self.pred = nn.Linear(self.out_dim,self.class_size)
    def forward(self,input_ids,attention_mask,adjmat,**kwargs):
        pr_outs = self.pretrain(input_ids=input_ids,attention_mask=attention_mask)
        outs = pr_outs["last_hidden_state"]
        #outs = self.encode(pr_outs["last_hidden_state"])
        caps_out = self.caps(outs)
        # output layer
        output = self.pred(caps_out.sum(dim=1))
        logits = F.log_softmax(output,dim=-1)
        return logits


class GraphESHGAT(nn.Module):
    def __init__(self,config):
        super(GraphESHGAT,self).__init__()
        self.config = config
        self.num_layers = config.num_layers
        self.hid_dim = config.hid_dim
        self.out_dim = config.out_dim
        self.dropout = config.dropout
        self.filter_sizes = config.filter_sizes

        self.pretrain = AutoModel.from_pretrained(config.pretrain_path)
        # graph processing
        self.graph_list = nn.ModuleList()
        self.vae_list = nn.ModuleList()
        for k in range(self.num_layers):
            if k==0:
                self.vae_list.append(VGAE(self.pretrain.config.hidden_size,self.hid_dim,self.hid_dim,self.dropout))
                self.graph_list.append(GraphConvolution(self.pretrain.config.hidden_size,self.hid_dim))
            else:
                self.vae_list.append(VGAE(self.hid_dim,self.hid_dim,self.hid_dim,self.dropout))
                self.graph_list.append(HyperGraphAttentionLayerSparse(self.hid_dim,self.hid_dim,self.dropout,
                                                                      alpha=0.1,transfer=True))
        self.cnn = MultiCNNLayer(self.hid_dim,self.hid_dim,self.filter_sizes,self.out_dim,self.dropout)
        
        # output layer
        self.class_size = config.class_size
        self.pred = nn.Linear(self.out_dim,self.class_size)
        
    def forward(self,input_ids,attention_mask,adjmat,**kwargs):
        pr_outs = self.pretrain(input_ids=input_ids,attention_mask=attention_mask)
        outs = pr_outs["last_hidden_state"]
        #outs = self.encode(pr_outs["last_hidden_state"])
        # The first channel
        g_hid = outs
        for k in range(self.num_layers):
            A_hid = self.vae_list[k](g_hid,adjmat)
            g_hid = self.graph_list[k](g_hid,A_hid)
        g_out = self.cnn(g_hid)
        output = self.pred(g_out.sum(dim=1))
        logits = F.log_softmax(output,dim=-1)
        return logits


class TestESHGAT(nn.Module):
    def __init__(self,config):
        super(TestESHGAT,self).__init__()
        self.config = config
        self.num_layers = config.num_layers
        self.hid_dim = config.hid_dim
        self.out_dim = config.out_dim
        self.dropout = config.dropout
        self.num_routing = config.num_routing
        self.num_capsules = config.num_capsules

        self.pretrain = AutoModel.from_pretrained(config.pretrain_path)
        # graph processing 
        self.graph_list = nn.ModuleList()
        self.re_layer1 = VGAE(self.pretrain.config.hidden_size,self.hid_dim,self.hid_dim,self.dropout)
        self.re_layer2 = VGAE(self.hid_dim,self.hid_dim,self.hid_dim,self.dropout)
        for k in range(self.num_layers):
            if k==0:
                self.graph_list.append(GraphConvolution(self.pretrain.config.hidden_size,self.hid_dim))
            else:
                self.graph_list.append(HyperGraphAttentionLayerSparse(self.hid_dim,self.hid_dim,self.dropout,
                                                                      alpha=0.1,transfer=True))
        
        self.caps = CapsuleLayer(self.num_capsules,self.hid_dim,self.out_dim,self.num_routing)
        # output layer
        self.class_size = config.class_size
        self.pred = nn.Linear(self.out_dim,self.class_size)
    def forward(self,input_ids,attention_mask,adjmat,**kwargs):
        pr_outs = self.pretrain(input_ids=input_ids,attention_mask=attention_mask)
        outs = pr_outs["last_hidden_state"]
        g_hid = outs
        for k in range(self.num_layers):
            if k==0:
                A_hid = self.re_layer1(g_hid,adjmat)
            else:
                A_hid = self.re_layer2(g_hid,adjmat)
            g_hid = self.graph_list[k](g_hid,A_hid)
        caps_out = self.caps(g_hid)
        # output layer
        output = self.pred(caps_out.sum(dim=1))
        logits = F.log_softmax(output,dim=-1)
        return logits


class ESHGAT(nn.Module):
    def __init__(self,config):
        super(ESHGAT,self).__init__()
        self.config = config
        self.num_layers = config.num_layers
        self.hid_dim = config.hid_dim
        self.out_dim = config.out_dim
        self.dropout = config.dropout
        self.num_routing = config.num_routing
        self.num_capsules = config.num_capsules
        self.filter_sizes = config.filter_sizes

        self.pretrain = AutoModel.from_pretrained(config.pretrain_path)
        # graph processing
        self.graph_list = nn.ModuleList()
        self.vae_list = nn.ModuleList()
        for k in range(self.num_layers):
            if k==0:
                self.vae_list.append(VGAE(self.pretrain.config.hidden_size,self.hid_dim,self.hid_dim,self.dropout))
                self.graph_list.append(GraphConvolution(self.pretrain.config.hidden_size,self.hid_dim))
            else:
                self.vae_list.append(VGAE(self.hid_dim,self.hid_dim,self.hid_dim,self.dropout))
                self.graph_list.append(HyperGraphAttentionLayerSparse(self.hid_dim,self.hid_dim,self.dropout,
                                                                      alpha=0.1,transfer=True))
        self.cnn = MultiCNNLayer(self.hid_dim,self.hid_dim,self.filter_sizes,self.out_dim,self.dropout)
        
        self.caps = CapsuleLayer(self.num_capsules,self.pretrain.config.hidden_size,self.out_dim,self.num_routing)
        # funsion layer
        self.sfu = SFU(self.out_dim,self.out_dim)
        # output layer
        self.class_size = config.class_size
        self.pred = nn.Linear(self.out_dim,self.class_size)

    def forward(self,input_ids,attention_mask,adjmat,**kwargs):
        """
        params: input_ids -->(b,s)
        params: attention_mask -->(b,s)
        params: adjmat -->(b,s,s)
        """
        pr_outs = self.pretrain(input_ids=input_ids,attention_mask=attention_mask)
        outs = pr_outs["last_hidden_state"]
        # The first channel
        g_hid = outs
        for k in range(self.num_layers):
            A_hid = self.vae_list[k](g_hid,adjmat)
            g_hid = self.graph_list[k](g_hid,A_hid)
        g_out = self.cnn(g_hid)
        # The second channel
        caps_out = self.caps(outs)
        # fusion layer
        fusion_out = self.sfu(caps_out,g_out)
        # output layer
        output = self.pred(fusion_out.sum(dim=1))
        logits = F.log_softmax(output,dim=-1)
        return logits


class ESCapsHGAT(nn.Module):
    def __init__(self,config):
        super(ESCapsHGAT,self).__init__()
        self.config = config
        self.num_layers = config.num_layers
        self.hid_dim = config.hid_dim
        self.out_dim = config.out_dim
        self.dropout = config.dropout
        self.num_capsules = config.num_capsules
        self.num_routing = config.num_routing

        self.pretrain = AutoModel.from_pretrained(config.pretrain_path)
        # graph processing
        self.graph_list = nn.ModuleList()
        self.vae_list = nn.ModuleList()
        for k in range(self.num_layers):
            if k==0:
                self.vae_list.append(VGAE(self.pretrain.config.hidden_size,self.hid_dim,self.hid_dim,self.dropout))
                self.graph_list.append(GraphConvolution(self.pretrain.config.hidden_size,self.hid_dim))
            else:
                self.vae_list.append(VGAE(self.hid_dim,self.hid_dim,self.hid_dim,self.dropout))
                self.graph_list.append(HyperGraphAttentionLayerSparse(self.hid_dim,self.hid_dim,self.dropout,
                                                                      alpha=0.1,transfer=True))
        self.poolcaps = CapsuleLayer(self.num_capsules,self.pretrain.config.hidden_size,self.hid_dim,self.num_routing)
        self.graphcaps = CapsuleLayer(self.num_capsules,self.hid_dim,self.hid_dim,self.num_routing)
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
        g_hid = outs
        caps_hid = self.poolcaps(outs)
        for k in range(self.num_layers):
            AMat = self.vae_list[k](g_hid,adjmat)
            g_hid = self.graph_list[k](g_hid,AMat)
        g_hid = self.graphcaps(g_hid)
        att_hid,atts = self.att(g_hid,caps_hid)
        mix_hid = self.sfu(att_hid,g_hid)
        # output layer
        output = self.pred(mix_hid.sum(dim=1))
        logits = F.log_softmax(output,dim=-1)
        if att_output:
            return logits,atts
        else:
            return logits