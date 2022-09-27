import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import VGAE, GraphConvolution
from ..layers import SFU
from ..layers import MutiAttentionLayer
from ..layers import CapsuleLayer
from ..layers import MatchNetwork

class FGGNN(nn.Module):
    def __init__(self,config,single=False,w_embedding=None,c_embedding=None):
        super(FGGNN,self).__init__()
        self.single = single
        self.num_layers = config.num_layers
        self.hid_dim = config.hid_dim
        self.out_dim = config.out_dim
        self.dropout = config.dropout
        self.capsules_dim = config.capsules_dim
        self.num_capsules = config.num_capsules
        self.dim_v = config.dim_v
        self.dim_k = config.dim_k
        # words embedding
        if w_embedding is None:
            self.num_words = config.num_words
            self.w_embedding_dim = config.w_embedding_dim
            self.w_embedding = nn.Embedding(self.num_words,self.w_embedding_dim)
        else:
            self.num_words,self.w_embedding_dim = w_embedding.shape
            self.w_embedding = nn.Embedding.from_pretrained(w_embedding)
        self.w_mask_embedding = nn.Embedding(2,self.w_embedding_dim)
        # chars embedding
        if c_embedding is None:
            self.num_chars = config.num_chars
            self.c_embedding_dim = config.c_embedding_dim
            self.c_embedding = nn.Embedding(self.num_chars,self.c_embedding_dim)
        else:
            self.num_chars,self.c_embedding_dim = c_embedding.shape
            self.c_embedding = nn.Embedding.from_pretrained(c_embedding)
        self.c_mask_embedding = nn.Embedding(2,self.c_embedding_dim)
        self.lstm = nn.LSTM(self.c_embedding_dim,self.hid_dim//2,batch_first=True,bidirectional=True)
        # graph processing
        self.graph_list = nn.ModuleList()
        self.vae_list = nn.ModuleList()
        for k in range(self.num_layers):
            if k==0:
                self.vae_list.append(VGAE(self.w_embedding_dim,self.hid_dim,self.hid_dim,self.dropout))
                self.graph_list.append(GraphConvolution(self.w_embedding_dim,self.hid_dim))
            else:
                self.vae_list.append(VGAE(self.hid_dim,self.hid_dim,self.hid_dim,self.dropout))
                self.graph_list.append(GraphConvolution(self.hid_dim,self.hid_dim))
        # funsion layer
        self.att = MutiAttentionLayer(self.hid_dim,self.hid_dim,self.hid_dim)
        self.sfu = SFU(self.hid_dim,self.hid_dim)
        # capsule layer
        self.caps = CapsuleLayer(self.num_capsules,self.hid_dim,self.capsules_dim,self.num_capsules)
        # output layer
        if single:
            self.class_size = config.class_size
            self.pred = nn.Linear(self.capsules_dim,self.class_size)
        else:
            self.class_size = config.class_size
            self.label_size = config.label_size
            self.pred = MatchNetwork(self.capsules_dim,self.hid_dim,self.label_size,self.class_size,self.dropout)
    def forward(self,chars2ids,c_mask,words2ids,w_mask,entropy_mat):
        # chars embedding and encoding
        c_emb = self.c_embedding(chars2ids)
        c_mask_emb = self.c_mask_embedding(c_mask)
        c_out = c_emb*torch.sigmoid(c_mask_emb)        
        dc_hid,_ = self.lstm(c_out)
        # words embedding and graph encoding
        w_emb = self.w_embedding(words2ids)
        w_mask_emb = self.c_mask_embedding(w_mask)
        gw_hid = w_emb*torch.sigmoid(w_mask_emb)
        for k in range(self.num_layers):
            A_hid = self.vae_list[k](gw_hid,entropy_mat)
            gw_hid = self.graph_list[k](gw_hid,A_hid)
        # fusion layer
        datt_hid,datts = self.att(dc_hid,gw_hid)
        dsfu_hid = self.sfu(datt_hid,dc_hid)
        # capsule layer 
        dcaps_hid = self.caps(dsfu_hid)
        # output layer
        if self.single:
            output = self.pred(dcaps_hid.sum(dim=1))
            logits = F.log_softmax(output,dim=1)
        else:
            logits = self.pred(dcaps_hid)
        return logits,datts

