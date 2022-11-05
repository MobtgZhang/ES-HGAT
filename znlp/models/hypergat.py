import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import HyperGraphAttentionLayerSparse
from ..layers import MatchSimpleNet
from ..layers import SFU,MutiAttentionLayer

from transformers import XLNetModel,XLNetTokenizer
from transformers import BertModel,BertTokenizer
from transformers import RobertaModel,RobertaTokenizer
from transformers import AlbertModel,AlbertTokenizer

class HGNN_ATT(nn.Module):
    def __init__(self, input_size, n_hid, dropout=0.3):
        super(HGNN_ATT, self).__init__()
        self.dropout = dropout
        self.gat1 = HyperGraphAttentionLayerSparse(input_size, input_size, dropout=self.dropout, alpha=0.2, transfer = False, concat=True)
        self.lin = nn.Linear(input_size,n_hid)
        self.gelu = nn.GELU()
        self.gat2 = HyperGraphAttentionLayerSparse(n_hid, n_hid, dropout=self.dropout, alpha=0.2, transfer = True, concat=False)        
    def forward(self, x, adj_mat):
        x = self.gat1(x, adj_mat)
        x = self.gelu(self.lin(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gat2(x, adj_mat)
        return x

class HyperGAT(nn.Module):
    def __init__(self,config,w_embedding=None,**kwargs):
        super(HyperGAT,self).__init__()
        self.single = config.single
        self.dropout = config.dropout
        self.n_hid = config.n_hid
        if w_embedding is None:
            self.num_words = config.num_words
            self.embedding_dim = config.embedding_dim
            self.w_embedding = nn.Embedding(self.num_words,self.embedding_dim)
        else:
            self.num_words,self.embedding_dim = w_embedding.shape
            self.w_embedding = nn.Embedding.from_pretrained(w_embedding)
        self.gat = HGNN_ATT(self.embedding_dim,self.n_hid,self.dropout)
        if self.single:
            self.class_size = config.class_size
            self.pred = nn.Linear(self.n_hid,self.class_size)
        else:
            self.class_size = config.class_size
            self.label_size = config.label_size
            self.pred = MatchSimpleNet(self.n_hid,self.label_size,self.class_size)
    def forward(self,words2ids,paris_mat,**kwargs):
        """
        'words2ids', 'i_mask', 'content_chars', 'c_mask', 'content_words', 'w_mask', 'entropy_mat', 'paris_mat'
        """
        w_emb = self.w_embedding(words2ids)
        gat_hid = self.gat(w_emb,paris_mat)
        
        if self.single:
            logits = self.pred(gat_hid.sum(dim=1))
            logits = F.log_softmax(logits,dim=1)
        else:
            logits = self.pred(gat_hid)
        return logits

class PtrainGATModel(nn.Module):
    def __init__(self,config,w_embedding=None,**kwargs):
        super(PtrainGATModel,self).__init__()
        # chars embedding
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
        # words embedding
        if w_embedding is None:
            self.num_words = config.num_words
            self.w_embedding_dim = config.w_embedding_dim
            self.w_embedding = nn.Embedding(self.num_words,self.w_embedding_dim)
        else:
            self.num_words,self.w_embedding_dim = w_embedding.shape
            self.w_embedding = nn.Embedding.from_pretrained(w_embedding)
        self.w_mask_embedding = nn.Embedding(2,self.w_embedding_dim)
        self.hid_dim = config.n_hid
        self.out_dim = config.out_dim
        self.dropout = config.dropout
        self.single = config.single
        self.graph = HyperGraphAttentionLayerSparse(self.w_embedding_dim,self.w_embedding_dim,self.dropout,alpha=0.1,transfer=True)
        self.lin = nn.Linear(self.w_embedding_dim,self.hid_dim)
        self.gelu = nn.GELU()
        # funsion layer
        self.att = MutiAttentionLayer(self.hid_dim,self.hid_dim,self.hid_dim)
        self.sfu = SFU(self.hid_dim,self.hid_dim)
        self.encode = nn.Sequential(
            nn.Linear(self.pretrain.config.hidden_size,self.hid_dim),
            nn.GELU()
        )
        if self.single:
            self.class_size = config.class_size
            self.pred = nn.Linear(self.hid_dim,self.class_size)
        else:
            self.class_size = config.class_size
            self.label_size = config.label_size
            self.pred = MatchSimpleNet(self.hid_dim,self.label_size,self.class_size)
    def forward(self,content,words2ids,i_mask,paris_mat,**kwargs):
        dcaps_hid,_ = self.get_features(content,words2ids,i_mask,paris_mat)
        # output layer
        if self.single:
            output = self.pred(dcaps_hid.sum(dim=1))
            logits = F.log_softmax(output,dim=-1)
        else:
            logits = self.pred(dcaps_hid)
        return logits
    def get_features(self,content,words2ids,i_mask,paris_mat):
        # chars embedding and encoding
        device = next(self.parameters()).device
        outputs = self.tokenizer(content,return_tensors="pt",padding=True,truncation=True)
        outputs = outputs.to(device)
        pooled_out = self.pretrain( **outputs)
        dc_hid = pooled_out['last_hidden_state']
        dc_hid = self.encode(dc_hid)
        # words embedding and graph encoding
        w_emb = self.w_embedding(words2ids)
        w_mask_emb = self.w_mask_embedding(i_mask)
        gw_hid = w_emb*torch.sigmoid(w_mask_emb)
        gw_hid = self.graph(gw_hid,paris_mat)
        # fusion 
        gw_hid = self.gelu(self.lin(gw_hid))
        datt_hid,datts = self.att(dc_hid,gw_hid)
        dsfu_hid = self.sfu(datt_hid,dc_hid)
        return dsfu_hid,datts

