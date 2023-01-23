import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import VGAE, GraphConvolution,HyperGraphAttentionLayerSparse
from ..layers import SFU
from ..layers import MutiAttentionLayer
from ..layers import MatchNetwork,MatchSimpleNet

from transformers import XLNetModel,XLNetTokenizer
from transformers import BertModel,BertTokenizer
from transformers import RobertaModel,RobertaTokenizer
from transformers import AlbertModel,AlbertTokenizer

class FGGNNWord2Vec(nn.Module):
    def __init__(self,config,w_embedding=None,c_embedding=None,**kwargs):
        super(FGGNNWord2Vec,self).__init__()
        self.single = config.single
        self.num_layers = config.num_layers
        self.hid_dim = config.hid_dim
        self.out_dim = config.out_dim
        self.dropout = config.dropout
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
        self.lstm = nn.LSTM(self.c_embedding_dim,self.c_embedding_dim//2,bidirectional=True,batch_first=True)
        self.encode = nn.Sequential(
            nn.Linear(self.c_embedding_dim,self.hid_dim),
            nn.GELU()
        )
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
        # funsion layer
        self.att = MutiAttentionLayer(self.hid_dim,self.hid_dim,self.hid_dim)
        self.sfu = SFU(self.hid_dim,self.hid_dim)
        # output layer
        if self.single:
            self.class_size = config.class_size
            self.pred = nn.Linear(self.hid_dim,self.class_size)
        else:
            self.class_size = config.class_size
            self.label_size = config.label_size
            
            self.pred = MatchSimpleNet(self.hid_dim,self.label_size,self.class_size) 
    def get_features(self,content_chars,c_mask,words2ids,i_mask,entropy_mat):
        # chars embedding and encoding
        c_emb = self.c_embedding(content_chars)
        c_mask_emb = self.c_mask_embedding(c_mask)
        c_mask_emb = c_emb*torch.sigmoid(c_mask_emb)
        c_mask_emb,_ = self.lstm(c_mask_emb)
        cq_hid = self.encode(c_mask_emb)
        # words embedding and graph encoding
        w_emb = self.w_embedding(words2ids)
        w_mask_emb = self.w_mask_embedding(i_mask)
        gw_hid = w_emb*torch.sigmoid(w_mask_emb)
        for k in range(self.num_layers):
            A_hid = self.vae_list[k](gw_hid,entropy_mat)
            gw_hid = self.graph_list[k](gw_hid,A_hid)
        # fusion layer
        datt_hid,datts = self.att(cq_hid,gw_hid)
        dsfu_hid = self.sfu(datt_hid,cq_hid)
        return dsfu_hid,datts
    def forward(self,content_chars,c_mask,words2ids,i_mask,entropy_mat,**kwargs):
        """
        'words2ids', 'i_mask', 'content_chars', 'c_mask', 'content_words', 'w_mask', 'entropy_mat', 'paris_mat'
        """
        dcaps_hid,_ = self.get_features(content_chars,c_mask,words2ids,i_mask,entropy_mat)
        # output layer
        if self.single:
            output = self.pred(dcaps_hid.sum(dim=1))
            logits = F.log_softmax(output,dim=-1)
        else:
            logits = self.pred(dcaps_hid)
        return logits

class FGGNN(nn.Module):
    def __init__(self,config,w_embedding=None,**kwargs):
        super(FGGNN,self).__init__()
        self.single = config.single
        self.num_layers = config.num_layers
        self.hid_dim = config.hid_dim
        self.out_dim = config.out_dim
        self.dropout = config.dropout
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
        self.encode = nn.Sequential(
            nn.Linear(self.pretrain.config.hidden_size,self.hid_dim),
            nn.GELU()
        )
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
        # funsion layer
        self.att = MutiAttentionLayer(self.hid_dim,self.hid_dim,self.hid_dim)
        self.sfu = SFU(self.hid_dim,self.hid_dim)
        # output layer
        if self.single:
            self.class_size = config.class_size
            self.pred = nn.Linear(self.hid_dim,self.class_size)
        else:
            self.class_size = config.class_size
            self.label_size = config.label_size
            self.pred = MatchSimpleNet(self.hid_dim,self.label_size,self.class_size)
    def get_features(self,content,words2ids,i_mask,entropy_mat,**kwargs):
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
        for k in range(self.num_layers):
            A_hid = self.vae_list[k](gw_hid,entropy_mat)
            gw_hid = self.graph_list[k](gw_hid,A_hid)
        # fusion layer
        datt_hid,datts = self.att(dc_hid,gw_hid)
        # dsfu_hid = self.sfu(datt_hid,dc_hid)
        return datt_hid,datts
    def forward(self,content,words2ids,i_mask,entropy_mat,**kwargs):
        """
        'words2ids', 'i_mask', 'content_chars', 'c_mask', 'content_words', 'w_mask', 'entropy_mat', 'paris_mat'
        """
        dcaps_hid,_ = self.get_features(content,words2ids,i_mask,entropy_mat,**kwargs)
        # output layer
        if self.single:
            output = self.pred(dcaps_hid.sum(dim=1))
            logits = F.log_softmax(output,dim=-1)
        else:
            logits = self.pred(dcaps_hid)
        return logits

class GraphFGGNN(nn.Module):
    def __init__(self,config,w_embedding=None,**kwargs):
        super(GraphFGGNN,self).__init__()
        self.single = config.single
        self.num_layers = config.num_layers
        self.hid_dim = config.hid_dim
        self.out_dim = config.out_dim
        self.dropout = config.dropout
        # words embedding
        if w_embedding is None:
            self.num_words = config.num_words
            self.w_embedding_dim = config.w_embedding_dim
            self.w_embedding = nn.Embedding(self.num_words,self.w_embedding_dim)
        else:
            self.num_words,self.w_embedding_dim = w_embedding.shape
            self.w_embedding = nn.Embedding.from_pretrained(w_embedding)
        self.w_mask_embedding = nn.Embedding(2,self.w_embedding_dim)
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
        if self.single:
            self.class_size = config.class_size
            self.pred = nn.Linear(self.hid_dim,self.class_size)
        else:
            self.class_size = config.class_size
            self.label_size = config.label_size
            self.pred = MatchSimpleNet(self.hid_dim,self.label_size,self.class_size)
    def get_features(self,words2ids,i_mask,entropy_mat):
        # words embedding and graph encoding
        w_emb = self.w_embedding(words2ids)
        w_mask_emb = self.w_mask_embedding(i_mask)
        gw_hid = w_emb*torch.sigmoid(w_mask_emb)
        #A_hid = entropy_mat
        for k in range(self.num_layers):
            A_hid = torch.relu(self.vae_list[k](gw_hid,entropy_mat))
            gw_hid = self.graph_list[k](gw_hid,A_hid)
        return gw_hid
    def forward(self,words2ids,i_mask,entropy_mat,**kwargs):
        """
        'words2ids', 'i_mask', 'content_chars', 'c_mask', 'content_words', 'w_mask', 'entropy_mat', 'paris_mat'
        """
        dcaps_hid = self.get_features(words2ids,i_mask,entropy_mat)
        # output layer
        if self.single:
            output = self.pred(dcaps_hid.sum(dim=1))
            logits = F.log_softmax(output,dim=1)
        else:
            logits = self.pred(dcaps_hid)
        return logits



