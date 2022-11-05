import torch 
import torch.nn as nn
import torch.nn.functional as F

from ..layers import GraphConvolution
from ..layers import MatchSimpleNet
from ..layers import SFU,MutiAttentionLayer

from transformers import XLNetModel,XLNetTokenizer
from transformers import BertModel,BertTokenizer
from transformers import RobertaModel,RobertaTokenizer
from transformers import AlbertModel,AlbertTokenizer

class TextGCN(nn.Module):
    def __init__(self,config,w_embedding=None,**kwargs):
        super(TextGCN,self).__init__()
        self.single = config.single
        self.nhid_dim = config.nhid_dim
        self.ohid_dim = config.ohid_dim
        if w_embedding is None:
            self.num_words = config.num_words
            self.embedding_dim = config.embedding_dim
            self.w_embedding = nn.Embedding(self.num_words,self.embedding_dim)
        else:
            self.num_words,self.embedding_dim = w_embedding.shape
            self.w_embedding = nn.Embedding.from_pretrained(w_embedding)
        self.mask_embedding = nn.Embedding(2,self.embedding_dim)
        self.gc1 = GraphConvolution(self.embedding_dim, self.nhid_dim)
        self.gc2 = GraphConvolution(self.nhid_dim, self.ohid_dim)
        if self.single:
            self.class_size = config.class_size
            self.pred = nn.Linear(self.ohid_dim,self.class_size)
        else:
            self.class_size = config.class_size
            self.label_size = config.label_size
            self.pred = MatchSimpleNet(self.ohid_dim,self.label_size,self.class_size)
        self.gelu = nn.GELU()
    def forward(self,words2ids,i_mask,paris_mat,**kwargs):
        """
        'words2ids', 'i_mask', 'content_chars', 'c_mask', 'content_words', 'w_mask', 'entropy_mat', 'paris_mat'
        """
        w_emb = self.w_embedding(words2ids)
        w_mask = self.mask_embedding(i_mask)
        w_emb = w_emb*torch.sigmoid(w_mask)
        w_hid = self.gc1(w_emb,paris_mat)
        w_hid = self.gelu(w_hid)
        w_hid = self.gc2(w_hid,paris_mat)
        if self.single:
            logits = self.pred(w_hid.sum(dim=1))
            logits = F.log_softmax(logits,dim=1)
        else:
            logits = self.pred(w_hid)
        return logits
    
class PretrainGCNModel(nn.Module):
    def __init__(self,config,w_embedding=None,**kwargs):
        super(PretrainGCNModel,self).__init__()
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
        self.dropout = config.dropout
        self.single = config.single
        self.single = config.single
        self.nhid_dim = config.nhid_dim
        self.ohid_dim = config.ohid_dim
        if w_embedding is None:
            self.num_words = config.num_words
            self.embedding_dim = config.embedding_dim
            self.w_embedding = nn.Embedding(self.num_words,self.embedding_dim)
        else:
            self.num_words,self.embedding_dim = w_embedding.shape
            self.w_embedding = nn.Embedding.from_pretrained(w_embedding)
        self.mask_embedding = nn.Embedding(2,self.embedding_dim)
        self.gc1 = GraphConvolution(self.embedding_dim, self.nhid_dim)
        self.gc2 = GraphConvolution(self.nhid_dim, self.ohid_dim)
        # funsion layer
        self.att = MutiAttentionLayer(self.ohid_dim,self.ohid_dim,self.ohid_dim)
        self.sfu = SFU(self.ohid_dim,self.ohid_dim)
        self.encode = nn.Sequential(
            nn.Linear(self.pretrain.config.hidden_size,self.ohid_dim),
            nn.GELU()
        )
        if self.single:
            self.class_size = config.class_size
            self.pred = nn.Linear(self.ohid_dim,self.class_size)
        else:
            self.class_size = config.class_size
            self.label_size = config.label_size
            self.pred = MatchSimpleNet(self.ohid_dim,self.label_size,self.class_size)
        self.gelu = nn.GELU()
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
        pooled_out = self.pretrain(**outputs)
        dc_hid = pooled_out['last_hidden_state']
        dc_hid = self.encode(dc_hid)
        # words embedding and graph encoding
        w_emb = self.w_embedding(words2ids)
        w_mask_emb = self.w_mask_embedding(i_mask)
        gw_hid = w_emb*torch.sigmoid(w_mask_emb)
        
        w_hid = self.gc1(gw_hid,paris_mat)
        w_hid = self.gelu(w_hid)
        w_hid = self.gc2(w_hid,paris_mat)
        # fusion layer
        datt_hid,datts = self.att(dc_hid,w_hid)
        dsfu_hid = self.sfu(datt_hid,dc_hid)
        return dsfu_hid,datts


