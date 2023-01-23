import torch
import torch.nn as nn
import torch.nn.functional as F


from transformers import XLNetModel,XLNetTokenizer
from transformers import BertModel,BertTokenizer
from transformers import RobertaModel,RobertaTokenizer
from transformers import AlbertModel,AlbertTokenizer


from ..layers import MultiCNNLayer,MatchSimpleNet

class TextCNNWords(nn.Module):
    def __init__(self,config,w_embedding=None,**kwargs):
        super(TextCNNWords,self).__init__()
        self.n_filters = config.n_filters
        self.filter_sizes = config.filter_sizes
        self.out_dim = config.out_dim
        self.dropout = config.dropout
        if w_embedding is None:
            self.num_words = config.num_words
            self.embedding_dim = config.embedding_dim
            self.w_embedding = nn.Embedding(self.num_words,self.embedding_dim)
        else:
            self.num_words,self.embedding_dim = w_embedding.shape
            self.w_embedding = nn.Embedding.from_pretrained(w_embedding)
        self.mask_embedding = nn.Embedding(2,self.embedding_dim)
        self.cnn = MultiCNNLayer(self.embedding_dim,self.n_filters,self.filter_sizes,self.out_dim,self.dropout)
        self.class_size = config.class_size
        self.pred = nn.Linear(self.out_dim,self.class_size)
    def forward(self,content_words,w_mask,**kwargs):
        """
        'words2ids', 'i_mask', 'content_chars', 'c_mask', 'content_words', 'w_mask', 'entropy_mat', 'paris_mat'
        """
        w_emb = self.w_embedding(content_words)
        w_mask = self.mask_embedding(w_mask)
        w_hid = w_emb*torch.sigmoid(w_mask)
        w_hid = self.cnn(w_hid)
        logits = self.pred(w_hid)
        return logits

class TextCNNChars(nn.Module):
    def __init__(self,config,c_embedding=None,**kwargs):
        super(TextCNNChars,self).__init__()
        self.single = config.single
        self.n_filters = config.n_filters
        self.filter_sizes = config.filter_sizes
        self.out_dim = config.out_dim
        self.dropout = config.dropout
        if c_embedding is None:
            self.num_chars = config.num_chars
            self.embedding_dim = config.embedding_dim
            self.c_embedding = nn.Embedding(self.num_chars,self.embedding_dim)
        else:
            self.num_chars,self.embedding_dim = c_embedding.shape
            self.c_embedding = nn.Embedding.from_pretrained(c_embedding)
        self.mask_embedding = nn.Embedding(2,self.embedding_dim)
        self.cnn = MultiCNNLayer(self.embedding_dim,self.n_filters,self.filter_sizes,self.out_dim,self.dropout)
        self.class_size = config.class_size
        self.pred = nn.Linear(self.out_dim,self.class_size)
    def forward(self,content_chars,c_mask,**kwargs):
        """
        'words2ids', 'i_mask', 'content_chars', 'c_mask', 'content_words', 'w_mask', 'entropy_mat', 'paris_mat'
        """
        c_emb = self.c_embedding(content_chars)
        c_mask = self.mask_embedding(c_mask)
        c_hid = c_emb*torch.sigmoid(c_mask)
        c_hid = self.cnn(c_hid)
        logits = self.pred(c_hid.sum(dim=1))
        logits = F.log_softmax(logits,dim=1)
        return logits

class PretrainCNNModel(nn.Module):
    def __init__(self,config,**kwargs):
        super(PretrainCNNModel,self).__init__()
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
        
        self.n_filters = config.n_filters
        self.filter_sizes = config.filter_sizes
        self.out_dim = config.out_dim
        self.dropout = config.dropout
        self.hid_dim = config.hid_dim
        self.encode = nn.Sequential(
            nn.Linear(self.pretrain.config.hidden_size,self.hid_dim),
            nn.GELU()
        )
        self.cnn = MultiCNNLayer(self.hid_dim,self.n_filters,self.filter_sizes,self.out_dim,self.dropout)
        self.class_size = config.class_size
        self.pred = nn.Linear(self.out_dim,self.class_size)
    def forward(self,content,**kwargs):
        # chars embedding and encoding
        device = next(self.parameters()).device
        outputs = self.tokenizer(content,return_tensors="pt",padding=True,truncation=True)
        outputs = outputs.to(device)
        pooled_out = self.pretrain(**outputs)
        dc_hid = pooled_out['last_hidden_state']
        dc_hid = self.encode(dc_hid)
        dc_hid = self.cnn(dc_hid)
        # output layer
        output = self.pred(dc_hid.sum(dim=1))
        logits = F.log_softmax(output,dim=-1)
        return logits


