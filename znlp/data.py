import os
import copy
import math
import json
import time
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def to_var(object_value,device):
    if type(object_value) is dict:
        for key in object_value:
            object_value[key] = to_var(object_value[key],device)
    elif type(object_value) is list:
        for idx in range(len(object_value)):
            object_value[idx] = to_var(object_value[idx],device)
    elif type(object_value) is tuple:
        re_list = []
        for idx in range(len(object_value)):
            re_list.append(to_var(object_value[idx],device))
        object_value = re_list
    elif type(object_value) is str:
        return object_value
    else:
        object_value = torch.from_numpy(object_value).to(device)
    return object_value
class Dictionary:
    def __init__(self):
        self.name = 'default'
        self.ind2token = ['<PAD>','<START>','<END>','<UNK>',]
        self.token2ind = {'<PAD>':0,'<START>':1,'<END>':2,'<UNK>':3}
        self.start_index = 0
        self.end_index = len(self.ind2token)
    def __iter__(self):
        return self
    def __next__(self):
        if self.start_index < self.end_index:
            ret = self.ind2token[self.start_index]
            self.start_index += 1
            return ret
        else:
            raise StopIteration
    def __getitem__(self,item):
        if type(item) == str:
            return self.token2ind.get(item,3)
        elif type(item) == int:
            word = self.ind2token[item]
            return word
        else:
            raise IndexError()
    def add(self,word):
        if word not in self.token2ind:
            self.token2ind[word] = len(self.ind2token)
            self.ind2token.append(word)
            self.end_index = len(self.ind2token)
    def save(self,save_file):
        with open(save_file,"w",encoding="utf-8") as wfp:
            data = {
                "ind2token":self.ind2token,
                "token2ind":self.token2ind,
            }
            json.dump(data,wfp,ensure_ascii=False)
    @staticmethod
    def load(load_file):
        tp_dict = Dictionary()
        with open(load_file,"r",encoding="utf-8") as rfp:
            data = json.load(rfp)
            tp_dict.token2ind = data["token2ind"]
            tp_dict.ind2token = data["ind2token"]
            tp_dict.end_index = len(tp_dict.ind2token)
        return tp_dict
    def __contains__(self,word):
        assert type(word) == str
        return word in self.token2ind
    def __len__(self):
        return len(self.token2ind)
    def __repr__(self):
        return '{}(num_keys={})'.format(
            self.__class__.__name__,len(self.token2ind))
    def __str__(self):
        return '{}(num_keys={})'.format(
            self.__class__.__name__,len(self.token2ind))
class ContentReviewDataset(Dataset):
    def __init__(self,load_file_name,words_dict,chars_dict,max_limit_len=512,tokenizer=None):
        super(ContentReviewDataset,self).__init__()
        self.tokenizer = tokenizer
        with open(load_file_name,mode="rb") as rfp:
            self.dataset = pickle.load(rfp)
        self.words_dict = words_dict
        self.chars_dict = chars_dict
        self.max_limit_len = max_limit_len
    def __getitem__(self,idx):
        re_val = copy.copy(self.dataset[idx])
        content_chars = [self.chars_dict[w] for w in list(self.dataset[idx]['content'].replace(" ",""))]
        re_val['content_chars'] = content_chars
        content_words = [self.words_dict[w] for w in list(self.dataset[idx]['content'].split())]
        re_val['content_words'] = content_words
        id2words = [self.words_dict[w] for w in self.dataset[idx]['id2words']]
        re_val['id2words'] = id2words
        re_val['content'] = self.dataset[idx]['content'].replace(" ","")
        return re_val
    def __len__(self,):
        return len(self.dataset)
def batchfy(batch):
    """_summary_
    Args:
        batch : batchfy the dataset
    Returns:
        dict: 
        id2words,i_mask
        content_chars,c_mask
        content_words,w_mask
        entropy_mat,paris_mat
        list:
        labels_list
    """
    re_dict = {}
    index_list = [item['index'] for item in batch]
    # key words
    max_words_len =  max([len(item['id2words']) for item in batch])
    id2words_list = [item['id2words']+(max_words_len-len(item['id2words']))*[0] for item in batch]
    id2words_list = np.array(id2words_list,dtype=np.int64)
    i_mask_list = [len(item['id2words'])*[1]+(max_words_len-len(item['id2words']))*[0] for item in batch]
    i_mask_list = np.array(i_mask_list,dtype=np.int64)
    re_dict["words2ids"] = id2words_list
    re_dict["i_mask"] = i_mask_list
    # content chars
    max_content_c_len = max([len(item['content_chars']) for item in batch])
    chars_list = [item['content_chars']+(max_content_c_len-len(item['content_chars']))*[0] for item in batch]
    chars_list = np.array(chars_list,dtype=np.int64)
    c_mask_list = [len(item['content_chars'])*[1]+(max_content_c_len-len(item['content_chars']))*[0] for item in batch]
    c_mask_list = np.array(c_mask_list,dtype=np.int64)
    re_dict["content_chars"] = chars_list
    re_dict["c_mask"] = c_mask_list
    # content words
    max_content_w_len = max([len(item['content_words']) for item in batch])
    words_list = [item['content_words']+(max_content_w_len-len(item['content_words']))*[0] for item in batch]
    words_list = np.array(words_list,dtype=np.int64)
    w_mask_list = [len(item['content_words'])*[1]+(max_content_w_len-len(item['content_words']))*[0] for item in batch]
    w_mask_list = np.array(w_mask_list,dtype=np.int64)
    re_dict["content_words"] = words_list
    re_dict["w_mask"] = w_mask_list
    # entropy mat
    entropy_mat = [np.pad(item['entropy_mat'],[(0,max_words_len-item['entropy_mat'].shape[0]),(0,max_words_len-item['entropy_mat'].shape[1])]) if item['entropy_mat'].shape[0] != 0 else np.zeros(shape=(max_words_len,max_words_len)) for item in batch]
    entropy_mat = np.array(entropy_mat,np.float32)
    re_dict["entropy_mat"] = entropy_mat
    # paris mat
    paris_mat = [np.pad(item['paris_mat'],[(0,max_words_len-item['paris_mat'].shape[0]),(0,max_words_len-item['paris_mat'].shape[1])]) if item['paris_mat'].shape[0] != 0 else np.zeros(shape=(max_words_len,max_words_len)) for item in batch]
    paris_mat = np.array(paris_mat,np.float32)
    re_dict["paris_mat"] = paris_mat
    # content
    content_list = [item['content'] for item in batch]
    re_dict["content"] = content_list
    # label 
    labels_list = [item['label'] for item in batch]
    labels_list = np.array(labels_list,np.int64)
    return re_dict,labels_list
    
def create_paris_graph(content,id2words,window_size):
    doc_word_id_map = {}
    for j in range(len(id2words)):
        doc_word_id_map[id2words[j]] = j
    # sliding windows
    windows = []
    doc_len = len(content)
    if doc_len <= window_size:
        windows.append(content)
    else:
        for j in range(doc_len - window_size + 1):
            window = content[j: j + window_size]
            windows.append(window)

    word_pair_count = {}
    for window in windows:
        for p in range(1, len(window)):
            for q in range(0, p):
                word_p = window[p]
                word_q = window[q]
                if word_p == word_q:
                    continue
                word_pair_key = (word_p, word_q)
                # word co-occurrences as weights
                if word_pair_key in word_pair_count:
                    word_pair_count[word_pair_key] += 1.
                else:
                    word_pair_count[word_pair_key] = 1.
                # bi-direction
                word_pair_key = (word_q, word_p)
                if word_pair_key in word_pair_count:
                    word_pair_count[word_pair_key] += 1.
                else:
                    word_pair_count[word_pair_key] = 1.
    weight_len = len(id2words)
    weight = np.zeros(shape=(weight_len,weight_len),dtype=np.float64)

    for key in word_pair_count:
        word_p = key[0]
        word_q = key[1]
        if word_q not in doc_word_id_map or word_p not in doc_word_id_map:
            continue
        weight[doc_word_id_map[word_p],doc_word_id_map[word_q]] = word_pair_count[key]
    adj_mat = np.array(weight,dtype=np.float64)
    return adj_mat


def create_entropy_graph(content,id2words,window):
    words2freq = {}
    for word in content:
        if word in id2words:
            if word in words2freq:
                words2freq[word] += 1.0
            else:
                words2freq[word] = 1.0
    words2freq = {w:words2freq[w]/len(words2freq) for w in words2freq}
    win_list = []
    if window>=len(content):
        win_list.append(content)
    else:
        for idx in range(len(content) - window + 1):
            win_list.append(content[idx:idx+window])
    for win in win_list:
        w2f = {}
        for w in win:
            if w in id2words:
                if w not in w2f:
                    w2f[w] = 1.0
                else:
                    w2f[w] += 1.0    
        words2freq = {w:words2freq[w] + w2f[w]/len(win) if w in w2f else words2freq[w] for w in words2freq}
    mat = []
    for pw in id2words:
        tmp_list = []
        for qw in id2words:
            tmp_list.append(words2freq[pw]*math.log(1.0+words2freq[pw]/words2freq[qw]))
        mat.append(tmp_list)
    mat = np.array(mat,dtype=np.float64)
    return mat

class Timer(object):
    """Computes elapsed time."""
    def __init__(self):
        self.running = True
        self.total = 0
        self.start = time.time()
    def reset(self):
        self.running = True
        self.total = 0
        self.start = time.time()
        return self
    def resume(self):
        if not self.running:
            self.running = True
            self.start = time.time()
        return self
    def stop(self):
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self
    def time(self):
        if self.running:
            return self.total + time.time() - self.start
        return self.total
class DataSaver(object):
    """save every epoch datas."""
    def __init__(self,save_file_name):
        names = ["f1_score","em_score","loss","time"]
        self.value_list = pd.DataFrame(columns=names)
        self.save_file_name = save_file_name
    def add_values(self,f1_score,em_score,loss,time):
        data = {"f1_score":f1_score,
                "em_score":em_score,
                "loss":loss,
                "time":time}
        idx = len(self.value_list)
        self.value_list.loc[idx] = data
        self.value_list.to_csv(self.save_file_name,index=None)



