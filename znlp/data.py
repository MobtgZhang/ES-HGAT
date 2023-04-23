import os
import math
import json
import pickle
import numpy as np

from tqdm import tqdm
import torch
from transformers import AutoTokenizer

def create_entropy_graph(content,window_size):
    words2freq = {}
    content = list(content)
    for word in content:
        if word in words2freq:
            words2freq[word] += 1.0
        else:
            words2freq[word] = 1.0
    words2freq = {w:words2freq[w]/len(words2freq) for w in words2freq}
    win_list = []
    if window_size>=len(content):
        win_list.append(content)
    else:
        for idx in range(len(content) - window_size + 1):
            win_list.append(content[idx:idx+window_size])
    for win in win_list:
        w2f = {}
        for w in win:
            if w not in w2f:
                w2f[w] = 1.0
            else:
                w2f[w] += 1.0    
        words2freq = {w:words2freq[w] + w2f[w]/len(win) if w in w2f else words2freq[w] for w in words2freq}
    mat = []
    for pw in content:
        tmp_list = []
        for qw in content:
            tmp_list.append(words2freq[pw]*math.log(1.0+words2freq[pw]/words2freq[qw]))
        mat.append(tmp_list)
    mat = np.array(mat,dtype=np.float32)
    return mat
def create_paris_graph(content,window_size):
    doc_word_id_map = {}
    id2words = list(set(content))
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
    weight_len = len(content)
    weight = np.zeros(shape=(weight_len,weight_len),dtype=np.float32)

    for key in word_pair_count:
        word_p = key[0]
        word_q = key[1]
        if word_q not in doc_word_id_map or word_p not in doc_word_id_map:
            continue
        weight[doc_word_id_map[word_p],doc_word_id_map[word_q]] = word_pair_count[key]
    adj_mat = np.array(weight,dtype=np.float32)
    return adj_mat

class ContentReviewDataset(torch.utils.data.Dataset):
    def __init__(self,result_dir,tag_name,tokenizer_file,window,mat_type,limits_len=256):
        super(ContentReviewDataset,self).__init__()
        self.window = window
        self.mat_type = mat_type
        self.result_dir = result_dir
        self.tokenizer_file = tokenizer_file
        self.limits_len = limits_len
        load_file = os.path.join(result_dir,"%sset.json"%tag_name)
        print(tokenizer_file)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_file,
                add_special_tokens=False, # don't add CLS,SEP
                do_lower_case=True)  # do the lower case
        save_pkl_file = os.path.join(result_dir,"%s-graph.pkl"%tag_name)
        save_label_file = os.path.join(result_dir,"names-dict.json")
        with open(save_label_file,mode="r",encoding="utf-8") as rfp:
            data_dict = json.loads(rfp.read())
            self.class_size = len(data_dict["id2label"])
        if not os.path.exists(save_pkl_file):
            self.dataset = []
            with open(load_file,mode="r",encoding="utf-8") as rfp:
                all_lines = rfp.readlines()
                self.avg_len = [len(json.loads(item)["content"]) for item in all_lines]
                self.avg_len = sum(self.avg_len)/len(self.avg_len)
                for line in tqdm(all_lines,desc="loading dataset"):
                    data_dict = json.loads(line)
                    index = data_dict["index"]
                    content = data_dict["content"]
                    label = data_dict["label"]
                    outputs = self.tokenizer(content,
                                                padding=True,
                                                truncation=True,
                                                max_length=self.limits_len,)
                    input_ids = outputs["input_ids"] + [0]*(self.limits_len-len(outputs["input_ids"]))
                    attention_mask = [1]*len(outputs["input_ids"]) + [0]*(self.limits_len-len(outputs["input_ids"]))
                    mat = create_entropy_graph(outputs["input_ids"],window_size=self.window)
                    mat = np.pad(mat,[(0,self.limits_len-mat.shape[0]),(0,self.limits_len-mat.shape[1])])
                    tmp_dict = {
                        "index":index,
                        "label":label,
                        "mat":mat,
                        "input_ids":input_ids,
                        "attention_mask":attention_mask
                    }
                    self.dataset.append(tmp_dict)
            data_details = {
                "dataset":self.dataset,
                "window":self.window,
                "avg_len":self.avg_len,
                "limits_len":self.limits_len,
                "mat_type":self.mat_type,
                "tokenizer_file":self.tokenizer_file,
                "class_size":self.class_size
            }
            with open(save_pkl_file,mode="wb") as wfp:
                pickle.dump(data_details,wfp)
        else:
            with open(save_pkl_file,mode="rb") as rfp:
                data_details = pickle.load(rfp)
            self.dataset = data_details["dataset"]
            self.window = data_details["window"]
            self.avg_len = data_details["avg_len"]
            self.limits_len = data_details["limits_len"]
            self.mat_type = data_details["mat_type"]
            self.tokenizer_file = data_details["tokenizer_file"]
            self.class_size = data_details["class_size"]
    def __getitem__(self,idx):
        return self.dataset[idx]
    def __len__(self):
        return len(self.dataset)
def batchfy(batch):
    index = [bn['index'] for bn in batch]
    targets = np.array([bn["label"] for bn in batch])
    adjmat = np.array([bn["mat"] for bn in batch],dtype=np.float32)
    input_ids = np.array([bn["input_ids"] for bn in batch],dtype=np.int64)
    attention_mask = np.array([bn["attention_mask"] for bn in batch])
    re_dict = {
        "index":index,
        "input_ids":input_ids,
        "attention_mask":attention_mask,
        "adjmat":adjmat
    }
    return re_dict,targets
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
    elif type(object_value) is np.ndarray:
        object_value = torch.from_numpy(object_value).to(device)
    else:
        pass
    return object_value
