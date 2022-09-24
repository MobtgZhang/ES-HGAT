import json
import math
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

from znlp.data import Dictionary
from znlp.data import create_entropy_graph,create_paris_graph

def build_dictionary(result_dir,save_words_dict_file,save_chars_dict_file,data_type_list,data_seg=False):
    words_tmp_dict = Dictionary()
    if not data_seg:
        chars_tmp_dict = Dictionary()
    for data_type in data_type_list:
        load_dataset_file = os.path.join(result_dir,"%sset.json"%data_type)
        with open(load_dataset_file,mode="r",encoding="utf-8") as rfp:
            for line in rfp:
                data_dict = json.loads(line)
                content = data_dict['content']
                for word in content.split():
                    words_tmp_dict.add(word)
                if not data_seg:
                    content_line = list("".join(data_dict['content']))
                    for char in content_line:
                        chars_tmp_dict.add(char)
    words_tmp_dict.save(save_words_dict_file)
    if not data_seg:
        chars_tmp_dict.save(save_chars_dict_file)
def build_graph(result_dir,stop_words_file,dataset_type,window):
    with open(stop_words_file,mode='r',encoding='utf-8') as rfp:
        stop_words_dict = json.load(rfp)
    load_dataset_file = os.path.join(result_dir,"%sset.json"%dataset_type)
    save_graph_file = os.path.join(result_dir,"%s-graph.pkl"%dataset_type)
    all_datas = []
    with open(load_dataset_file,mode="r",encoding="utf-8")as rfp:
        for line in tqdm(rfp,desc="build %s dataset"%dataset_type):
            data_dict = json.loads(line)
            content = data_dict['content'].split()
            id2words = list(set([word for word in content if word not in stop_words_dict]))
            entropy_mat = create_entropy_graph(content,id2words,window)
            paris_mat = create_paris_graph(content,id2words,window)
            save_dict = {
                "index":data_dict['index'],
                "content":data_dict['content'],
                "id2words":id2words,
                "entropy_mat":entropy_mat,
                "paris_mat":paris_mat,
                "label":data_dict['labels']
            }
            all_datas.append(save_dict)
        with open(save_graph_file,mode="wb")as wfp:
            pickle.dump(all_datas,wfp)

