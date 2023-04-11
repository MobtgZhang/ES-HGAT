import os
import csv
import json
import random
import uuid
import pandas as pd
from tqdm import tqdm

def create_clueemotion2022(data_dir,result_dir):
    raw_labels = ['train','valid','test']
    new_labels = ['train','dev','test']
    id2label = set()
    save_dict_file = os.path.join(result_dir,"names-dict.json")
    for r_key,n_key in zip(raw_labels,new_labels):
        all_dataset = []
        load_txt_file = os.path.join(data_dir,"%s.txt"%r_key)
        save_json_file = os.path.join(result_dir,"%sset.json"%n_key)
        with open(load_txt_file,mode="r",encoding="utf-8") as rfp:
            for line in tqdm(rfp.readlines(),desc="%s dataset process"%r_key):
                data_dict = json.loads(line)
                save_data_dict = {
                    "index":data_dict['id'],
                    "content":data_dict['content'],
                    "label":data_dict['label']
                }
                if r_key == raw_labels[0]:
                    id2label.add(data_dict['label'])
                all_dataset.append(save_data_dict)
        if r_key == raw_labels[0]:
            id2label = list(id2label)
            label2id = {w:idx for idx,w in enumerate(id2label)}
            save_data_dict = {
                "id2label":id2label,
                "label2id":label2id
            }
            with open(save_dict_file,mode="w",encoding="utf-8") as wfp:
                jline = json.dumps(save_data_dict,ensure_ascii=False)
                wfp.write(jline)
        with open(save_json_file,mode="w",encoding="utf-8") as wfp:
            for item_dict in all_dataset:
                item_dict["label"] = label2id[item_dict["label"]]
                jline = json.dumps(item_dict,ensure_ascii=False)
                wfp.write(jline+"\n")
def create_industry(data_dir,result_dir,version,percentage=0.7):
    # labels
    label2id = {}
    load_labels_file = os.path.join(data_dir,'label-v%s.txt'%version)
    with open(load_labels_file,mode="r",encoding="utf-8") as rfp:
        for line in rfp:
            name,idx = line.split()
            label2id[name] = int(idx)
    id2label = [None]*len(label2id)
    for key  in label2id:id2label[label2id[key]] = key
    save_data_dict = {
        "id2label":id2label,
        "label2id":label2id
    }
    save_dict_file = os.path.join(result_dir,"names-dict.json")
    with open(save_dict_file,mode="w",encoding="utf-8") as wfp:
        jline = json.dumps(save_data_dict,ensure_ascii=False)
        wfp.write(jline)
    # dataset
    load_file = os.path.join(data_dir,'data-v%s.csv'%version)
    all_dataset = []
    with open(load_file,mode="r",encoding="utf-8") as rfp:
        rfp.readline()
        for row in tqdm(csv.reader(rfp),desc="loading file: %s"%load_file):
            index,cls_idx,content =  row
            tmp_dict = {
                "index":index,
                "content":content,
                "label":int(cls_idx)
            }
            all_dataset.append(tmp_dict)
    random.shuffle(all_dataset)
    train_len = int(len(all_dataset)*percentage)
    len_list =[all_dataset[:train_len],all_dataset[train_len:]]
    new_list = ['train','dev']
    for tp_dataset,tp_name in zip(len_list,new_list):
        save_file = os.path.join(result_dir,'%sset.json'%tp_name)
        with open(save_file,mode="w",encoding="utf-8") as wfp:
            for item in tp_dataset:
                jline = json.dumps(item,ensure_ascii=False)
                wfp.write(jline+"\n")
def create_chipCTC(data_dir,result_dir):
    raw_labels = ['train','dev']
    load_category_file = os.path.join(data_dir,"category.xlsx")
    raw_label = pd.read_excel(load_category_file)
    id2label = [name.strip() for name in raw_label["Label Name"].values]
    label2id = {name:str(idx) for idx,name in enumerate(id2label)}
    save_data_dict = {
        "id2label":id2label,
        "label2id":label2id
    }
    save_dict_file = os.path.join(result_dir,"names-dict.json")
    with open(save_dict_file,mode="w",encoding="utf-8") as wfp:
        jline = json.dumps(save_data_dict,ensure_ascii=False)
        wfp.write(jline)
    for tag_name in raw_labels:
        load_file_name = os.path.join(data_dir,"CHIP-CTC_%s.json"%tag_name)
        with open(load_file_name,mode="r",encoding="utf-8") as rfp:
            all_data_list = json.load(rfp)
            save_file_name = os.path.join(result_dir,"%sset.json"%tag_name)
            with open(save_file_name,mode="w",encoding="utf-8") as wfp:
                for item in tqdm(all_data_list,desc="saving file %s"%save_file_name):
                    tmp_dict = {
                        "index":item["id"],
                        "content":item["text"],
                        "label":label2id[item["label"].strip()]
                    }
                    jline = json.dumps(tmp_dict,ensure_ascii=False)
                    wfp.write(jline+"\n")
def create_N15News(data_dir,result_dir,percentage=0.7):
    load_filename = os.path.join(data_dir,"nytimes_dataset.json")
    all_processed_list = []
    with open(load_filename,mode="r",encoding="utf-8") as rfp:
        all_data_list = json.load(rfp)
        for item in all_data_list:
            tmp_dict = {
                "index":str(uuid.uuid1()).upper(),
                "content":item["caption"],
                "label":item["section"]
            }
            all_processed_list.append(tmp_dict)
    id2label = set()
    for item in all_processed_list:id2label.add(item["label"].strip())
    id2label = list(id2label)
    label2id = {name:str(idx) for idx,name in enumerate(id2label)}
    save_dict_file = os.path.join(result_dir,"names-dict.json")
    save_data_dict = {
        "id2label":id2label,
        "label2id":label2id
    }
    with open(save_dict_file,mode="w",encoding="utf-8") as wfp:
        jline = json.dumps(save_data_dict,ensure_ascii=False)
        wfp.write(jline)
    random.shuffle(all_processed_list)
    train_len = int(len(all_processed_list)*percentage)
    data_list = [all_processed_list[:train_len],all_processed_list[train_len:]]
    new_list = ['train','dev']
    for tp_dataset,tp_name in zip(data_list,new_list):
        save_file = os.path.join(result_dir,'%sset.json'%tp_name)
        with open(save_file,mode="w",encoding="utf-8") as wfp:
            for item in tp_dataset:
                item["label"] = label2id[item["label"]]
                jline = json.dumps(item,ensure_ascii=False)
                wfp.write(jline+"\n")
def split_dataset(args):
    result_dir = os.path.join(args.result_dir,args.dataset)
    data_dir = os.path.join(args.data_dir,args.dataset)    
    if args.dataset == "CLUEEmotion2020":
        create_clueemotion2022(data_dir,result_dir)
    elif args.dataset == "CHIP-CTC":
        create_chipCTC(data_dir,result_dir)
    elif args.dataset == "IndustryData":
        create_industry(data_dir,result_dir,version=args.version)
    elif args.dataset == "N15News":
        create_N15News(data_dir,result_dir)
    else:
        raise ValueError("Unknown dataset %s"%args.dataset)

