import os
import json
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir",default="./result",type=str)
    args = parser.parse_args()
    
    eng_dataset_dict = {"N15News"}
    
    heads_list = ["dataset","texts","class","avg.length","max.length","words","train/dev"]
    statistics_data = []
    for dataset in os.listdir(args.result_dir):
        tp_data_dir = os.path.join(args.result_dir,dataset)
        if not os.path.isdir(tp_data_dir):
            continue
        avg_length = 0
        max_length = 0
        data_dict = {}
        tags_name_dict = {
            "train":0,
            "dev":0
        }
        class_set = set()
        for tag_name in tags_name_dict:
            load_filename = os.path.join(args.result_dir,dataset,"%sset.json"%tag_name)
            with open(load_filename,mode="r",encoding="utf-8") as rfp:
                for line in rfp:
                    tp_dict = json.loads(line)
                    if dataset in eng_dataset_dict:
                        words_list = tp_dict["content"].split()
                    else:
                        words_list = list(tp_dict["content"])
                    class_set.add(tp_dict["label"])
                    avg_length += len(words_list)
                    max_length = max(max_length,len(words_list))
                    for word in words_list:
                        if word not in data_dict:
                            data_dict[word] = 1
                        else:
                            data_dict[word] = len(data_dict)
                    tags_name_dict[tag_name] += 1
        all_lens = sum([tags_name_dict[tag_name] for tag_name in tags_name_dict])
        tag_str = "".join(["%0.1f/"%(tags_name_dict[tag_name]/all_lens*100) for tag_name in tags_name_dict])[:-1]
        avg_length = round(avg_length/all_lens,2)
        statistics_data.append([dataset,all_lens,len(class_set),avg_length,max_length,len(data_dict),tag_str])
    save_filename = os.path.join(args.result_dir,"statistics.xlsx")
    pd.DataFrame(statistics_data).to_excel(save_filename,index=None,header=heads_list)
if __name__ == "__main__":
    main()

