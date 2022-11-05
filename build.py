import time
import argparse
import logging
import os


from znlp.data import Dictionary
from znlp.tools import JiebaTokenizer,JanomeTokenizer
from znlp.utils import build_embeddings
from preprocess import build_graph,build_dictionary
from csv_process import split_aichallenger2018_dataset,split_carsreviews_dataset
from csv_process import split_wrime_dataset
from csv_process import split_simplifyweibo4moods_dataset,split_clueemotion2020_dataset,split_nlpcc2018task1_dataset,split_toutiaonews_dataset
logger = logging.getLogger()


def check_args(args):
    result_dir = os.path.join("./result",args.dataset)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
def main(args):
    data_dir = os.path.join(args.data_dir,args.dataset)
    result_dir = os.path.join(args.result_dir,args.dataset)
    if args.dataset in ["AiChallenger2018","CLUEEmotion2020","TouTiaoNews","SimplifyWeibo4Moods","NLPCC2018Task1"]:
        args.lang = "zh"
        args.embedding_file = "./vec/cc.zh.300.vec"
    elif args.dataset in ["wrime-ver1","wrime-ver2"]:
        args.lang = "jp"
        args.embedding_file = "./vec/cc.ja.300.vec"
    else:
        raise ValueError("Unknow language: %s"%args.lang)
    print(args.embedding_file) 
    if args.dataset == "AiChallenger2018":
        if len(os.listdir(result_dir))==0:
            tokenizer = JiebaTokenizer()
            split_aichallenger2018_dataset(data_dir,result_dir,tokenizer,args.percentage)
    elif args.dataset == "CLUEEmotion2020":
        if len(os.listdir(result_dir))==0:
            tokenizer = JiebaTokenizer()
            split_clueemotion2020_dataset(data_dir,result_dir,tokenizer)
    elif args.dataset in ["wrime-ver1","wrime-ver2"]:
        data_dir = os.path.join(args.data_dir,"wrime")
        if len(os.listdir(result_dir))==0:
            tokenizer = JanomeTokenizer()
            split_wrime_dataset(data_dir,result_dir,tokenizer,args.dataset,args.name)
    elif args.dataset == "TouTiaoNews":
        data_dir = os.path.join(args.data_dir,"TouTiaoNews")
        if len(os.listdir(result_dir))==0:
            tokenizer = JiebaTokenizer()
            split_toutiaonews_dataset(data_dir,result_dir,tokenizer,args.percentage)
    elif args.dataset == "NLPCC2018Task1":
        data_dir = os.path.join(args.data_dir,"NLPCC2018Task1")
        if len(os.listdir(result_dir))==0:
            tokenizer = JiebaTokenizer()
            split_nlpcc2018task1_dataset(data_dir,result_dir,tokenizer)
    elif args.dataset == "SimplifyWeibo4Moods":
        if len(os.listdir(result_dir))==0:
            tokenizer = JiebaTokenizer()
            split_simplifyweibo4moods_dataset(data_dir,result_dir,tokenizer,args.percentage)
    else:
        raise ValueError("The doesn't exist %s dataset."%args.dataset)
    logger.info("The file saved in %s"%result_dir)
    
    type_list = ['train','valid','test']
    words_dict_file = os.path.join(result_dir,"words-dict.json")
    chars_dict_file = os.path.join(result_dir,"chars-dict.json")
    if not os.path.exists(words_dict_file) or not os.path.exists(chars_dict_file) :
        build_dictionary(result_dir,words_dict_file,chars_dict_file,type_list[:-1])
    words_embedding_file = os.path.join(result_dir,"words-embedding.npy")
    chars_embedding_file = os.path.join(result_dir,"chars-embedding.npy") 
    if not os.path.exists(words_embedding_file):
        data_dict = Dictionary.load(words_dict_file)
        build_embeddings(args.embedding_file,words_embedding_file,data_dict)
    if not os.path.exists(chars_embedding_file):
        chars_dict_file = Dictionary.load(chars_dict_file)
        build_embeddings(args.embedding_file,chars_embedding_file,chars_dict_file)
    stop_words_file = os.path.join(args.data_dir,'stopwords_%s.json'%args.lang)
    for dataset_type in type_list:
        save_graph_file = os.path.join(result_dir,'%s-graph.pkl'%dataset_type)
        load_dataset_file = os.path.join(result_dir,"%sset.json"%dataset_type)
        if not os.path.exists(load_dataset_file):
            continue
        if not os.path.exists(save_graph_file):
            build_graph(result_dir,stop_words_file,dataset_type,args.window) 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',default='./data',type=str)
    parser.add_argument('--result-dir',default='./result',type=str)
    parser.add_argument('--log-dir',default='./log',type=str)
    parser.add_argument('--dataset',default='CLUEEmotion2020',type=str)
    parser.add_argument('--percentage',default=0.7,type=float)
    parser.add_argument('--name',default="Writer",type=str)
    parser.add_argument('--embedding-file',default=None,type=str)
    parser.add_argument('--window',default=4,type=int)
    args = parser.parse_args()
    check_args(args)
    # first, create a project logger
    logger.setLevel(logging.INFO)  # The main log level switch.
    # second, create a handler,which is used to write in files
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    logfile = os.path.join(args.log_dir,rq + '.log')
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)  # the switch to print log file
    # create a streamhandler to print the log in terminal, level is above error.
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # third, define the format of the handler.
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    # fourth, add the logger in handler.
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(str(args))
    main(args)


