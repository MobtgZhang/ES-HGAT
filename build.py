import os
import time
import argparse
import logging
logger = logging.getLogger()

from preprocess import split_dataset

def check_args(args):
    result_dir = os.path.join("./result",args.dataset)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    assert args.dataset in os.listdir(args.data_dir)
def main(args):
    result_dir = os.path.join(args.result_dir,args.dataset)
    if len(os.listdir(result_dir))==0:
        split_dataset(args)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',default='./data',type=str)
    parser.add_argument('--result-dir',default='./result',type=str)
    parser.add_argument('--log-dir',default='./log',type=str)
    parser.add_argument('--dataset',default='CLUEEmotion2020',type=str)
    parser.add_argument('--percentage',default=0.7,type=float)
    parser.add_argument('--embedding-file',default=None,type=str)
    parser.add_argument('--window',default=4,type=int)
    parser.add_argument('--version',default="3.0",type=str)
    parser.add_argument('--mat-type',default="entropy",type=str)
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
    
