import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
import json
import os

from .Datasets import DatasetFactory

categories = ["classification", "commonsense", "mrc"]
train_set = "train.csv"
dev_set = "dev.csv"
test_set = "test.csv"
info_file = "info.json"
dataset_path = "./"


# dataset Load needs
# task_names
# cateogries : classification, commonsense, mrc
# files structure : train.csv, dev.csv, test.csv with info.json
# json file has ... choices, type, columns
def LoadDataset(args,task_args,ids,split="trainval"):

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    task_datasets_train = {}
    task_datasets_val = {}
    task_dataloader_train = {}
    task_dataloader_val = {}
    task_ids = []
    task_batch_size = {}
    task_num_iters = {}

    for i, task_id in enumerate(ids):
        task = "TASK" + task_id
        task_name = task_args[task]["name"]
        task_ids.append(task)
        dataroot = task_args[task]["dataroot"]
        task_datasets_train[task] = None
            




def dataset_loader(path, categories, name):
    datasets = load_dataset('csv', data_files={'train': './commonsenseqa/train_set.csv','test': './commonsenseqa/test_set.csv','validation': './commonsenseqa/dev_set.csv'})

def main():
    dataset_names = ["hello word"]

if __name__ == '__main__':
    import yaml
    import os
    from os.path import dirname, abspath
    print(dirname(dirname(abspath(__file__))))
    with open(dirname(dirname(abspath(__file__))) + '/multi_tasks.yml') as f:
        task_args = yaml.load(f, Loader=yaml.FullLoader)
        print(task_args['TASK1'])
    ids = [1,2,3]
    main(task_args)