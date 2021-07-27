import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


from mtl_datasets import DatasetFactory
from custom_tokenizers.muppet_tokenizer import RobertaMuppetTokenizer

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
def LoadDataset(args, task_args,ids,split="trainval"):

    tokenizer = RobertaMuppetTokenizer.from_pretrained(args["model_name_or_path"], use_fast=True)


    task_datasets = {}
    task_datasets_loss = {}
    task_dataloader = {}
    task_ids = []

    for i, task_id in enumerate(ids):
        task = "TASK" + str(task_id)
        task_name = task_args[task]["name"]
        task_ids.append(task)
        dataroot = task_args[task]["dataroot"]
        max_seq_length = task_args[task]['max_seq_length']
        split = task_args[task]['train_split']
        loss = task_args[task]['loss']
        task_datasets_loss[task] = loss
        task_datasets[task] = DatasetFactory[task_name](task_name, dataroot, split, max_seq_length, tokenizer)
        task_dataloader[task] = task_datasets[task]()
        
    return (task_datasets, 
        task_dataloader,
        task_datasets_loss, 
        task_ids
    )

def main(args,task_args, ids):
    task_datasets, task_dataloader, task_datasets_loss, task_ids = LoadDataset(args, task_args, ids)
    print("need_random")

if __name__ == '__main__':
    import yaml
    import os
    from os.path import dirname, abspath
    print(dirname(dirname(abspath(__file__))))
    with open(dirname(dirname(abspath(__file__))) + '/multi_tasks.yml') as f:
        task_args = yaml.load(f, Loader=yaml.FullLoader)
        print(task_args['TASK1'])
    ids = [1,2,3]
    args = {"model_name_or_path" : "roberta-base"}
    main(args, task_args, ids)