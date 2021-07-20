import datasets
from datasets import load_dataset
import json

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
def dataset_load_and_merge(task_names, categories, name):
    multiple_datasets = {}
    for category in categories:
        

def dataset_loader(path, categories, name):
    datasets = load_dataset('csv', data_files={'train': './commonsenseqa/train_set.csv','test': './commonsenseqa/test_set.csv','validation': './commonsenseqa/dev_set.csv'})

def main():
    dataset_names = ["hello word"]

if __name__ == '__main__':
    main()