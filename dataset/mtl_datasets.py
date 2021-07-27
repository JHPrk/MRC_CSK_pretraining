from datasets import load_dataset, load_metric, ClassLabel
import random
import pandas as pd
import json
from transformers import AutoTokenizer
import os

train_set = "train.csv"
dev_set = "dev.csv"
info_file = "info.json"
dir_path = os.path.dirname(os.path.abspath(__file__))

cls_task_token = "madeupword0002"
com_token="madeupword0001"
mrc_token="madeupword0000"


class Dataset :
    def __init__(self, task, dataroot, split, max_seq_length, tokenizer):
        path = dir_path + "/" + dataroot + '/'
        self.datasets = load_dataset('csv', data_files={'train': path + train_set, 'validation' : path + dev_set})
        self.configs = None
        with open(path + info_file) as json_file:
            self.configs = json.load(json_file)
        self.task = task
        self.split = split
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer

    def __call__(self):
        self.datasets = self.datasets.map(self.preprocess_function, batched=True)
        return self.datasets

class CoLADataset(Dataset) :
    def __init__(self, task, dataroot, split, max_seq_length, tokenizer):
        super().__init__(task,dataroot,split,max_seq_length, tokenizer)

    def preprocess_function(self, examples):
        sentences = [cls_task_token + sentence for sentence in examples["sentence"]]
        examples['class'] = [self.task] * len(examples["sentence"])
        tokenized_examples = self.tokenizer(sentences, truncation=True)

        return {k : v for k, v in tokenized_examples.items()}

class SocialIQADataset(Dataset) :
    def __init__(self, task, dataroot, split, max_seq_length, tokenizer):
        super().__init__(task,dataroot,split,max_seq_length, tokenizer)
        self.choice_names = ["answerA", "answerB", "answerC", "answerD", "answerE"]
        
    def preprocess_function(self, examples):
        first_sentences = [[com_token + question] * 5 for i, question in enumerate(examples["question"])]
        question_headers = examples["question"]
        second_sentences = [[f"{examples[choice][i]}" for choice in self.choice_names] for i, header in enumerate(question_headers)]
        examples['class'] = [self.task] * len(examples["question"])

        # Flatten
        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])
        
        tokenized_examples = self.tokenizer(first_sentences, second_sentences, truncation=True)

        # Un-flatten
        mapped_result = {k: [v[i:i+5] for i in range(0, len(v), 5)] for k, v in tokenized_examples.items()}
        return mapped_result

class MultiRCDataset(Dataset) :
    #passage,question,answer,label
    def __init__(self, task, dataroot, split, max_seq_length, tokenizer):
        super().__init__(task,dataroot,split,max_seq_length, tokenizer)
        
    def preprocess_function(self, examples):
        passage_question = [mrc_token + examples["question"][i] + "\n" + passage  for i, passage in enumerate(examples["passage"])]
        second_answer = [str(answer) for answer in examples["answer"]]
        examples['class'] = [self.task] * len(examples["question"])

        tokenized_examples = self.tokenizer(passage_question, second_answer, truncation=True)
        return {k : v for k, v in tokenized_examples.items()}


DatasetFactory = {
    "CoLA": CoLADataset,
    "SocialIQA": SocialIQADataset,
    "MultiRC": MultiRCDataset,
}




if __name__ == "__main__":
    #task = "CoLA"
    #task = "SocialIQA"
    task = "MultiRC"
    #dataroot = "classification/CoLA"
    #dataroot = "commonsense/socialIQA"
    dataroot = "mrc/multirc"
    max_seq_length = 128
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    split = "trainval"
    cola = DatasetFactory[task](task, dataroot, split, max_seq_length, tokenizer)
    processed = cola()
    print(processed)
    print("hello")
