from datasets import load_dataset, load_metric, ClassLabel
import random
import pandas as pd
import json
from transformers import AutoTokenizer
import os
from typing import List, Optional
from transformers import RobertaTokenizerFast, AddedToken

train_set = "train.csv"
dev_set = "dev.csv"
info_file = "info.json"
dir_path = os.path.dirname(os.path.abspath(__file__))

cls_task_token = "madeupword0002"
mc_token="madeupword0001"
span_token="madeupword0000"
task_types = ["span", "mc", "cls"]

train_key = "train"
eval_key = "eval"

class Dataset :
    def __init__(self, task, dataroot, split, task_type, task_category, task_choices, max_seq_length, tokenizer):
        path = dir_path + "/" + dataroot + '/'
        self.datasets = load_dataset('csv', data_files={train_key: path + train_set, eval_key : path + dev_set})
        self.configs = None
        with open(path + info_file) as json_file:
            self.configs = json.load(json_file)
        self.task = task
        self.task_type = task_type
        self.task_category = task_category
        self.split = split
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.task_choices = task_choices

    def __call__(self):
        self.datasets = self.datasets.map(self.preprocess_function, batched=True)
        return self.datasets

class CoLADataset(Dataset) :
    def __init__(self, task, dataroot, split, task_type, task_category, task_choices, max_seq_length, tokenizer):
        super().__init__(task, dataroot, split, task_type, task_category, task_choices, max_seq_length, tokenizer)

    def preprocess_function(self, examples):
        sentences = [cls_task_token + sentence for sentence in examples["sentence"]]
        examples['class'] = [self.task] * len(examples["sentence"])
        tokenized_examples = self.tokenizer(sentences, truncation=True)

        return {k : v for k, v in tokenized_examples.items()}

class CommonsenseQADataset(Dataset) :
    def __init__(self, task, dataroot, split, task_type, task_category, task_choices, max_seq_length, tokenizer):
        super().__init__(task, dataroot, split, task_type, task_category, task_choices, max_seq_length, tokenizer)
        self.choice_names = ["answerA", "answerB", "answerC", "answerD", "answerE"]
        
    def preprocess_function(self, examples):
        first_sentences = [[mc_token + question] * 5 for i, question in enumerate(examples["question"])]
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

class SocialIQADataset(Dataset) :
    def __init__(self, task, dataroot, split, task_type, task_category, task_choices, max_seq_length, tokenizer):
        super().__init__(task, dataroot, split, task_type, task_category, task_choices, max_seq_length, tokenizer)
        self.choice_names = ["answerA", "answerB", "answerC"]
        
    def preprocess_function(self, examples):
        first_sentences = [[mc_token + question] * 3 for i, question in enumerate(examples["Context"])]
        question_headers = examples["Context"]
        second_sentences = [[f"{examples[choice][i]}" for choice in self.choice_names] for i, header in enumerate(question_headers)]
        examples['class'] = [self.task] * len(examples["Context"])

        # Flatten
        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])
        
        tokenized_examples = self.tokenizer(first_sentences, second_sentences, truncation=True)

        # Un-flatten
        mapped_result = {k: [v[i:i+3] for i in range(0, len(v), 3)] for k, v in tokenized_examples.items()}
        return mapped_result

class MultiRCDataset(Dataset) :
    #passage,question,answer,label
    def __init__(self, task, dataroot, split, task_type, task_category, task_choices, max_seq_length, tokenizer):
        super().__init__(task, dataroot, split, task_type, task_category, task_choices, max_seq_length, tokenizer)
        
    def preprocess_function(self, examples):
        passage_question = [cls_task_token + examples["question"][i] + "\n" + passage  for i, passage in enumerate(examples["passage"])]
        second_answer = [str(answer) for answer in examples["answer"]]
        examples['class'] = [self.task] * len(examples["question"])

        tokenized_examples = self.tokenizer(passage_question, second_answer, truncation=True)
        return {k : v for k, v in tokenized_examples.items()}

class SquadDataset(Dataset) :
    def __init__(self, task, dataroot, split, task_type, task_category, task_choices, max_seq_length, tokenizer) :
        super().__init__(task, dataroot, split, task_type, task_category, task_choices, max_seq_length, tokenizer) 
    
    def preprocess_function(self, examples) :
        span_token_question = [span_token + examples["question"][i] for i, question in enumerate(examples["question"])]
        tokenized_examples = self.tokenizer(
            span_token_question,
            examples['context'],
            truncation=True,
            max_length=self.max_seq_length,
            return_overflowing_tokens=True, 
            return_offsets_mapping=True, 
        )
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping") 
        offset_mapping = tokenized_examples.pop("offset_mapping")

        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            span_index = input_ids.index(50261) ## must be modified
            sequence_ids = tokenized_examples.sequence_ids(i)
            sample_index = sample_mapping[i]
            answers = eval(examples["answers"][sample_index])

            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(span_index)
                tokenized_examples["end_positions"].append(span_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                token_start_index = 0
                while sequence_ids[token_start_index] != 1 :
                    token_start_index += 1

                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1 :
                    token_end_index -= 1

                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(span_index)
                    tokenized_examples["end_positions"].append(span_index)
                else:
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples  
    

DatasetFactory = {
    "CoLA": CoLADataset,
    "MultiRC": MultiRCDataset,
    "SocialIQA": SocialIQADataset,
    "CommonsenseQA": CommonsenseQADataset,
    "SQuAD1.1": SquadDataset
}

if __name__ == "__main__":
    #task = "CoLA"
    #task = "SocialIQA"
    # task = "MultiRC"
    task = "Squad1.1"
    #dataroot = "classification/CoLA"
    #dataroot = "commonsense/socialIQA"
    dataroot = "mrc/squad1.1"
    # dataroot = "mrc/multirc"
    task_choices = None
    max_seq_length = 1024
    task_type='span'
    task_category='mrc'
    tokenizer = RobertaMuppetTokenizer.from_pretrained('roberta-base')
    split = "trainval"
    squad = DatasetFactory[task](task, dataroot, split, task_type, task_category, task_choices, max_seq_length, tokenizer)
    processed = squad()
    print(processed['train']['input_ids'][0])
    print(tokenizer.decode(processed['train']['input_ids'][0]))
    print("hello")
