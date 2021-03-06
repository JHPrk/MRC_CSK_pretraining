import datasets
from datasets import load_dataset
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from random import choices
from collections import Counter

from dataset.mtl_datasets import DatasetFactory, task_types
from dataset.mtl_data_collator import CollatorFactory
from dataset.mtl_sampler import SamplerFactory, compute_sampling_probability

from datasets.utils.logging import set_verbosity_error
import torch
set_verbosity_error()
datasets.logging.set_verbosity(datasets.logging.ERROR)

categories = ["classification", "commonsense", "mrc"]
train_set = "train.csv"
dev_set = "dev.csv"
test_set = "test.csv"
info_file = "info.json"
dataset_path = "./"
accepted_keys = ["input_ids", "attention_mask", "label"]
task_5_keys = ["input_ids", "attention_mask", "start_positions", "end_positions"]

class StrIgnoreDevice(str):
    def to(self, device):
        return self

class MtpDataLoader:
    def __init__(self, model_name_or_path, batch_size, tokenizer, 
        task_configs, task_datasets, task_datasets_loss, task_datasets_collator, 
        task_datasets_sampler, task_datasets_loader, task_ids, task_types, split_val = "train"):

        self.task_configs = task_configs
        self.task_datasets = task_datasets
        self.task_datasets_loss = task_datasets_loss
        self.task_datasets_collator = task_datasets_collator
        self.task_datasets_sampler = task_datasets_sampler
        self.task_datasets_loader = task_datasets_loader
        self.task_ids = task_ids
        self.task_types = task_types
        self.model_name_or_path = model_name_or_path
        self.tokenizer = tokenizer
        self.split_val = split_val

        self.sampling_probability, self.total_datasize = compute_sampling_probability(task_configs, split_val, task_ids)
        self.batch_size = batch_size
        self.total_steps = int(self.total_datasize / self.batch_size)
        self.cur = 0
        
    @classmethod
    def create(cls, model_name_or_path, task_ids, batch_size, tokenizer, task_args,split="trainval"):
        task_configs, task_datasets, task_datasets_loss, task_datasets_collator, task_datasets_sampler, task_datasets_loader, task_ids, task_types = cls.LoadDataset(task_ids, batch_size, tokenizer, task_args)
        train_dataset = cls(model_name_or_path, batch_size, tokenizer, task_configs, task_datasets["train"], task_datasets_loss, task_datasets_collator, task_datasets_sampler["train"], task_datasets_loader["train"], task_ids, task_types, "train")
        eval_dataset = cls(model_name_or_path, batch_size, tokenizer, task_configs, task_datasets["eval"], task_datasets_loss, task_datasets_collator, task_datasets_sampler["eval"], task_datasets_loader["eval"], task_ids, task_types, "eval")
        return train_dataset, eval_dataset
    # dataset Load needs
    # task_names
    # cateogries : classification, commonsense, mrc
    # files structure : train.csv, dev.csv, test.csv with info.json
    # json file has ... choices, type, columns
    @classmethod
    def LoadDataset(cls, task_ids, batch_size, tokenizer, task_args, split="trainval"):
        ids = task_ids


        task_configs = {}
        task_datasets = {}
        task_datasets_collator = {}
        task_datasets_sampler = {}
        task_datasets_loader = {}
        task_datasets_loss = {}
        task_ids = []
        task_types = []
        cls_id = 1
        task_datasets["train"] = {}
        task_datasets["eval"] = {}
        task_datasets_sampler["eval"] = {}
        task_datasets_sampler["train"] = {}
        task_datasets_loader["train"] = {}
        task_datasets_loader["eval"] = {}

        for i, task_id in enumerate(ids):
            task = "TASK" + str(task_id)
            task_name = task_args[task]["name"]
            split = task_args[task]['train_split']
            if "train" in split :
                task_ids.append(task)
            task_type = task_args[task]["type"]
            if task_type == "cls":
                task_type += str(task_id)
                cls_id += 1
            task_types.append(task_type)
            task_category = task_args[task]["category"]
            task_choices = task_args[task]["choices"]
            dataroot = task_args[task]["dataroot"]
            max_seq_length = task_args[task]['max_seq_length']
            loss = task_args[task]['loss']
            task_datasets_loss[task] = loss
            task_configs[task] = DatasetFactory[task_name](task_name, dataroot, split, task_type, task_category, task_choices, max_seq_length, tokenizer)
            cur_datasets = task_configs[task]()
            task_datasets_collator[task] = CollatorFactory[task_name](tokenizer)

            if "train" in split :
                task_datasets["train"][task] = cur_datasets["train"]
                if task == "TASK5":
                    task_datasets["train"][task].set_format(type='torch', columns=['input_ids', 'attention_mask', 'start_positions', 'end_positions'])
                else:
                    task_datasets["train"][task].set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
                task_datasets_sampler["train"][task] = SamplerFactory[task_name](task_datasets["train"][task])
                task_datasets_loader["train"][task] = DataLoader(
                    task_datasets["train"][task],
                    sampler=task_datasets_sampler["train"][task],
                    collate_fn=task_datasets_collator[task],
                    batch_size=batch_size,
                    pin_memory=True,
                )
            if "eval" in split:
                task_datasets["eval"][task] = cur_datasets["eval"]
                if task == "TASK5":
                    task_datasets["eval"][task].set_format(type='torch', columns=['input_ids', 'attention_mask', 'start_positions', 'end_positions'])
                else:
                    task_datasets["eval"][task].set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
                task_datasets_sampler["eval"][task] = SamplerFactory[task_name](task_datasets["eval"][task])
                task_datasets_loader["eval"][task] = DataLoader(
                    task_datasets["eval"][task],
                    sampler=task_datasets_sampler["eval"][task],
                    collate_fn=task_datasets_collator[task],
                    batch_size=batch_size,
                    pin_memory=True,
                )

        return (task_configs, 
            task_datasets,
            task_datasets_loss, 
            task_datasets_collator,
            task_datasets_sampler,
            task_datasets_loader,
            task_ids,
            task_types
        )

    def get_scaling_factor(self, task_name):
        return self.task_configs[task_name].task_choices if isinstance(self.task_configs[task_name].task_choices,int) else 512
        
    def _num_each_task_in_batch(self):
        batch_samples = choices(self.task_ids, self.sampling_probability, k=self.batch_size)
        return Counter(batch_samples)

    def _get_batch(self, task_batch_counter):
        batch = {}
        for task in self.task_ids:
            task_batch = task_batch_counter[task]
            indices = []
            if task_batch == 0 :
                continue
            for i,x in enumerate(self.task_datasets_sampler[task]) : 
                indices.append(x)
                if (i + 1) % task_batch == 0 : 
                    break
            #features = self.task_datasets[task][indices]
            if task == "TASK5" : 
                batch_keys = task_5_keys
            else:
                batch_keys = accepted_keys
            features= [{k:v for k, v in self.task_datasets[task][i].items() if k in batch_keys} for i in indices]
            batch[task] = self.task_datasets_collator[task](features)
            """if batch[task]['input_ids'].shape[1] > 514 :
                for input_ids in batch[task]['input_ids']:
                    if input_ids[-1] != 1:
                        print(self.tokenizer.decode(input_ids))
                assert batch[task]['input_ids'].shape[1] > 514"""
        return batch    

    def __iter__(self) :
        self.cur = 0
        while self.total_steps > self.cur:
            task_batch_counter = self._num_each_task_in_batch()
            batch = self._get_batch(task_batch_counter)
            self.cur += 1
            yield batch

    def __len__(self):
        return self.total_steps

    def select(self, total_num):
        self.total_datasize = total_num
        self.total_steps = int(self.total_datasize / self.batch_size)
        if self.split_val == "eval":
            for task in self.task_datasets : 
                self.task_datasets[task].select(range(total_num))
                self.task_datasets_loader[task] = DataLoader(
                    self.task_datasets[task],
                    sampler=self.task_datasets_sampler[task],
                    collate_fn=self.task_datasets_collator[task],
                    batch_size=self.batch_size,
                    pin_memory=True,
                )


    def get_task_types(self):
        return self.task_types


def main(args,task_args):
    #task_configs, task_datasets, task_datasets_loss, task_datasets_collator, task_datasets_sampler, task_datasets_loader, task_ids = LoadDataset(args, task_args, ids)
    #sampling_probability = compute_sampling_probability(task_configs, "train")
    #print(sampling_probability)
    #task_batch_counter = num_each_task_in_batch(task_ids, sampling_probability, args["batch_size"])
    #print(task_batch_counter)
    #batch = get_batch(task_ids, task_batch_counter, task_datasets_collator, task_datasets, task_datasets_sampler)
    #for value in batch:
    #    print(batch[value])
    # task 1 multirc: (bn, 511) output: (bn) 0, 1
    # task 2 cola: (bn, 23) output: (bn) 0, 1
    # task 3 socialIQA: (bn, 3, 40) output: (bn) 0, 1, 2
    # task 4 CommonsenseQA: (bn, 5, 28) output: (bn) 0, 1, 2, 3, 4
    mtp_loader = MtpDataLoader(args, task_args)
    for i, batch in enumerate(mtp_loader):
        print("steps : ", i)
        for task_batch in batch:
            print(task_batch, " : ", len(batch[task_batch]['labels']))

if __name__ == '__main__':
    import yaml
    import os
    from os.path import dirname, abspath
    print(dirname(dirname(abspath(__file__))))
    with open(dirname(dirname(abspath(__file__))) + '/multi_tasks.yml') as f:
        task_args = yaml.load(f, Loader=yaml.FullLoader)
        print(task_args['TASK1'])
    args = {"model_name_or_path" : "roberta-base", "batch_size" : 64, "task_ids" : [1, 2, 3, 4]}
    main(args, task_args)