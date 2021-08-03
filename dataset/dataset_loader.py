import datasets
from datasets import load_dataset
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from random import choices
from collections import Counter

from dataset.mtl_datasets import DatasetFactory, task_types
from custom_tokenizers.muppet_tokenizer import RobertaMuppetTokenizer
from dataset.mtl_data_collator import CollatorFactory
from dataset.mtl_sampler import SamplerFactory, compute_sampling_probability

categories = ["classification", "commonsense", "mrc"]
train_set = "train.csv"
dev_set = "dev.csv"
test_set = "test.csv"
info_file = "info.json"
dataset_path = "./"



class MtpDataLoader:
    def __init__(self, task_ids, model_name_or_path, batch_size, task_args, split="trainval"):
        task_configs, task_datasets, task_datasets_loss, task_datasets_collator, task_datasets_sampler, task_datasets_loader, task_ids = self.LoadDataset(task_ids, model_name_or_path, task_args)
        self.task_configs = task_configs
        self.task_datasets = task_datasets
        self.task_datasets_loss = task_datasets_loss
        self.task_datasets_collator = task_datasets_collator
        self.task_datasets_sampler = task_datasets_sampler
        self.task_datasets_loader = task_datasets_loader
        self.task_ids = task_ids

        self.sampling_probability, self.total_datasize = compute_sampling_probability(task_configs, "train")
        self.batch_size = batch_size
        self.total_steps = int(self.total_datasize / self.batch_size)
        self.cur = 0
        

    # dataset Load needs
    # task_names
    # cateogries : classification, commonsense, mrc
    # files structure : train.csv, dev.csv, test.csv with info.json
    # json file has ... choices, type, columns
    def LoadDataset(self, task_ids, model_name_or_path, task_args, split="trainval"):
        ids = task_ids
        tokenizer = RobertaMuppetTokenizer.from_pretrained(model_name_or_path, use_fast=True)


        task_configs = {}
        task_datasets = {}
        task_datasets_collator = {}
        task_datasets_sampler = {}
        task_datasets_loader = {}
        task_datasets_loss = {}
        task_ids = []

        for i, task_id in enumerate(ids):
            task = "TASK" + str(task_id)
            task_name = task_args[task]["name"]
            task_ids.append(task)
            task_type = task_args[task]["type"]
            task_category = task_args[task]["category"]
            task_choices = task_args[task]["choices"]
            dataroot = task_args[task]["dataroot"]
            max_seq_length = task_args[task]['max_seq_length']
            split = task_args[task]['train_split']
            loss = task_args[task]['loss']
            task_datasets_loss[task] = loss
            task_configs[task] = DatasetFactory[task_name](task_name, dataroot, split, task_type, task_category, task_choices, max_seq_length, tokenizer)
            cur_datasets = task_configs[task]()

            task_datasets[task] = {}
            task_datasets[task]["train"] = cur_datasets["train"]
            task_datasets[task]["train"].set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
            task_datasets[task]["eval"] = cur_datasets["eval"]
            task_datasets[task]["eval"].set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

            task_datasets_collator[task] = CollatorFactory[task_name](tokenizer)

            task_datasets_sampler[task] = {}
            task_datasets_sampler[task]["train"] = SamplerFactory[task_name](task_datasets[task]["train"])
            task_datasets_sampler[task]["eval"] = SamplerFactory[task_name](task_datasets[task]["eval"])

            task_datasets_loader[task] = {}
            """
            task_datasets_loader[task]["train"] = DataLoader(
                    task_datasets[task]["train"],
                    sampler=task_datasets_sampler[task]["train"],
                    collate_fn=task_datasets_collator[task],
                    batch_size=2,
                    pin_memory=True,
                )
            task_datasets_loader[task]["eval"] = DataLoader(
                    task_datasets[task]["eval"],
                    sampler=task_datasets_sampler[task]["eval"],
                    collate_fn=task_datasets_collator[task],
                    batch_size=2,
                    pin_memory=True,
                )
                """
        return (task_configs, 
            task_datasets,
            task_datasets_loss, 
            task_datasets_collator,
            task_datasets_sampler,
            task_datasets_loader,
            task_ids
        )

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
            for i,x in enumerate(self.task_datasets_sampler[task]["train"]) : 
                indices.append(x)
                if (i + 1) % task_batch == 0 : 
                    break
            features= [{k:v for k, v in self.task_datasets[task]["train"][i].items()} for i in indices]
            batch[task] = self.task_datasets_collator[task](features)
        return batch    

    def __iter__(self) :
        self.cur = 0
        while self.total_steps > self.cur:
            task_batch_counter = self._num_each_task_in_batch()
            batch = self._get_batch(task_batch_counter)
            self.cur += 1
            yield batch


def main(args,task_args):
    from tqdm.auto import tqdm
    #task_configs, task_datasets, task_datasets_loss, task_datasets_collator, task_datasets_sampler, task_datasets_loader, task_ids = LoadDataset(args, task_args, ids)
    #sampling_probability = compute_sampling_probability(task_configs, "train")
    #print(sampling_probability)
    #task_batch_counter = num_each_task_in_batch(task_ids, sampling_probability, args["batch_size"])
    #print(task_batch_counter)
    #batch = get_batch(task_ids, task_batch_counter, task_datasets_collator, task_datasets, task_datasets_sampler)
    #for value in batch:
    #    print(batch[value])
    mtp_loader = MtpDataLoader(args, task_args)
    progress = tqdm(range(mtp_loader.total_steps))
    for i, batch in enumerate(mtp_loader):
        print("steps : ", i)
        for task_batch in batch:
            print(task_batch, " : ", len(batch[task_batch]['labels']))
        progress.update(1)

if __name__ == '__main__':
    import yaml
    import os
    from os.path import dirname, abspath
    print(dirname(dirname(abspath(__file__))))
    with open(dirname(dirname(abspath(__file__))) + '/multi_tasks.yml') as f:
        task_args = yaml.load(f, Loader=yaml.FullLoader)
        print(task_args['TASK1'])
    args = {"model_name_or_path" : "roberta-base", "batch_size" : 64, "task_ids" : [2,3,4]}
    main(args, task_args)