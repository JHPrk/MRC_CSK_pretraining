import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import torch
import math

import datasets
from datasets import load_metric

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    AdamW,
    get_scheduler,
    SchedulerType
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from typing import List
import yaml
from dataset.dataset_loader import MtpDataLoader, StrIgnoreDevice
from custom_tokenizers.muppet_tokenizer import RobertaMuppetTokenizer
from tqdm.auto import tqdm
from model.mtl_model import MultitaskModel

from torch.optim.lr_scheduler import (
    LambdaLR,
    ReduceLROnPlateau,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
)
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

from evaluation.evaluate_function import TASK_METRICS, postprocess_qa_predictions

import datasets
from datasets.utils.logging import set_verbosity_error
set_verbosity_error()
datasets.logging.set_verbosity(datasets.logging.ERROR)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.9.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    num_warmup_steps: int = field(
        default=0,
        metadata={
            "help":"Number of steps for the warmup in the lr scheduler."
        }
    )
    max_train_steps: int = field(
        default=None,
        metadata={
            "help":"Total number of training steps to perform. If provided, overrides num_train_epochs."
        }
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    total_batch_size: Optional[int] = field(
        default=64,
        metadata={
            "help": "Batch size of sampled data according to data distribution"
        },
    )
    task_ids: List[int] = field(default_factory=list,
        metadata={
            "help": "multi-tasking learning dataset ids ([1,2,3])"
        },
    )
    tasks: List[str] = field(default_factory=list,
        metadata={
            "help": "multi-tasking learning dataset names ([\"multiRC\", \"CoLA\", \"SocialIQA\", \"SQUAD\"])"
        },
    )

def mkdir_p(path):
    """ Creates a path recursively without throwing an error if it already exists
    :param path: path to create
    :return: None
    """
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath("./config.json"))
    dataset_args = {}
    with open(data_args.dataset_config_name) as f:
        dataset_args = yaml.load(f, Loader=yaml.FullLoader)
    """
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    """
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    ddp_kwargs =  DistributedDataParallelKwargs(find_unused_parameters=training_args.gradient_accumulation_steps > 1)
    #ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")


    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    
    if model_args.model_name_or_path:
        tokenizer = RobertaMuppetTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    total_batch_size = training_args.per_device_train_batch_size * accelerator.num_processes * training_args.gradient_accumulation_steps
    train_dataloader, eval_dataloader = MtpDataLoader.create(model_args.model_name_or_path, data_args.task_ids, total_batch_size, tokenizer, dataset_args)
    #model_name_or_path, task_ids, batch_size, task_args, tokenizer
    #### 진양씨 모델 붙여넣으세요~~
    model_name = model_args.model_name_or_path
    task_types = train_dataloader.get_task_types()
    task_types = list(set(task_types))
    task_types.sort()
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        multitask_model = MultitaskModel.create(
            model_name = model_name,
            model_types = ["cls1","cls2","mc","span"]
        )
    else :
        multitask_model = MultitaskModel.create(
            model_name = model_name,
            model_types = task_types
        )
    multitask_model.to(device)

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # data_loader.task_datasets["TASK1"]["train"].select(range(100))
    if training_args.do_train:
        if data_args.max_train_samples is not None:
            train_dataloader.select(data_args.max_train_samples)
    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            eval_dataloader.select(data_args.max_eval_samples)
    # Trainer 클래스 선언해서 모델을 돌려야한다.

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in multitask_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in multitask_model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)

    multitask_model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        multitask_model, optimizer, train_dataloader, eval_dataloader
    )
    #with open(f"params_{accelerator.process_index}.txt", "w") as fo:
    #    for i, param in enumerate(multitask_model.parameters()):
    #        fo.write(f"{str(i)}\t{param.size()}\n")

    if training_args.resume_from_checkpoint is not None:
        unwrapped_model = accelerator.unwrap_model(multitask_model)
        unwrapped_model.load_state_dict(torch.load(checkpoint+"/pytorch_model.bin"))

    total_batch_size = training_args.per_device_train_batch_size * accelerator.num_processes * training_args.gradient_accumulation_steps

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    num_update_steps_per_epoch = int(num_update_steps_per_epoch + 0.5)

    if model_args.max_train_steps is None:
        model_args.max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch
    else:
        training_args.num_train_epochs = math.ceil(model_args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name= training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps= model_args.num_warmup_steps,
        num_training_steps= model_args.max_train_steps,
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Total optimization steps = {model_args.max_train_steps}")


    devices = {"TASK1" : "cuda:1", "TASK2" : "cuda:2", "TASK3" : "cuda:4", "TASK4" : "cuda:4", "TASK5" : "cuda:6", "TASK6" : "cuda:3"}
    datasets.logging.set_verbosity(datasets.logging.ERROR)
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(model_args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    mkdir_p(training_args.output_dir)
    """multitask_model.eval()
    eval_metric = {}
    for key in eval_dataloader.task_datasets_loader:
        for step, batch in enumerate(tqdm(eval_dataloader.task_datasets_loader[key])):
            task_batch = batch
            task_batch["task_name"] = StrIgnoreDevice(key)
            multitask_model.to(devices[key])
            try : 
                task_batch = task_batch.to(devices[key]) # device
            except:
                task_batch = {k : v.to(devices[key]) for k, v in task_batch.items() } #device
            with torch.no_grad():
                outputs = multitask_model(**task_batch)
            
            if key != "TASK5":
                logits = outputs.logits
                predictions = logits.argmax(dim=-1)
                TASK_METRICS[key].add_batch(
                    predictions=accelerator.gather(predictions),
                    references=accelerator.gather(batch["labels"]),
                )
            else :
                preds, label = postprocess_qa_predictions(task_batch, outputs)
                TASK_METRICS[key].add_batch(
                    predictions=accelerator.gather(preds),
                    references=accelerator.gather(label),
                )

        eval_metric[key] = TASK_METRICS[key].compute()
        task_name = dataset_args[key]["name"]
        logger.info(f"epoch init, key {task_name} : {eval_metric[key]}")
    with open(training_args.output_dir + f"/eval_epoch_init.txt", "w") as fo:
        for key2 in eval_metric:
            task_name = dataset_args[key2]["name"]
            fo.write(f"epoch init, key {task_name} : {eval_metric[key2]}\n")"""
    for epoch in range(training_args.num_train_epochs):
        acc_loss = {}
        for task_id in train_dataloader.task_ids:
            acc_loss[task_id] = []
        multitask_model.train()
        for step, batch in enumerate(tqdm(train_dataloader)):
            cur_loss = {}
            for key in batch:
                task_batch = batch[key]
                task_batch["task_name"] = StrIgnoreDevice(key)
                multitask_model.to(devices[key])
                try : 
                    task_batch = task_batch.to(devices[key]) # device
                except:
                    task_batch = {k : v.to(devices[key]) for k, v in task_batch.items() }
                #task_batch.to(device)
                outputs = multitask_model(**task_batch)
                # scaling need
                scale_factor = train_dataloader.get_scaling_factor(key)
                loss = outputs.loss / training_args.gradient_accumulation_steps / math.log(scale_factor, 1000)
                cur_loss[key] = outputs.loss.item()
                accelerator.backward(loss)
            if (completed_steps) % 50 == 0 :
                with open(training_args.output_dir + f"/epoch_{str(epoch)}_step_{str(step)}.txt", "w") as fo:
                    for key2 in cur_loss:
                        task_name = dataset_args[key2]["name"]
                        fo.write(f"key {task_name} loss : {cur_loss[key2]}\n")
                for task_id in train_dataloader.task_ids:
                    if task_id in cur_loss.keys():
                        acc_loss[task_id].append(cur_loss[task_id])
                    else:
                        acc_loss[task_id].append(" ")
            if step % training_args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                multitask_model.to(device)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                progress_bar.set_postfix(cur_loss)
                completed_steps += 1

            if completed_steps >= model_args.max_train_steps:
                break
        multitask_model.eval()
        eval_metric = {}
        for key in eval_dataloader.task_datasets_loader:
            for step, batch in enumerate(tqdm(eval_dataloader.task_datasets_loader[key])):
                task_batch = batch
                task_batch["task_name"] = StrIgnoreDevice(key)
                multitask_model.to(devices[key])
                try : 
                    task_batch = task_batch.to(devices[key]) # device
                except:
                    task_batch = {k : v.to(devices[key]) for k, v in task_batch.items() } #device
                with torch.no_grad():
                    outputs = multitask_model(**task_batch)
                
                if key != "TASK5":
                    logits = outputs.logits
                    predictions = logits.argmax(dim=-1)
                    TASK_METRICS[key].add_batch(
                        predictions=accelerator.gather(predictions),
                        references=accelerator.gather(batch["labels"]),
                    )
                else :
                    preds, label = postprocess_qa_predictions(task_batch, outputs)
                    TASK_METRICS[key].add_batch(
                        predictions=accelerator.gather(preds),
                        references=accelerator.gather(label),
                    )

            eval_metric[key] = TASK_METRICS[key].compute()
            task_name = dataset_args[key]["name"]
            logger.info(f"epoch {epoch}, key {task_name} : {eval_metric[key]}")
        with open(training_args.output_dir + f"/eval_epoch_{str(epoch)}.txt", "w") as fo:
            for key2 in eval_metric:
                task_name = dataset_args[key2]["name"]
                fo.write(f"epoch {epoch}, key {task_name} : {eval_metric[key2]}\n")
        with open(training_args.output_dir + f"/epoch_{str(epoch)}_loss.txt", "w") as fo:
            for key2 in acc_loss:
                task_name = dataset_args[key2]["name"]
                fo.write(f"key {task_name} loss : {acc_loss[key2]}\n")
        check_point_dir = training_args.output_dir + "/epoch_" + str(epoch)
        mkdir_p(check_point_dir)
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(multitask_model)
        unwrapped_model.save_pretrained(check_point_dir, save_function=accelerator.save)

    if training_args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(multitask_model)
        unwrapped_model.save_pretrained(training_args.output_dir, save_function=accelerator.save)
    # Initialize our Trainer
    """
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:

        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub:
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "fill-mask"}
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

        trainer.push_to_hub(**kwargs)

    """


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
