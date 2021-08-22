from datasets import load_metric
import collections
import numpy as np
from tqdm.auto import tqdm
import torch

TASK_METRICS = {
    "TASK1" : load_metric("accuracy"),
    "TASK2" : load_metric("glue", "cola"),
    "TASK3" : load_metric("accuracy"),
    "TASK4" : load_metric("accuracy"),
    "TASK4" : load_metric("accuracy"),
    "TASK5" : load_metric("accuracy") #load_metric("squad")
}


def postprocess_qa_predictions(features, raw_predictions):
    all_start_logits, all_end_logits = raw_predictions["start_logits"], raw_predictions["end_logits"]
    # Build a map example to its corresponding features.
    # Let's loop over all the examples!
    # Those are the indices of the features associated to the current example.
    
    #context = example["context"]
    # Looping through all the features associated to the current example.
    # We grab the predictions of the model for this feature.
    #start_logits = all_start_logits[i][features["start_positions"][i]]
    #end_logits = all_end_logits[i][features["end_positions"][i]]
        
        
    # Go through all possibilities for the `n_best_size` greater start and end logits.
    start_indexes = torch.argmax(all_start_logits, dim=1)
    end_indexes = torch.argmax(all_end_logits, dim=1)

    start_preds = start_indexes == features["start_positions"]
    end_preds = end_indexes == features["end_positions"]

    fin_preds = (start_preds == end_preds).to(torch.long)
    label = (start_preds == start_preds).to(torch.long)
    return fin_preds, label