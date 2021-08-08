# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from io import open
import json
import logging
import os
import sys

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger(__name__)

LossMap = {
    "BCEWithLogitLoss": nn.BCEWithLogitsLoss(reduction="mean"),
    "CrossEntropyLoss": nn.CrossEntropyLoss(),
}

def ForwardModelsTrain(
    args,
    task_cfg,
    device,
    task_id,
    task_count,
    task_iter_train,
    task_dataloader_train,
    model,
    task_losses,
):
    # given the current task, decided whether to forward the model and forward with specific loss.

    # reset the task iteration when needed.
    if task_count[task_id] % len(task_dataloader_train[task_id]) == 0:
        task_iter_train[task_id] = iter(task_dataloader_train[task_id])

    task_count[task_id] += 1
    # get the batch
    batch = task_iter_train[task_id].next()
    batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)

    if task_id == "TASK4" or task_id == "TASK17":
        features, spatials, image_mask, question, target, input_mask, segment_ids, multiple_choice_ids, co_attention_mask, question_id = (
            batch
        )
    else:
        features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, question_id = (
            batch
        )

    batch_size = features.size(0)
    if task_cfg[task_id]["process"] in ["dialog"]:
        max_num_bbox = features.size(1)
        nround = question.size(1)
        num_options = question.size(2)
        rbatch_size = batch_size * nround
        question = question.view(rbatch_size, question.size(2), question.size(3))
        target = target.view(-1)
        input_mask = input_mask.view(
            rbatch_size, input_mask.size(2), input_mask.size(3)
        )
        segment_ids = segment_ids.view(
            rbatch_size, segment_ids.size(2), segment_ids.size(3)
        )
        co_attention_mask = co_attention_mask.view(
            rbatch_size,
            co_attention_mask.size(2),
            co_attention_mask.size(3),
            co_attention_mask.size(4),
        )

        features = (
            features.unsqueeze(1)
            .unsqueeze(1)
            .expand(batch_size, nround, num_options, max_num_bbox, 2048)
            .contiguous()
            .view(-1, max_num_bbox, 2048)
        )
        spatials = (
            spatials.unsqueeze(1)
            .unsqueeze(1)
            .expand(batch_size, nround, num_options, max_num_bbox, 5)
            .contiguous()
            .view(-1, max_num_bbox, 5)
        )
        image_mask = (
            image_mask.unsqueeze(1)
            .expand(batch_size, nround, num_options, max_num_bbox)
            .contiguous()
            .view(-1, max_num_bbox)
        )

        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))
        co_attention_mask = co_attention_mask.view(
            -1, co_attention_mask.size(2), co_attention_mask.size(3)
        )
        batch_size = rbatch_size

    elif task_cfg[task_id]["process"] in ["expand"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = (
            features.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox, 2048)
            .contiguous()
            .view(-1, max_num_bbox, 2048)
        )
        spatials = (
            spatials.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox, 5)
            .contiguous()
            .view(-1, max_num_bbox, 5)
        )
        image_mask = (
            image_mask.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox)
            .contiguous()
            .view(-1, max_num_bbox)
        )
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))
        co_attention_mask = co_attention_mask.view(
            -1, co_attention_mask.size(2), co_attention_mask.size(3)
        )

    elif task_cfg[task_id]["process"] in ["retrieval"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(-1, features.size(2), features.size(3))
        spatials = spatials.view(-1, spatials.size(2), spatials.size(3))
        image_mask = image_mask.view(-1, image_mask.size(2))
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))
        co_attention_mask = co_attention_mask.view(
            -1, co_attention_mask.size(2), co_attention_mask.size(3)
        )

    elif task_cfg[task_id]["process"] in ["nlvr"]:
        batch_size = features.size(0)
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(
            batch_size * 2, int(features.size(1) / 2), features.size(2)
        )
        spatials = spatials.view(
            batch_size * 2, int(spatials.size(1) / 2), spatials.size(2)
        )
        image_mask = image_mask.view(batch_size * 2, int(image_mask.size(1) / 2))
        question = question.repeat(1, 2)
        question = question.view(batch_size * 2, int(question.size(1) / 2))
        input_mask = input_mask.repeat(1, 2)
        input_mask = input_mask.view(batch_size * 2, int(input_mask.size(1) / 2))
        segment_ids = segment_ids.repeat(1, 2)
        segment_ids = segment_ids.view(batch_size * 2, int(segment_ids.size(1) / 2))
        co_attention_mask = co_attention_mask.view(
            batch_size * 2,
            int(co_attention_mask.size(1) / 2),
            co_attention_mask.size(2),
        )

    task_tokens = question.new().resize_(question.size(0), 1).fill_(int(task_id[4:]))
    output = model(
        question,
        features,
        spatials,
        segment_ids,
        input_mask,
        image_mask,
        co_attention_mask,
        task_tokens,
    )

    # for different task, we use different output to calculate the loss.
    if task_cfg[task_id]["type"] == "VL-classifier":
        loss = task_losses[task_id](vil_prediction, target)
        loss = loss.mean() * target.size(1)
        batch_score = compute_score_with_logits(vil_prediction, target).sum() / float(
            batch_size
        )

    elif task_cfg[task_id]["type"] == "VL-classifier-GQA":
        loss = task_losses[task_id](vil_prediction_gqa, target)
        loss = loss.mean() * target.size(1)
        batch_score = compute_score_with_logits(
            vil_prediction_gqa, target
        ).sum() / float(batch_size)

    elif task_cfg[task_id]["type"] == "VL-logit":
        vil_logit = vil_logit.view(batch_size, num_options)
        loss = task_losses[task_id](vil_logit, target)
        _, preds = torch.max(vil_logit, 1)
        batch_score = float((preds == target).sum()) / float(batch_size)

    elif task_cfg[task_id]["type"] == "V-logit":
        loss = task_losses[task_id](vision_logit, target)
        loss = loss.mean() * target.size(1)
        _, select_idx = torch.max(vision_logit, dim=1)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = float(torch.sum(select_target > 0.5)) / batch_size

    elif task_cfg[task_id]["type"] == "V-logit-mc":
        vision_logit = vision_logit[:, 101:]
        vision_logit = vision_logit.squeeze(2).gather(1, multiple_choice_ids)
        vision_logit = vision_logit.unsqueeze(2)
        loss = task_losses[task_id](vision_logit, target)
        loss = loss.mean() * target.size(1)
        _, preds = torch.max(vision_logit, dim=1)
        _, target = torch.max(target, dim=1)
        batch_score = float((preds == target).sum()) / float(batch_size)

    elif task_cfg[task_id]["type"] == "VL-binary-classifier":
        loss = task_losses[task_id](vil_binary_prediction, target)
        loss = loss.mean()
        batch_score = compute_score_with_logits(
            vil_binary_prediction, target
        ).sum() / float(batch_size)

    elif task_cfg[task_id]["type"] == "VL-tri-classifier":
        loss = task_losses[task_id](vil_tri_prediction, target)
        loss = loss.mean()
        batch_score = compute_score_with_logits(
            vil_tri_prediction, target
        ).sum() / float(batch_size)

    return loss, batch_score