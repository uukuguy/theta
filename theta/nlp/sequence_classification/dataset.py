#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

from ..bert4torch.utils import sequence_padding
from .tagging import TaskLabels, TaskTag, TaggedData
from .utils import split_text_tags, split_sentences


def encode_text(text_a, text_b, task_tag, task_labels, max_length, tokenizer):
    label2id = task_labels.label2id

    token_ids, segment_ids = tokenizer.encode(text_a, text_b, maxlen=max_length)
    label = label2id[task_tag.label]

    return (
        (token_ids, segment_ids),
        label,
    )


def encode_sentences(text_tags_list, task_labels, max_length, tokenizer):

    token_ids_list, segment_ids_list = [], []
    labels_list = []
    for text_a, text_b, task_tag in tqdm(text_tags_list):

        ((token_ids, segment_ids), label) = encode_text(
            text_a, text_b, task_tag, task_labels, max_length, tokenizer
        )

        token_ids_list.append(token_ids)
        segment_ids_list.append(segment_ids)

        labels_list.append(label)

    return ((token_ids_list, segment_ids_list), labels_list)


class TaskDataset(Dataset):
    def __init__(self, args, data_generator, tokenizer):
        super().__init__()
        self.args = args
        self.data_generator = data_generator
        self.tokenizer = tokenizer

        self.data = [d for d in data_generator()]

        text_tags_list = [
            (tagged_data.text_a, tagged_data.text_b, tagged_data.label)
            for tagged_data in self.data
        ]

        ((token_ids_list, segment_ids_list), labels_list) = encode_sentences(
            text_tags_list, self.args.task_labels, self.args.max_length, self.tokenizer
        )

        self.token_ids_list = token_ids_list
        self.segment_ids_list = segment_ids_list

        self.labels_list = labels_list

    def __len__(self):
        return len(self.token_ids_list)

    def __getitem__(self, i):

        tagged_data = self.data[i]

        token_ids = self.token_ids_list[i]
        segment_ids = self.segment_ids_list[i]

        label = self.labels_list[i]

        return (tagged_data, (token_ids, segment_ids), label)
