#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

from ...bert4torch.utils import sequence_padding
from ..tagging import TaskLabels, TaskTag, TaggedData
from .utils import split_text_tags, split_sentences


def get_default_tokenizer(dict_path):
    from ...bert4torch.tokenizers import Tokenizer

    tokenizer = Tokenizer(dict_path, do_lower_case=True)

    return tokenizer


def encode_text(text, tags, task_labels, max_length, tokenizer):
    entities_label2id = task_labels.entities_label2id

    tokens = tokenizer.tokenize(text, maxlen=max_length)
    mapping = tokenizer.rematch(text, tokens)
    start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
    end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
    token_ids = tokenizer.tokens_to_ids(tokens)

    labels = np.zeros((len(entities_label2id), max_length, max_length))
    for tag in tags:
        start, end, label = tag.s, tag.s + len(tag.m) - 1, tag.c
        if start in start_mapping and end in end_mapping:
            start = start_mapping[start]
            end = end_mapping[end]
            label = entities_label2id[label]
            labels[label, start, end] = 1
    labels = labels[:, : len(token_ids), : len(token_ids)]

    return (tokens, mapping), token_ids, labels


def encode_sentences(text_tags_list, task_labels, max_length, tokenizer):

    tokens_list, mappings_list = [], []
    token_ids_list = []
    labels_list = []
    for text, tags in tqdm(text_tags_list):

        ((tokens, mapping), token_ids, labels) = encode_text(
            text, tags, task_labels, max_length, tokenizer
        )

        tokens_list.append(tokens)
        mappings_list.append(mapping)

        token_ids_list.append(token_ids)

        labels_list.append(labels)

    return ((tokens_list, mappings_list), token_ids_list, labels_list)


class TaskDataset(Dataset):
    def __init__(self, args, data_generator, tokenizer):
        super().__init__()
        self.args = args
        self.data_generator = data_generator
        self.tokenizer = tokenizer

        self.data = [d for d in data_generator()]

        text_tags_list = [
            (tagged_data.text, tagged_data.tags) for tagged_data in self.data
        ]

        ((tokens_list, mappings_list), token_ids_list, labels_list) = encode_sentences(
            text_tags_list, self.args.task_labels, self.args.max_length, self.tokenizer
        )

        self.tokens_list, self.mappings_list = tokens_list, mappings_list

        self.token_ids_list = token_ids_list

        self.labels_list = labels_list

    def __len__(self):
        return len(self.token_ids_list)

    def __getitem__(self, i):

        tagged_data = self.data[i]

        tokens = self.tokens_list[i]
        mapping = self.mappings_list[i]

        token_ids = self.token_ids_list[i]

        labels = self.labels_list[i]

        return (tagged_data, (tokens, mapping), token_ids, labels)
