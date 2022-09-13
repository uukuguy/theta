#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from torch.utils.data import Dataset

from .utils import split_text_tags, split_sentences


def get_default_tokenizer(dict_path):
    from ..bert4torch.tokenizers import Tokenizer
    tokenizer = Tokenizer(dict_path, do_lower_case=True)

    return tokenizer


def encode_text(text, tags, categories_label2id, max_length, tokenizer):
    tokens = tokenizer.tokenize(text, maxlen=max_length)
    mapping = tokenizer.rematch(text, tokens)
    start_mapping = {j[0]: i
                     for i, j in enumerate(mapping)
                     if j}
    end_mapping = {j[-1]: i
                   for i, j in enumerate(mapping)
                   if j}
    token_ids = tokenizer.tokens_to_ids(tokens)

    labels = np.zeros((len(categories_label2id), max_length, max_length))
    for tag in tags:
        start, end, label = tag['start'], tag['start'] + \
            len(tag['mention']) - 1, tag['category']
        if start in start_mapping and end in end_mapping:
            start = start_mapping[start]
            end = end_mapping[end]
            label = categories_label2id[label]
            labels[label, start, end] = 1
    labels = labels[:, :len(token_ids), :len(token_ids)]

    return token_ids, labels


def encode_sentences(text_list, tags_list, categories_label2id, max_length, tokenizer):

    token_ids_list, labels_list = [], []

    for text, tags in zip(text_list, tags_list):
        token_ids, labels = encode_text(text, tags, categories_label2id, max_length, tokenizer)

        token_ids_list.append(token_ids)
        labels_list.append(labels)

    return token_ids_list, labels_list


class NerDataset(Dataset):

    def __init__(self, args, data_generator, ner_labels, tokenizer, do_split=False):
        super().__init__()
        self.args = args
        self.data_generator = data_generator
        self.ner_labels = ner_labels
        self.tokenizer = tokenizer

        self.categories_label2id = {label: i
                                    for i, label in enumerate(ner_labels)}
        self.categories_d2label = {i: label
                                   for i, label in enumerate(ner_labels)}

        #  self.data = [(idx, text, tags) for idx, text, tags in data_generator()]
        self.data = [d for d in data_generator()]

        self.token_ids_list = []
        self.labels_list = []

        for d in self.data:
            idx, text, tags = d[:3]
            text_list = [text]
            tags_list = [tags]

            if self.args.do_split:
                sentences = split_sentences(text)
                assert "".join(sentences) == text, f"text:{text}, sentences: {sentences}"
                text_list.extend(sentences)

                sent_tags_list = split_text_tags(sentences, tags)
                tags_list.extend(sent_tags_list)
            assert len(text_list) == len(tags_list)

            token_ids_list, labels_list = encode_sentences(
                text_list, tags_list, self.categories_label2id, self.args.max_length, self.tokenizer
            )

            self.token_ids_list.extend(token_ids_list)
            self.labels_list.extend(labels_list)

    def __len__(self):
        #  return len(self.data)
        return len(self.token_ids_list)

    def __getitem__(self, index):
        token_ids_list, labels_list = self.token_ids_list[index], self.labels_list[index]

        return (token_ids_list, labels_list)

    #  def __getitem__(self, index):
    #      data = self.data[index]
    #      idx, text, tags = data
    #
    #      text_list = [text]
    #      tags_list = [tags]
    #
    #      if self.args.do_split:
    #          sentences = split_sentences(text)
    #          assert "".join(sentences) == text
    #          text_list.extend(sentences)
    #
    #          sent_tags_list = split_text_tags(sentences, tags)
    #          tags_list.extend(sent_tags_list)
    #      assert len(text_list) == len(tags_list)
    #
    #      # 第一行是全句，以下各行是分句列表
    #      token_ids_list, labels_list = encode_sentences(
    #          text_list, tags_list, self.categories_label2id, self.args.max_length, self.tokenizer
    #      )
    #
    #      return (token_ids_list, labels_list)
