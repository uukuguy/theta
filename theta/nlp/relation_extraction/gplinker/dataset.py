#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset

from ...bert4torch.utils import sequence_padding
from ..tagging import TaskLabels, TaskTag, SubjectTag, ObjectTag, TaggedData
from .utils import split_text_tags, split_sentences


def get_default_tokenizer(dict_path):
    from ...bert4torch.tokenizers import Tokenizer

    tokenizer = Tokenizer(dict_path, do_lower_case=True)

    return tokenizer


def encode_text(text, tags, task_labels, max_length, tokenizer):
    entities_label2id, relations_label2id = (
        task_labels.entities_label2id,
        task_labels.relations_label2id,
    )

    def search(pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i : i + n] == pattern:
                return i
        return -1

    token_ids, segment_ids = tokenizer.encode(text, maxlen=max_length)

    tokens = tokenizer.tokenize(text, maxlen=max_length)
    assert len(tokens) == len(
        token_ids
    ), f"tokens: {len(tokens)}, {tokens}, token_ids: {len(token_ids)}, {token_ids}"
    mapping = tokenizer.rematch(text, tokens)
    start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
    end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
    # 整理三元组 {s: [(o, p)]}
    spoes = set()
    for tag in tags:
        sh, oh = tag.s.s, tag.o.s
        s_m, o_m = tag.s.m, tag.o.m

        st = sh + len(s_m)
        ot = oh + len(o_m)
        assert text[sh:st] == s_m
        assert text[oh:ot] == o_m

        if sh in start_mapping and oh in start_mapping:
            sh = start_mapping[sh]
            oh = start_mapping[oh]
        else:
            sh = -1
            oh = -1

        #  sh = search(s, token_ids)
        #  oh = search(o, token_ids)
        #
        #  if sh != -1 and oh != -1 and sh0 != -1 and oh0 != -1:
        #      if sh != sh0 or oh != oh0:
        #          if (
        #              tokens[sh : sh + len(s)] != tokens[sh0 : sh0 + len(s)]
        #              or tokens[oh : oh + len(o)] != tokens[oh0 : oh0 + len(o)]
        #          ):
        #
        #              print("-------------------")
        #              #  print("tokens:", tokens)
        #              print(
        #                  "subject:",
        #                  tag["subject"]["mention"],
        #                  "object:",
        #                  tag["object"]["mention"],
        #              )
        #              print(
        #                  "search:",
        #                  (sh, oh),
        #                  tokens[sh : sh + len(s)],
        #                  tokens[oh : oh + len(o)],
        #              )
        #              print(
        #                  "my:",
        #                  (sh0, oh0),
        #                  tokens[sh0 : sh0 + len(s)],
        #                  tokens[oh0 : oh0 + len(o)],
        #              )

        if sh != -1 and oh != -1:
            s, p, o = tag.s, tag.p, tag.o
            s = tokenizer.encode(s.m)[0][1:-1]
            p = relations_label2id[p]
            o = tokenizer.encode(o.m)[0][1:-1]

            st = sh + len(s) - 1
            ot = oh + len(o) - 1
            if (
                sh >= max_length
                or st >= max_length
                or oh >= max_length
                or ot >= max_length
            ):
                print(
                    "!!! tag overlap max_length",
                    max_length,
                    "text:",
                    text,
                    "tag:",
                    tag,
                    "s:",
                    (sh, st),
                    "o:",
                    (oh, ot),
                )
            else:
                spoes.add((sh, st, p, oh, ot))
    # 构建标签
    entity_labels = [set() for _ in range(2)]
    head_labels = [set() for _ in range(len(relations_label2id))]
    tail_labels = [set() for _ in range(len(relations_label2id))]
    for sh, st, p, oh, ot in spoes:
        entity_labels[0].add((sh, st))
        entity_labels[1].add((oh, ot))
        head_labels[p].add((sh, oh))
        tail_labels[p].add((st, ot))
    for label in entity_labels + head_labels + tail_labels:
        if not label:  # 至少要有一个标签
            label.add((0, 0))  # 如果没有则用0填充
    # [subject/object=2, 实体个数, 实体起终点]
    entity_labels = sequence_padding([list(l) for l in entity_labels])
    # [关系个数, 该关系下subject/object配对数, subject/object起点]
    head_labels = sequence_padding([list(l) for l in head_labels])
    # [关系个数, 该关系下subject/object配对数, subject/object终点]
    tail_labels = sequence_padding([list(l) for l in tail_labels])

    return (
        (tokens, mapping),
        (token_ids, segment_ids),
        (entity_labels, head_labels, tail_labels),
    )


def encode_sentences(text_tags_list, task_labels, max_length, tokenizer):

    tokens_list, mappings_list = [], []
    token_ids_list, segment_ids_list = [], []
    entity_labels_list, head_labels_list, tail_labels_list = [], [], []
    for text, tags in tqdm(text_tags_list):

        (
            (tokens, mapping),
            (token_ids, segment_ids),
            (entity_labels, head_labels, tail_labels),
        ) = encode_text(text, tags, task_labels, max_length, tokenizer)

        tokens_list.append(tokens)
        mappings_list.append(mapping)

        token_ids_list.append(token_ids)
        segment_ids_list.append(segment_ids)

        entity_labels_list.append(entity_labels)
        head_labels_list.append(head_labels)
        tail_labels_list.append(tail_labels)

    return (
        (tokens_list, mappings_list),
        (token_ids_list, segment_ids_list),
        (entity_labels_list, head_labels_list, tail_labels_list),
    )


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

        (
            (tokens_list, mappings_list),
            (token_ids_list, segment_ids_list),
            (entity_labels_list, head_labels_list, tail_labels_list),
        ) = encode_sentences(
            text_tags_list, self.args.task_labels, self.args.max_length, self.tokenizer
        )
        self.tokens_list, self.mappings_list = tokens_list, mappings_list
        self.token_ids_list, self.segment_ids_list = token_ids_list, segment_ids_list
        self.entity_labels_list, self.head_labels_list, self.tail_labels_list = (
            entity_labels_list,
            head_labels_list,
            tail_labels_list,
        )

        print(
            f"texts: {len(self.data)}, tokens: {len(self.tokens_list)}, mappings: {len(self.mappings_list)}, token_ids: {len(self.token_ids_list)}, segment_ids: {len(self.segment_ids_list)}, entity_labels: {len(self.entity_labels_list)}, head_labels: {len(self.head_labels_list)}, tail_labels: {len(self.tail_labels_list)}"
        )

    def __len__(self):
        return len(self.token_ids_list)

    def __getitem__(self, i):

        tagged_data = self.data[i]

        tokens = self.tokens_list[i]
        mapping = self.mappings_list[i]

        token_ids = self.token_ids_list[i]
        segment_ids = self.segment_ids_list[i]

        entity_labels = self.entity_labels_list[i]
        head_labels = self.head_labels_list[i]
        tail_labels = self.tail_labels_list[i]

        return (
            tagged_data,
            (tokens, mapping),
            (token_ids, segment_ids),
            (entity_labels, head_labels, tail_labels),
        )
