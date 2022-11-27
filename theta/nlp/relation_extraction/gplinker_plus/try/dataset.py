#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset

from .utils import split_text_tags, split_sentences
from ...bert4torch.utils import sequence_padding


def get_default_tokenizer(dict_path):
    from ...bert4torch.tokenizers import Tokenizer

    tokenizer = Tokenizer(dict_path, do_lower_case=True)

    return tokenizer


#  def encode_text(
#      text, tags, entities_label2id, relations_label2id, max_length, tokenizer
#  ):
#      def search(pattern, sequence):
#          """从sequence中寻找子串pattern
#          如果找到，返回第一个下标；否则返回-1。
#          """
#          n = len(pattern)
#          for i in range(len(sequence)):
#              if sequence[i : i + n] == pattern:
#                  return i
#          return -1
#
#      token_ids, segment_ids = tokenizer.encode(text, maxlen=max_length)
#
#      tokens = tokenizer.tokenize(text, maxlen=max_length)
#      assert len(tokens) == len(
#          token_ids
#      ), f"tokens: {len(tokens)}, {tokens}, token_ids: {len(token_ids)}, {token_ids}"
#      mapping = tokenizer.rematch(text, tokens)
#      start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
#      end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
#      # 整理三元组 {s: [(o, p)]}
#      spoes = set()
#      for tag in tags:
#          s, p, o = tag["subject"]["mention"], tag["predicate"], tag["object"]["mention"]
#
#          s = tokenizer.encode(s)[0][1:-1]
#          p = relations_label2id[p]
#          o = tokenizer.encode(o)[0][1:-1]
#
#          sh, oh = tag["subject"]["start"], tag["object"]["start"]
#          if sh in start_mapping and oh in start_mapping:
#              sh0 = start_mapping[sh]
#              oh0 = start_mapping[oh]
#          else:
#              sh0 = -1
#              oh0 = -1
#
#          sh = search(s, token_ids)
#          oh = search(o, token_ids)
#
#          if sh != -1 and oh != -1 and sh0 != -1 and oh0 != -1:
#              if sh != sh0 or oh != oh0:
#                  if (
#                      tokens[sh : sh + len(s)] != tokens[sh0 : sh0 + len(s)]
#                      or tokens[oh : oh + len(o)] != tokens[oh0 : oh0 + len(o)]
#                  ):
#
#                      print("-------------------")
#                      #  print("tokens:", tokens)
#                      print(
#                          "subject:",
#                          tag["subject"]["mention"],
#                          "object:",
#                          tag["object"]["mention"],
#                      )
#                      print(
#                          "search:",
#                          (sh, oh),
#                          tokens[sh : sh + len(s)],
#                          tokens[oh : oh + len(o)],
#                      )
#                      print(
#                          "my:",
#                          (sh0, oh0),
#                          tokens[sh0 : sh0 + len(s)],
#                          tokens[oh0 : oh0 + len(o)],
#                      )
#          sh, oh = sh0, oh0
#
#          if sh != -1 and oh != -1:
#              spoes.add((sh, sh + len(s) - 1, p, oh, oh + len(o) - 1))
#      # 构建标签
#      entity_labels = [set() for _ in range(2)]
#      head_labels = [set() for _ in range(len(relations_label2id))]
#      tail_labels = [set() for _ in range(len(relations_label2id))]
#      for sh, st, p, oh, ot in spoes:
#          entity_labels[0].add((sh, st))
#          entity_labels[1].add((oh, ot))
#          head_labels[p].add((sh, oh))
#          tail_labels[p].add((st, ot))
#      for label in entity_labels + head_labels + tail_labels:
#          if not label:  # 至少要有一个标签
#              label.add((0, 0))  # 如果没有则用0填充
#      # [subject/object=2, 实体个数, 实体起终点]
#      entity_labels = sequence_padding([list(l) for l in entity_labels])
#      # [关系个数, 该关系下subject/object配对数, subject/object起点]
#      head_labels = sequence_padding([list(l) for l in head_labels])
#      # [关系个数, 该关系下subject/object配对数, subject/object终点]
#      tail_labels = sequence_padding([list(l) for l in tail_labels])
#
#      return (
#          tokens,
#          mapping,
#          token_ids,
#          segment_ids,
#          entity_labels,
#          head_labels,
#          tail_labels,
#      )
#
#
#  def encode_sentences(
#      text_list, tags_list, entities_label2id, relations_label2id, max_length, tokenizer
#  ):
#
#      tokens_list, mappings_list = [], []
#      token_ids_list, segment_ids_list = [], []
#      entity_labels_list, head_labels_list, tail_labels_list = [], [], []
#      for text, tags in zip(text_list, tags_list):
#
#          (
#              tokens,
#              mapping,
#              token_ids,
#              segment_ids,
#              entity_labels,
#              head_labels,
#              tail_labels,
#          ) = encode_text(
#              text, tags, entities_label2id, relations_label2id, max_length, tokenizer
#          )
#
#          tokens_list.append(tokens)
#          mappings_list.append(mapping)
#          token_ids_list.append(token_ids)
#          segment_ids_list.append(segment_ids)
#          entity_labels_list.append(entity_labels)
#          head_labels_list.append(head_labels)
#          tail_labels_list.append(tail_labels)
#
#      return (
#          tokens_list,
#          mappings_list,
#          token_ids_list,
#          segment_ids_list,
#          entity_labels_list,
#          head_labels_list,
#          tail_labels_list,
#      )


class NerDataset(Dataset):
    def __init__(
        self,
        args,
        data_generator,
        entity_labels,
        relation_labels,
        tokenizer,
        do_split=False,
    ):
        super().__init__()
        self.args = args
        self.data_generator = data_generator
        self.entity_labels = entity_labels
        self.relation_labels = relation_labels
        self.tokenizer = tokenizer

        self.entities_label2id = {label: i for i, label in enumerate(entity_labels)}
        self.entities_id2label = {i: label for i, label in enumerate(entity_labels)}
        self.relations_label2id = {label: i for i, label in enumerate(relation_labels)}
        self.relations_id2label = {i: label for i, label in enumerate(relation_labels)}

        #  self.data = [(idx, text, tags) for idx, text, tags in data_generator()]
        self.data = [d for d in data_generator()]

        #  self.token_ids_list = []
        #  self.segment_ids_list = []
        #  self.entity_labels_list = []
        #  self.head_labels_list = []
        #  self.tail_labels_list = []
        #
        #  self.tokens_list = []
        #  self.mappings_list = []
        #
        #  for d in tqdm(self.data, desc="load dataset", ncols=160):
        #      idx, text, tags = d[:3]
        #
        #      (
        #          tokens,
        #          mapping,
        #          token_ids,
        #          segment_ids,
        #          entity_labels,
        #          head_labels,
        #          tail_labels,
        #      ) = encode_text(
        #          text,
        #          tags,
        #          self.entities_label2id,
        #          self.relations_label2id,
        #          self.args.max_length,
        #          self.tokenizer,
        #      )
        #
        #      self.tokens_list.append(tokens)
        #      self.mappings_list.append(mapping)
        #
        #      self.token_ids_list.append(token_ids)
        #      self.segment_ids_list.append(segment_ids)
        #
        #      self.entity_labels_list.append(entity_labels)
        #      self.head_labels_list.append(head_labels)
        #      self.tail_labels_list.append(tail_labels)
        #
        #  print(
        #      f"texts: {len(self.data)}, tokens: {len(self.tokens_list)}, mappings: {len(self.mappings_list)}, token_ids: {len(self.token_ids_list)}, segment_ids: {len(self.segment_ids_list)}, entity_labels: {len(self.entity_labels_list)}, head_labels: {len(self.head_labels_list)}, tail_labels: {len(self.tail_labels_list)}"
        #  )

    def __len__(self):
        return len(self.data)
        #  return len(self.token_ids_list)

    def __getitem__(self, i):

        return self.data[i]

        #  text = self.data[i][1]
        #  tokens = self.tokens_list[i]
        #  mapping = self.mappings_list[i]
        #
        #  token_ids = self.token_ids_list[i]
        #  segment_ids = self.segment_ids_list[i]
        #  entity_labels = self.entity_labels_list[i]
        #  head_labels = self.head_labels_list[i]
        #  tail_labels = self.tail_labels_list[i]
        #
        #  return (
        #      text,
        #      tokens,
        #      mapping,
        #      token_ids,
        #      segment_ids,
        #      entity_labels,
        #      head_labels,
        #      tail_labels,
        #  )
