#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset

try:
    import rich

    def print(*args, **kwargs):
        rich.print(*args, **kwargs)


except:
    pass

from .utils import split_text_tags, split_sentences


def get_default_tokenizer(dict_path):
    from ..bert4torch.tokenizers import Tokenizer

    tokenizer = Tokenizer(dict_path, do_lower_case=True)

    return tokenizer


def encode_text(
    text, tags, entities_label2id, relations_label2id, max_length, tokenizer
):
    tokens = tokenizer.tokenize(text, maxlen=max_length)
    mapping = tokenizer.rematch(text, tokens)
    start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
    end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
    token_ids = tokenizer.tokens_to_ids(tokens)

    labels = np.zeros(
        (len(relations_label2id), len(entities_label2id), max_length, max_length)
    )
    for tag in tags:
        relation, entity_tags = tag["relation"], tag["entities"]

        relation = relations_label2id[relation]
        for tag in entity_tags:
            start, end, label = (
                tag["start"],
                tag["start"] + len(tag["mention"]) - 1,
                tag["category"],
            )
            if start in start_mapping and end in end_mapping:
                start = start_mapping[start]
                end = end_mapping[end]
                label = entities_label2id[label]
                labels[relation, label, start, end] = 1
    labels = labels[:, :, : len(token_ids), : len(token_ids)]

    return token_ids, labels


def encode_sentences(
    text_list, tags_list, entities_label2id, relations_label2id, max_length, tokenizer
):

    token_ids_list, labels_list = [], []

    for text, tags in zip(text_list, tags_list):
        token_ids, labels = encode_text(
            text, tags, entities_label2id, relations_label2id, max_length, tokenizer
        )

        token_ids_list.append(token_ids)
        labels_list.append(labels)

    return token_ids_list, labels_list


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

        self.token_ids_list = []
        self.labels_list = []

        for d in tqdm(self.data, desc="load dataset", ncols=160):
            idx, text, tags = d[:3]
            text = text.rstrip()
            text = text.replace("\n", " ")

            text_list = []
            tags_list = []
            if not self.args.do_split:
                text_list = [text]
                tags_list = [tags]
            else:
                #  print("idx:", idx)
                #  print("text", text)

                sentences = split_sentences(text)
                sents_text = "".join(sentences)
                #  print("sents_text:", sents_text)
                assert sents_text == text

                #  print("sentences:", sentences)
                #  print(f"full_tags:", tags)

                sent_rel_entities = []
                for x in tags:
                    rel = x["relation"]
                    entities = x["entities"]
                    #  print("rel:", rel, "entities:", entities)
                    sent_entities_list = split_text_tags(sentences, entities)
                    #  print("sent_entities_list:", sent_entities_list)
                    if len(sent_rel_entities) == 0:
                        for sent, ents in zip(sentences, sent_entities_list):
                            sent_rel_entities.append([sent, {rel: ents}])
                    else:
                        for j, (sent, ents) in enumerate(
                            zip(sentences, sent_entities_list)
                        ):
                            #  print("sent_entities:", sent_entities)
                            rel_ents = sent_rel_entities[j][1]  # .([sent, {rel: ents}])
                            if rel not in rel_ents:
                                rel_ents[rel] = []
                            rel_ents[rel].extend(ents)
                #  print("sent_rel_entities", sent_rel_entities)
                for j, x in enumerate(sent_rel_entities):
                    sre = sent_rel_entities[j]
                    sent_text, rel_ents = sre
                    tmp_tags = []
                    for rel, ents in rel_ents.items():
                        if len(ents) == 0:
                            continue
                        ents = sorted(ents, key=lambda x: x["start"])
                        tmp_tags.append({"relation": rel, "entities": ents})
                    text_list.append(sent_text)
                    tags_list.append(tmp_tags)

                #  print("text_list:", text_list)
                #  print("tags_list:", tags_list)

            assert len(text_list) == len(tags_list)

            token_ids_list, labels_list = encode_sentences(
                text_list,
                tags_list,
                self.entities_label2id,
                self.relations_label2id,
                self.args.max_length,
                self.tokenizer,
            )

            self.token_ids_list.extend(token_ids_list)
            self.labels_list.extend(labels_list)

    def __len__(self):
        #  return len(self.data)
        return len(self.token_ids_list)

    def __getitem__(self, index):
        token_ids_list, labels_list = (
            self.token_ids_list[index],
            self.labels_list[index],
        )

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
