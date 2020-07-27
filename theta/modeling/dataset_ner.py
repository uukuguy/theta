#!/usr/bin/env python
# -*- coding: utf-8 -*-

# NER数据集持久化，与标注文件之间的互转。

import os, json, re
from dataclasses import dataclass, field
from typing import List
from tqdm import tqdm
from loguru import logger


# 实体标签
@dataclass
class EntityTag:
    # 标签类别
    category: str = ""
    # 标签起始位置
    start: int = -1
    # 标签文本
    mention: str = ""

    def clear(self):
        pass

    def to_dict(self):
        return {
            'category': self.category,
            'start': self.start,
            'mention': self.mention
        }

    def from_dict(self, dict_data):
        self.clear()
        for k in self.__dict__.keys():
            if k in dict_data:
                v = dict_data[k]
                setattr(self, k, v)
        return self


# 打过标签的文本
@dataclass
class TaggedText:
    guid: str
    text: str
    tags: List[EntityTag] = field(default_factory=list)

    def clear(self):
        entities = []

    def to_dict(self):
        return {
            'guid': self.guid,
            'text': self.text,
            'tags': [x.to_dict() for x in self.tags]
        }

    def from_dict(self, dict_data):
        self.clear()
        if 'guid' in dict_data:
            self.guid = dict_data['guid']
        elif 'text' in dict_data:
            self.text = dict_data['text']
        elif 'tags' in dict_data:
            for x in dict_data['tags']:
                self.tags.append(EntityTag().from_dict(x))

        return self

    def add_tag(self, tag: EntityTag):
        self.tags.append(tag)


# 实体标签数据集
class NerDataset:
    def __init__(self, name, ner_labels=None, ner_connections=None):
        self.name = name
        self.ner_labels = ner_labels
        self.ner_connections = ner_connections
        self.tagged_text_list: List[TaggedText] = []

    def __len__(self):
        return len(self.tagged_text_list)

    def __item__(self, idx):
        return self.tagged_text_list[idx]

    def __iter__(self):
        for x in self.tagged_text_list:
            yield x

    def info(self):
        logger.info(f"Total: {len(self.tagged_text_list)}")

    def append(self, tagged_text: TaggedText):
        self.tagged_text_list.append(tagged_text)

    def extend(self, other_dataset):
        self.tagged_text_list.extend(other_dataset)

    def save(self, filename: str):
        """
        {'guid': '0000', 'text': "sample text", 'tags': [{'category': "实体类别1", start: 10, mention: "实体文本"}, ...]}
        """
        with open(filename, 'w') as wt:
            for tagged_text in tqdm(self.tagged_text_list,
                                    desc="Save tagged text list."):
                data_dict = tagged_text.to_dict()
                wt.write(f"{json.dumps(data_dict, ensure_ascii=False)}\n")
        logger.info(f"Saved {filename}")

    def load_from_file(self, filename: str):
        logger.info(f"Loading {filename}")
        lines = [
            json.loads(x.strip()) for x in open(filename, 'r').readlines()
        ]

        def data_generator(data_source=None):
            for json_data in lines:
                yield json_data['guid'], json_data['text'], None, json_data[
                    'tags']

        return self.load(data_generator)

    def load_from_brat_data(self, brat_data_dir):
        from .brat import brat_data_generator
        return self.load(brat_data_generator, brat_data_dir)

    def load(self, data_generator, data_source=None):
        for guid, text, _, json_tags in data_generator(data_source):
            tagged_text = TaggedText(guid, text)
            for json_tag in json_tags:
                entity_tag = EntityTag().from_dict(json_tag)
                tagged_text.add_tag(entity_tag)
            self.tagged_text_list.append(tagged_text)
        return self

    def export_to_brat_files(self, brat_data_dir, max_pages=10):
        from .brat import export_brat_files
        export_brat_files(self.tagged_text_list,
                          self.ner_labels,
                          self.ner_connections,
                          brat_data_dir,
                          max_pages=max_pages)

    def import_from_brat_files(self, brat_data_dir):
        from .brat import import_brat_files
        import_brat_files(self.tagged_text_list, brat_data_dir)

    def to_poplar_file(self, poplar_file):
        from .poplar import save_poplar_file
        save_poplar_file(poplar_file,
                         self.ner_labels,
                         self.ner_connections,
                         start_page=0,
                         max_pages=100)

    def append_from_poplar_file(self, poplar_file):
        from .poplar import poplar_data_generator
        return self.load(poplar_data_generator(poplar_file))


def ner_data_generator(train_file, dataset_name=None, ner_labels=None, ner_connections=None):
    ner_dataset = NerDataset("kgcs_entities", ner_labels, ner_connections)
    ner_dataset.load_from_file(train_file)
    for tagged_text in tqdm(ner_dataset):
        guid = tagged_text.guid
        text = tagged_text.text
        tags = [x.to_dict() for x in tagged_text.tags]
        yield guid, text, None, tags


if __name__ == '__main__':
    pass
