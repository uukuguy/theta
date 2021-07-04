#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, re, json
from loguru import logger
from tqdm import tqdm

ner_labels = []
assert len(ner_labels) > 0


def clean_text(text):
    #  content = re.sub("<.*?>", '', content)
    #  content = re.sub("[\s]", '', content)
    #  content = re.sub("[\n\t]", ' ', content)
    #  content = re.sub("[\n\t]", '。', content)

    #  text = title + content
    #  text = re.sub("[^0-9\u4e00-\u9fa5]", '', text)
    return text


def train_data_generator():
    train_data_file = "./data/rawdata/train.json"
    lines = open(train_data_file).readlines()
    lines = [line.strip() for line in lines]
    for idx, line in enumerate(tqdm(lines, desc=f"{train_data_file}")):
        json_data = json.loads(line)
        guid = json_data['doc_id']
        text = json_data['text']
        text = clean_text(text)

        tags = [{
            'category': 'category',
            'start': 0,
            'mention': 'mention',
        }]

        yield guid, text, None, tags


def test_data_generator():
    test_data_file = "./data/rawdata/test.json"
    lines = open(test_data_file).readlines()
    lines = [line.strip() for line in lines]
    for idx, line in enumerate(tqdm(lines, desc=f"{test_data_file}")):
        json_data = json.loads(line)
        guid = json_data['doc_id']
        text = json_data['text']
        text = clean_text(text)

        yield guid, text, None, None


def prepare_samples():
    """
    准备训练集和测试集数据
    """

    from theta.nlp.data.samples import EntitySamples
    from theta.nlp.data.data_utils import yield_data_generator
    train_samples = EntitySamples(labels_list=ner_labels,
                                  data_generator=train_data_generator)
    test_samples = EntitySamples(labels_list=ner_labels,
                                 data_generator=test_data_generator)

    return train_samples, test_samples
