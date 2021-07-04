#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, re, json
from loguru import logger
from tqdm import tqdm

glue_labels = []
assert len(glue_labels) > 0


def clean_text(text):
    #  content = re.sub("<.*?>", '', content)
    #  content = re.sub("[\s]", '', content)
    #  content = re.sub("[\n\t]", ' ', content)
    #  content = re.sub("[\n\t]", '。', content)

    #  text = title + content
    #  text = re.sub("[^0-9\u4e00-\u9fa5]", '', text)
    return text


def train_data_generator():
    yield "guid", "text_a", "text_b", "label"


def test_data_generator():
    yield "guid", "text", None, None


def prepare_samples():
    """
    准备训练集和测试集数据
    """

    from theta.nlp.data.samples import GlueSamples
    from theta.nlp.data.data_utils import yield_data_generator
    train_samples = GlueSamples(labels_list=glue_labels,
                                data_generator=train_data_generator)
    test_samples = GlueSamples(labels_list=glue_labels,
                               data_generator=test_data_generator)

    return train_samples, test_samples
