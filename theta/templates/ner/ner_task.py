#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import re

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

ner_labels = []
ner_connections = []


def clean_text(text):
    if text:
        text = text.strip()
    return text


def train_data_generator(train_file):
    if train_file is None:
        train_file = 'data/task2_train_reformat.tsv'

    # 标准theta ner文件格式
    from theta.modeling import ner_data_generator
    for guid, text, _, tags in ner_data_generator(train_file):
        # 逐行输出guid, text, tags
        # tags格式: {'category': 'c', 'start': s, 'mention': m}
        yield guid, text, None, tags

    #  lines = read_train_file(train_file)
    #
    #  for i, x in enumerate(tqdm(lines)):
    #      guid = f"{i}"
    #      text = clean_text(x['originalText'])
    #      entities = x['entities']
    #
    #      tags = []
    #      for entity in entities:
    #          start_pos = entity['start_pos']
    #          end_pos = entity['end_pos']
    #          category = entity['label_type']
    #          mention = text[start_pos:end_pos]
    #
    #          tags.append({
    #              'category': category,
    #              'start': start_pos,
    #              'mention': mention
    #          })
    #
    #      yield str(i), text, None, tags


def test_data_generator(test_file):
    if test_file is None:
        test_file = 'data/test.tsv',

    from theta.modeling import ner_data_generator
    for guid, text, _, _ in ner_data_generator(train_file):
        # 逐行输出guid, text
        yield guid, text, None, None


def generate_submission(args):
    reviews = json.load(open(args.reviews_file, 'r'))
    submission_file = f"{args.submissions_dir}/{args.dataset_name}_submission_{args.local_id}.json"

    json.dump(reviews,
              open(submission_file, 'w'),
              ensure_ascii=False,
              indent=2)

    logger.info(f"Saved {len(reviews)} lines in {submission_file}")
