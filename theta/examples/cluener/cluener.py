#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json, os, re

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

ner_labels = [
    'address', 'book', 'company', 'game', 'government', 'movie', 'name',
    'organization', 'position', 'scene'
]
ner_connections = []
logger.info(f"ner_labels: {ner_labels}")


def clean_text(text):
    if text:
        text = text.strip()
    return text


def load_cluener_data(train_file):

    from theta.utils import load_json_file
    lines = load_json_file(train_file)

    for i, x in enumerate(tqdm(lines)):
        guid = f"train-{i}"
        text = clean_text(x['text'])

        # -------------------- 训练数据json格式 --------------------
        #  {
        #      "text": "万通地产设计总监刘克峰；",
        #      "label": {
        #          "name": {
        #              "刘克峰": [[8, 10]]
        #          },
        #          "company": {
        #              "万通地产": [[0, 3]]
        #          },
        #          "position": {
        #              "设计总监": [[4, 7]]
        #          }
        #      }
        #  }

        tags = []
        labels = x['label']
        classes = labels.keys()
        for c in classes:
            c_labels = labels[c]
            #  logger.debug(f"c_labels:{c_labels}")
            for label, span in c_labels.items():
                s, e = span[0]
                m = text[s:e + 1]
                tags.append({'category': c, 'start': s, 'mention': m})

        yield guid, text, None, tags


def train_data_generator(train_file):
    if train_file is None:
        train_file = 'data/train.json'

    for guid, text, _, tags in load_cluener_data(train_file):
        yield guid, text, None, tags
    #  # 标准theta ner文件格式
    #  from theta.modeling import ner_data_generator
    #  for guid, text, _, tags in ner_data_generator(train_file):
    #      # 逐行输出guid, text, tags
    #      # tags格式: {'category': 'c', 'start': s, 'mention': m}
    #      yield guid, text, None, tags


def test_data_generator(test_file):
    if test_file is None:
        test_file = 'data/test.json'

    for i, line in enumerate(
            tqdm(open(test_file).readlines(), desc="Test data: ")):
        guid = f"{i}"
        json_data = json.loads(line.strip())
        text = clean_text(json_data['text'])

        yield guid, text, None, None


def generate_submission(args):
    reviews_file = args.reviews_file
    reviews = json.load(open(reviews_file, 'r'))

    submission_file = f"./submissions/{args.dataset_name}_predict.json"
    test_results = []
    for guid, json_data in tqdm(reviews.items()):
        text = json_data['text']

        classes = {}
        for json_entity in json_data['tags']:
            c = json_entity['category']
            s = json_entity['start']
            m = json_entity['mention']
            if c not in classes:
                classes[c] = {}
            if m not in classes[c]:
                classes[c][m] = []
            classes[c][m].append([s, s + len(m) - 1])
        test_results.append({'id': guid, 'text': text, 'label': classes})

    with open(submission_file, 'w') as wt:
        for line in test_results:
            wt.write(f"{json.dumps(line, ensure_ascii=False)}\n")

    logger.info(f"Saved {len(reviews)} lines in {submission_file}")


def eval_data_generator(eval_file):
    if eval_file is None:
        eval_file = 'data/eval.json'
    raise NotImplementedError


def evaluate(dev_file, reviews_file):
    from theta.modeling import ner_evaluate
    eval_results = ner_evaluate(dev_file, reviews_file, eval_data_generator)
    logger.warning(f"eval_results: {eval_results}")


if __name__ == '__main__':
    import sys
    dev_file = sys.argv[1]
    reviews_file = sys.argv[2]

    evaluate(dev_file, reviews_file)
