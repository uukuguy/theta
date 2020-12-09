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
                m = text[s:e+1]
                tags.append({
                    'category': c,
                    'start': s,
                    'mention': m
                }
                )

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

    from theta.modeling import ner_data_generator
    for guid, text, _, _ in ner_data_generator(train_file):
        # 逐行输出guid, text
        yield guid, text, None, None


def generate_submission(args, reviews_file=None, submission_file=None):
    if reviews_file is None:
        reviews_file = args.reviews_file
    reviews = json.load(open(reviews_file, 'r'))

    if submission_file is None:
        submission_file = f"{args.submissions_dir}/{args.dataset_name}_submission_{args.local_id}.json"

    #  json.dump(reviews,
    #            open(submission_file, 'w'),
    #            ensure_ascii=False,
    #            indent=2)
    from collections import defaultdict
    entities = defaultdict(list)
    from theta.modeling import ner_data_generator
    for guid, text, _, tags in tqdm(ner_data_generator(reviews_file)):
        for tag in tags:
            c = tag['category']
            s = tag['start']
            m = tag['mention']
            if len(m) <= 32:
                entities[c].append(m)

    for c, ents in entities.items():
        entities[c] = sorted(list(set(ents)))

    json.dump({'entities': entities},
              open(submission_file, 'w'),
              ensure_ascii=False,
              indent=2)

    logger.info(f"Saved {len(reviews)} lines in {submission_file}")


def eval_data_generator(eval_file):
    if eval_file is None:
        eval_file = 'data/eval.json'
    raise NotImplementedError


def evaluate(dev_file, reviews_file):
    from theta.modeling import ner_evaluate
    macro_acc, macro_recall, macro_f1, micro_acc, micro_recall, micro_f1 = ner_evaluate(
        dev_file, reviews_file, eval_data_generator)


if __name__ == '__main__':
    import sys
    dev_file = sys.argv[1]
    reviews_file = sys.argv[2]

    evaluate(dev_file, reviews_file)
