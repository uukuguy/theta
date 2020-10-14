#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json, os, re

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
        train_file = 'data/train.json'

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
