#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json, re
from tqdm import tqdm
from loguru import logger

ner_labels = []
ner_connections = []


def clean_text(text):
    if text:
        text = text.strip()
    return text


def train_data_generator(train_file):
    # 标准theta ner文件格式
    from theta.modeling import ner_data_generator
    for guid, text, _, tags in ner_data_generator(train_file):
        # 逐行输出guid, text, tags
        # tags格式: {'category': 'c', 'start': s, 'mention': m}
        yield guid, text, None, tags


def test_data_generator(test_file):
    from theta.modeling import ner_data_generator
    for guid, text, _, _ in ner_data_generator(train_file):
        # 逐行输出guid, text
        yield guid, text, None, None


def generate_submission(args):
    reviews = json.load(open(args.reviews_file, 'r'))

    submission_file = f"{args.submissions_dir}/{args.dataset_name}_submission_{args.local_id}.json"

    final_result = {}
    for guid, json_data in reviews.items():
        entities = []
        for json_entity in json_data['tags']:
            c = json_entity['category']
            s = json_entity['start']
            m = json_entity['mention']
            e = s + len(m)

            entities.append({
                'label_type': c,
                'overlap': 0,
                'start_pos': s,
                'end_pos': e
            })

        final_result[guid] = entities

    json.dump(final_result,
              open(submission_file, 'w'),
              ensure_ascii=False,
              indent=2)

    logger.info(f"Saved {len(reviews)} lines in {submission_file}")

    return submission_file
