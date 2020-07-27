#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json, re
from tqdm import tqdm
from loguru import logger
from theta.utils import load_json_file

ner_labels = ['疾病和诊断', '影像检查', '实验室检验', '手术', '药物', '解剖部位']
ner_connections = []


def clean_text(text):
    if text:
        text = text.strip()
    return text


def train_data_generator(train_file):

    lines = load_json_file(train_file)

    for i, x in enumerate(tqdm(lines)):
        guid = str(i)
        text = clean_text(x['originalText'])

        tags = []
        entities = x['entities']
        for entity in entities:
            start_pos = entity['start_pos']
            end_pos = entity['end_pos'] - 1
            category = entity['label_type']
            mention = text[start_pos:end_pos + 1]

            tags.append({
                'category': category,
                'start': start_pos,
                'mention': mention
            })

        yield str(i), text, None, tags


def test_data_generator(test_file):
    if test_file is None:
        test_file = './data/test1.txt'

    lines = load_json_file(test_file)
    for i, s in enumerate(tqdm(lines)):
        guid = str(i)
        text_a = clean_text(s['originalText'])

        yield guid, text_a, None, None


def generate_submission(args):
    reviews_file = f"{args.latest_dir}/{args.dataset_name}_reviews_{args.local_id}.json"
    reviews = json.load(open(reviews_file, 'r'))

    submission_file = f"{args.submissions_dir}/{args.dataset_name}_submission_{args.local_id}.json.txt"
    with open(submission_file, 'w') as wt:
        for guid, json_data in reviews.items():
            output_data = {'originalText': json_data['text'], 'entities': []}
            for json_entity in json_data['tags']:

                c = json_entity['category']
                s = json_entity['start']
                m = json_entity['mention']
                e = s + len(m)

                ent_len = e - s + 1
                if ent_len >= 32:
                    continue

                output_data['entities'].append({
                    'label_type': c,
                    'overlap':
                    0,
                    'start_pos': s,
                    'end_pos': e
                })
            output_data['entities'] = sorted(output_data['entities'],
                                             key=lambda x: x['start_pos'])
            output_string = json.dumps(output_data, ensure_ascii=False)
            wt.write(f"{output_string}\n")

    logger.info(f"Saved {len(reviews)} lines in {submission_file}")
