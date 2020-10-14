#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json, os, re
from copy import deepcopy

from loguru import logger
from tqdm import tqdm

predicate_labels = []


def load_data(filename):
    D = []
    with open(filename, 'r', encoding='utf8') as f:
        pbar = tqdm()
        for l in f:
            l = json.loads(l)
            d = {'text': l['text'], 'spo_list': []}
            pbar.update()
            for spo in l['spo_list']:
                s = spo['subject']
                p = spo['predicate']
                o = spo['object']['@value']
                o_type = spo['object_type']['@value']
                s_type = spo['subject_type']
                p_label = f"{s_type}_{p}_{o_type}"
                d['spo_list'].append((s, p_label, o))
                #  pbar.set_description(f"({s}, {p_label}, {o})")
            D.append(d)
    return D


def clean_text(text):
    if text:
        text = text.strip()
    return text


def fix_span(text):
    # ex: C++Builder网络开发实例（附光盘）/计算机开发与制作实例丛书
    # ex: 9月11日晚，*ST大控公告
    #  b_text = deepcopy(text)
    #  text = text.strip()
    chars = ['+', '*', '[', ']', '-', '\\', '(', ')']
    for c in chars:
        text = text.replace(c, f"\{c}")
    #  if b_text != text:
    #      logger.info(f"b_text: {b_text}, text: {text}")
    return text


def train_data_generator(train_file):
    if train_file is None:
        train_file = "./data/train_data.json"

    for i, d in enumerate(tqdm(D)):
        guid = f"{i}"
        text = clean_text(d['text'])

        spo_list = d['spo_list']
        tags = []
        for sub, predicate, obj in spo_list:
            tags.append(
                ((s_start, s_mention), predicate, (o_start, o_mention)))
        if tags:
            yield guid, text, None, tags
        #  if i > 20000:
        #      break

    logger.warning(
        f"multi_subs: {multi_subs}/{total_subs}({multi_subs/total_subs:.2f}), multi_objs: {multi_objs}/{total_objs}({multi_objs/total_objs:.2f})"
    )


def test_data_generator(test_file):
    if test_file is None:
        test_file = "./data/test1.json"

    with open(test_file, 'r', encoding='utf8') as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    for guid, l in enumerate(tqdm(lines, desc="Test")):
        guild = f"{guid}"
        json_data = json.loads(l)
        text = clean_text(json_data['text'])
        yield guid, text, None, None


def generate_submission(args):
    reviews = json.load(open(args.reviews_file, 'r'))

    submission_file = f"{args.submissions_dir}/{args.dataset_name}_submission_{args.local_id}.json"

    json.dump(reviews,
              open(submission_file, 'w'),
              ensure_ascii=False,
              indent=2)

    logger.info(f"Saved {len(reviews)} lines in {submission_file}")


def main():
    pass


if __name__ == '__main__':
    main()
