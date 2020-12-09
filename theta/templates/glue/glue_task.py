#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json, os, re

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

#  glue_labels = ['现在', '过去', '将来', '其他']
glue_labels = ['现在', '过去', '将来']


def clean_text(text):
    if text:
        text = text.strip()
    return text


def train_data_generator(train_file=None):

    if train_file is None:
        train_file = 'data/data_train.json'

    json_data = json.load(open(train_file, 'r'))

    for i, x in enumerate(tqdm(json_data)):
        guid = f"{i}"
        text = clean_text(x['text'])
        label = "True"

        yield str(i), text, None, label


def test_data_generator(test_file=None):
    if test_file is None:
        test_file = 'data/sentences.json'

    json_data = json.load(open(test_file, 'r'))

    for i, x in enumerate(tqdm(json_data)):
        guid = f"{i}"
        text = clean_text(x['text'])
        yield guid, text, None, None


def generate_submission(args, reviews_file=None, submission_file=None):
    if reviews_file is None:
        reviews_file = args.reviews_file
    reviews = json.load(open(args.reviews_file, 'r'))

    if submission_file is None:
        submission_file = f"{args.submissions_dir}/{args.dataset_name}_submission_{args.local_id}.json"

    json.dump(reviews,
              open(submission_file, 'w'),
              ensure_ascii=False,
              indent=2)

    logger.info(f"Saved {len(reviews)} lines in {submission_file}")
