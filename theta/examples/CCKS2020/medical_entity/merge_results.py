#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from tqdm import tqdm
from loguru import logger
import numpy as np

# 1
#  files = [
#      # 0.850367
#      "./submissions/medical_entity_submission_04e39028c6cf11ea971e4e0abd8d028e.json.txt",
#      # 0.846762
#      "./submissions/medical_entity_submission_42bb5158b83111ea9e89e8611f2e5a0e.json.txt",
#      # 0.849197
#      "./submissions/medical_entity_submission_5013151cc6ae11eaa7894e0abd8d028e.json.txt",
#  ]

# 2
files = [
    # 0.850367
    "./submissions/medical_entity_submission_04e39028c6cf11ea971e4e0abd8d028e.json.txt",
    # 0.849187
    "./submissions/medical_entity_submission_4ce45136c73911eab73d4e0abd8d028e.json.txt",
    # 0.849197
    "./submissions/medical_entity_submission_5013151cc6ae11eaa7894e0abd8d028e.json.txt",
]
num_files = len(files)

json_datas = []
for file in files:
    logger.debug(f"{file}")
    lines = [json.loads(x.strip()) for x in open(file, 'r').readlines()]
    json_datas.append(lines)


def is_identical(ent0, ent1):
    return ent0['label_type'] == ent1['label_type'] and ent0[
        'start_pos'] == ent1['start_pos'] and ent0['end_pos'] == ent1['end_pos']


def find_ent(entities, ent):
    for x in entities:
        if is_identical(x, ent):
            return True
    return False


def merge_duplicate_entities():
    final_events = []
    for events in tqdm(zip(*json_datas)):
        #  logger.info(f"events: {events}")
        evt_entities = [evt['entities'] for evt in events]

        evt_entities = [x for z in evt_entities for x in z]

        evt_entities = sorted(evt_entities, key=lambda x: x['start_pos'])

        remain_entities = []
        duplicate_entities = []
        for i, ent in enumerate(evt_entities):
            found = False
            for j in range(i + 1, len(evt_entities)):
                ent1 = evt_entities[j]
                if ent1['start_pos'] > ent['start_pos']:
                    break
                if is_identical(ent, ent1):
                    found = True
                    if not find_ent(duplicate_entities, ent):
                        duplicate_entities.append(ent)
                    break
            if not found:
                remain_entities.append(ent)

        final_evt = events[0]
        final_evt['entities'] = duplicate_entities
        final_events.append(final_evt)

    final_results_file = './submissions/medical_entity_merge_duplicate_entities_2.json.txt'
    with open(final_results_file, 'w') as wt:
        for evt in final_events:
            wt.write(f"{json.dumps(evt, ensure_ascii=False)}\n")
    logger.info(f"Saved {final_results_file}")


def merge_all_identical_entities():
    final_events = []
    for events in tqdm(zip(*json_datas)):
        #  logger.info(f"events: {events}")
        evt_entities = [evt['entities'] for evt in events]

        evt_entities = [x for z in evt_entities for x in z]

        evt_entities = sorted(evt_entities, key=lambda x: x['start_pos'])

        remain_entities = []
        duplicate_entities = []
        for i, ent in enumerate(evt_entities):
            num_identical = 0
            found = False
            for j in range(i + 1, len(evt_entities)):
                ent1 = evt_entities[j]
                if ent1['start_pos'] > ent['start_pos']:
                    break
                if is_identical(ent, ent1):
                    found = True
                    num_identical += 1

            if not found:
                remain_entities.append(ent)
            else:
                if num_identical >= num_files - 1:
                    if not find_ent(duplicate_entities, ent):
                        duplicate_entities.append(ent)

        final_evt = events[0]
        final_evt['entities'] = duplicate_entities
        final_events.append(final_evt)

    final_results_file = './submissions/medical_entity_merge_all_identical_entities.json.txt'
    with open(final_results_file, 'w') as wt:
        for evt in final_events:
            wt.write(f"{json.dumps(evt, ensure_ascii=False)}\n")
    logger.info(f"Saved {final_results_file}")


merge_duplicate_entities()
#  merge_all_identical_entities()
