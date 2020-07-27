#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from tqdm import tqdm
from loguru import logger
STEP = 2

#  events_0 = json.load(
#      open(
#          './outputs/f7b9ad82bb4411ea906ce8611f2e5a0e/medical_event_reviews_f7b9ad82bb4411ea906ce8611f2e5a0e.json',
#          'r'))
#  events_1 = json.load(
#      open(
#          './outputs/5692590cbc5911eaa90ce8611f2e5a0e/medical_event_reviews_5692590cbc5911eaa90ce8611f2e5a0e.json',
#          'r'))

#  STEP = 2
#
#  events_0 = json.load(
#      open(
#          './outputs/f7b9ad82bb4411ea906ce8611f2e5a0e/medical_event_reviews_f7b9ad82bb4411ea906ce8611f2e5a0e.json',
#          'r'))
#  events_1 = json.load(
#      open(
#          './outputs/515f9590b6e011eabe8de8611f2e5a0e/medical_event_reviews_515f9590b6e011eabe8de8611f2e5a0e.json',
#          'r'))

#  STEP = 3
#  events_0 = json.load(open('./submissions/merge_2_results.json', 'r'))
#  events_1 = json.load(
#      open(
#          './outputs/5692590cbc5911eaa90ce8611f2e5a0e/medical_event_reviews_5692590cbc5911eaa90ce8611f2e5a0e.json',
#          'r'))

#  STEP = 4
#  events_0 = json.load(open('./submissions/merge_3_results.json', 'r'))
#  events_1 = json.load(
#      open(
#          './outputs/c5bf0d78bc6a11eaa317e8611f2e5a0e/medical_event_reviews_c5bf0d78bc6a11eaa317e8611f2e5a0e.json',
#          'r'))

#  # 0.765544
#  reviews_0 = '2ad2020ec50a11eaa2f7e8611f2e5a0e'
#  events_0 = json.load(
#      open(f'./outputs/{reviews_0}/medical_event_reviews_{reviews_0}.json', 'r'))
#
#  # 0.760053
#  reviews_1 = '3fae0204c4e711eaaaaae8611f2e5a0e'
#  #  reviews_1 = '5d3103bec45411eaace7fa163e51b5c3'
#  events_1 = json.load(
#      open(f'./outputs/{reviews_1}/medical_event_reviews_{reviews_1}.json', 'r'))

# ------ online f1: 0.785288 -----
# 0.780739
reviews_0 = "ee620d52c5e311eab56ae8611f2e5a0e"
events_0 = json.load(
    open(f'./outputs/{reviews_0}/medical_event_reviews_{reviews_0}.json', 'r'))
# 0.765544
reviews_1 = '2ad2020ec50a11eaa2f7e8611f2e5a0e'
events_1 = json.load(
    open(f'./outputs/{reviews_1}/medical_event_reviews_{reviews_1}.json', 'r'))

reviews_2 = '5d3103bec45411eaace7fa163e51b5c3'
events_2 = json.load(
    open(f'./outputs/{reviews_2}/medical_event_reviews_{reviews_2}.json', 'r'))

reviews_3 = '3fae0204c4e711eaaaaae8611f2e5a0e'
events_3 = json.load(
    open(f'./outputs/{reviews_3}/medical_event_reviews_{reviews_3}.json', 'r'))


def is_identical(e0, e1):
    return e0['category'] == e1['category'] and e0['start'] == e1[
        'start'] and e0['end'] == e1['end']


def find_identical(es, e):
    for e0 in es:
        if is_identical(e0, e):
            return True
    return False


def merge_results(events_0, events_1):

    for guid, eevts_0 in tqdm(events_0.items()):
        eevts_1 = events_1[guid]
        entities_0 = eevts_0['entities']
        entities_1 = eevts_1['entities']

        new_entities = []
        for e_1 in entities_1:
            found = False
            for e_0 in entities_0:
                s0 = e_0['start']
                e0 = e_0['end']
                s1 = e_1['start']
                e1 = e_1['end']
                if s0 >= s1 and s0 <= e1:
                    found = True
                    continue
                if e0 >= s1 and e0 <= e1:
                    found = True
                    continue
                if s1 >= s0 and s1 <= e0:
                    found = True
                    continue
                if e1 >= s0 and e1 <= e0:
                    found = True
                    continue
            if not found:
                if not find_identical(new_entities, e_1):
                    new_entities.append(e_1)
        entities_0.extend(new_entities)
        eevts_0['entities'] = sorted(entities_0, key=lambda x: x['start'])

    return events_0


events_0 = merge_results(events_0, events_1)
#  events_0 = merge_results(events_0, events_2)
#  events_0 = merge_results(events_0, events_3)

merge_results_file = f"./submissions/merge_{reviews_0}_{reviews_1}_results.json"
#  merge_results_file = f"./submissions/merge_{reviews_0}_total_3_results.json"
json.dump(events_0,
          open(merge_results_file, 'w'),
          ensure_ascii=False,
          indent=2)
logger.info(f"Saved {merge_results_file}")


def generate_submission(submission_file):
    import xlwt
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet(f"medical_event")

    worksheet.write(0, 0, label='原文')
    worksheet.write(0, 1, label='肿瘤原发部位')
    worksheet.write(0, 2, label='原发病灶大小')
    worksheet.write(0, 3, label='转移部位')

    #  reviews_file = f"{args.latest_dir}/{args.dataset_name}_reviews_{args.local_id}.json"
    reviews_file = merge_results_file
    reviews = json.load(open(reviews_file, 'r'))

    idx = 1
    for guid, json_data in reviews.items():
        text = json_data['text']
        entities = json_data['entities']
        label_entities = {}
        for entity in entities:
            c = entity['category']
            s = entity['start']
            e = entity['end'] + 1
            entity_text = text[s:e]

            if s > len(text) or e > len(text):
                continue
            if len(entity_text) == 0 or len(entity_text) > 16:
                continue
            if ';' in entity_text or '、' in entity_text:
                continue

            if c not in label_entities:
                label_entities[c] = []
            label_entities[c].append(entity_text)

        worksheet.write(idx, 0, label=text)
        if '肿瘤部位' in label_entities:
            worksheet.write(idx, 1, ','.join(label_entities['肿瘤部位']))
        if '病灶大小' in label_entities:
            worksheet.write(idx, 2, ','.join(label_entities['病灶大小']))
        if '转移部位' in label_entities:
            worksheet.write(idx, 3, ','.join(label_entities['转移部位']))

        idx += 1

    workbook.save(submission_file)

    logger.info(f"Saved {submission_file}")


submission_file = f"./submissions/merge_{reviews_0}_{reviews_1}_results.xlsx"
#  submission_file = f"./submissions/merge_{reviews_0}_total_3_results.xlsx"
generate_submission(submission_file)
