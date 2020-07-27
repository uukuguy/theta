#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from tqdm import tqdm
from loguru import logger

submission_file = "./submissions/medical_entity_submission_01d49246c68511eaba9e4e0abd8d028e.json.txt"
lines = [line.strip() for line in open(submission_file, 'r')]
json_data = [json.loads(line) for line in lines]

for x in tqdm(json_data):
    entities = x['entities']
    new_entities = []
    for ent in entities:
        s = ent['start_pos']
        e = ent['end_pos']
        ent_len = e - s
        if ent_len >= 32:
            continue
        new_entities.append(ent)
    x['entities'] = new_entities

fixed_file = f"{submission_file}_fix.txt"
with open(fixed_file, 'w') as wt:
    for x in json_data:
        line = json.dumps(x, ensure_ascii=False)
        wt.write(f"{line}\n")
logger.info(f"Saved {fixed_file}")

#  json.dump(json_data,
#            open(f"{submission_file}_fix.txt", 'w'),
#            ensure_ascii=False)
