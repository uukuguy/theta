#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, time
from pathlib import Path
from loguru import logger
from .trainer import generate_dataloader
from .onnx import export_onnx, inference_from_onnx
from .ner_utils import LabeledText, show_ner_datainfo, get_ner_preds_reviews, save_ner_preds, load_ner_examples, load_ner_labeled_examples
from .glue_utils import show_glue_datainfo, load_glue_examples
from .common_args import add_common_args


def augement_entities(all_text_entities, labels_map):
    aug_tokens = []
    for i, (guid, text, entities) in enumerate(
            tqdm(all_text_entities, desc=f"Augement {num_augements}X")):

        #  print(f"-------------------{json_file}--------------------")
        #  print(text)
        #  print(entities)
        #  for entity in entities:
        #      s = entity['start_pos']
        #      e = entity['end_pos']
        #      print(f"{entity['label_type']}: {text[s:e]}")
        #  print("----------------------------------------")
        if entities:
            for ai in range(num_augements):
                e_idx = random.randint(0, len(entities) - 1)
                entity = entities[e_idx]

                label_type = entity['label_type']
                s = entity['start_pos']
                e = entity['end_pos']

                labels = labels_map[label_type]
                idx = random.randint(0, len(labels) - 1)
                new_entity_text = labels[idx]

                text = text[:s] + new_entity_text + text[e:]

                assert len(new_entity_text) >= 0
                delta = len(new_entity_text) - (e - s)

                entity['end_pos'] = entity['start_pos'] + len(new_entity_text)
                entity['mention'] = new_entity_text

                assert text[
                    entity['start_pos']:entity['end_pos']] == new_entity_text

                for n, e in enumerate(entities):
                    if n > e_idx:
                        e['start_pos'] += delta
                        e['end_pos'] += delta

                aug_tokens.append(
                    (f"{guid}-a{ai}", text, copy.deepcopy(entities)))

    #  for guid, text, entities in aug_tokens:
    #      text_a = text
    #      for entity in entities:
    #          logger.debug(f"{guid}: text_a: {text_a}")
    #          logger.debug(
    #              f"text_a[entity['start_pos']:entity['end_pos']]: {text_a[entity['start_pos']:entity['end_pos']]}"
    #          )
    #          logger.debug(
    #              f"mention {entity['mention']} in {text_a.find(entity['mention'])}"
    #          )
    #          logger.debug(f"entity: {entity}")
    #          assert text_a[entity['start_pos']:entity['end_pos']] == entity[
    #              'mention']

    return aug_tokens


#  def data_seg_generator(lines, ner_labels, seg_len=0, seg_backoff=0):
#      all_text_entities = []
#      labels_map = {}
#
#      for i, s in enumerate(tqdm(lines)):
#          guid = str(i)
#          text = s['originalText'].strip()
#          entities = s['entities']
#
#          new_entities = []
#          used_span = []
#          entities = sorted(entities, key=lambda e: e['start_pos'])
#          for entity in entities:
#              if entity['label_type'] not in ner_labels:
#                  continue
#              entity['mention'] = text[entity['start_pos']:entity['end_pos']]
#              s = entity['start_pos']
#              e = entity['end_pos']
#
#              overlap = False
#              for us in used_span:
#                  if s >= us[0] and s < us[1]:
#                      overlap = True
#                      break
#                  if e > us[0] and e <= us[1]:
#                      overlap = True
#                      break
#              if overlap:
#                  logger.warning(
#                      f"Overlap! {i} mention: {entity['mention']}, used_span: {used_span}"
#                  )
#                  continue
#              used_span.append((s, e))
#
#              new_entities.append(entity)
#          entities = new_entities
#
#          guid = str(i)
#
#          seg_offset = 0
#          if seg_len <= 0:
#              seg_len = max_seq_length
#
#          for (seg_text, ) in seg_generator((text, ), seg_len, seg_backoff):
#              text_a = seg_text
#
#              seg_start = seg_offset
#              seg_end = seg_offset + min(seg_len, len(seg_text))
#              labels = [
#                  (x['label_type'], x['start_pos'] - seg_offset,
#                   x['end_pos'] - 1 - seg_offset) for x in entities
#                  if x['start_pos'] >= seg_offset and x['end_pos'] <= seg_end
#              ]
#
#              # 没有标注存在的文本片断不用于训练
#              if labels:
#                  yield guid, text_a, None, labels
#
#                  if num_augements > 0:
#                      seg_entities = [{
#                          'start_pos': x['start_pos'] - seg_offset,
#                          'end_pos': x['end_pos'] - seg_offset,
#                          'label_type': x['label_type'],
#                          'mention': x['mention']
#                      } for x in entities if x['start_pos'] >= seg_offset
#                                      and x['end_pos'] <= seg_end]
#                      all_text_entities.append((guid, text_a, seg_entities))
#
#                      for entity in seg_entities:
#                          label_type = entity['label_type']
#                          s = entity['start_pos']  # - seg_offset
#                          e = entity['end_pos']  #- seg_offset
#                          #  print(s, e)
#                          assert e >= s
#                          #  logger.debug(
#                          #      f"seg_start: {seg_start}, seg_end: {seg_end}, seg_offset: {seg_offset}"
#                          #  )
#                          #  logger.debug(f"s: {s}, e: {e}")
#                          assert s >= 0 and e <= len(seg_text)
#                          #  if s >= len(seg_text) or e >= len(seg_text):
#                          #      continue
#
#                          entity_text = seg_text[s:e]
#                          #  print(label_type, entity_text)
#
#                          assert len(entity_text) > 0
#                          if label_type not in labels_map:
#                              labels_map[label_type] = []
#                          labels_map[label_type].append(entity_text)
#
#              seg_offset += seg_len - seg_backoff
#
#      if num_augements > 0:
#          aug_tokens = augement_entities(all_text_entities, labels_map)
#          for guid, text, entities in aug_tokens:
#              text_a = text
#              for entity in entities:
#                  #  logger.debug(f"text_a: {text_a}")
#                  #  logger.debug(
#                  #      f"text_a[entity['start_pos']:entity['end_pos']]: {text_a[entity['start_pos']:entity['end_pos']]}"
#                  #  )
#                  #  logger.debug(
#                  #      f"mention {entity['mention']} in {text_a.find(entity['mention'])}"
#                  #  )
#                  #  logger.debug(f"entity: {entity}")
#                  assert text_a[entity['start_pos']:entity['end_pos']] == entity[
#                      'mention']
#              labels = [
#                  (entity['label_type'], entity['start_pos'],
#                   entity['end_pos'] - 1) for entity in entities
#                  if entity['end_pos'] <= (
#                      min(len(text_a), seg_len) if seg_len > 0 else len(text_a))
#              ]
#              yield guid, text_a, None, labels
