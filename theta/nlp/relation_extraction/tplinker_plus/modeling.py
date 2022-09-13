#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json
import numpy as np
from tqdm import tqdm
import torch
#  from sched import scheduler
#  from tensorboardX import SummaryWriter
#  writer = SummaryWriter(log_dir='./tensorboard_log')  # prepare summary writer

try:
    import rich

    def print(*arg, **kwargs):
        rich.print(*arg, **kwargs)
except:
    pass

script_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

from ..bert4torch.models import build_transformer_model, BaseModel
from ..bert4torch.losses import MultilabelCategoricalCrossentropy
from ..bert4torch.optimizers import get_linear_schedule_with_warmup
from ..bert4torch.utils import sequence_padding, Callback
# from ..bert4torch.layers import GlobalPointer
from ..bert4torch.layers import EfficientGlobalPointer as GlobalPointer

from .dataset import encode_text, encode_sentences

#  from dataset_a_b_x_y import masks_a, masks_b, masks_x, masks_y
#  from .dataset import max_length, categories_label2id, categories_id2label

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# BERT base
#  bert_model_path = f"{script_path}/../pretrained/bert-base-chinese"


# 定义bert上的模型结构
class Model(BaseModel):

    def __init__(self, bert_model_path, ner_vocab_size, ner_head_size):
        super().__init__()

        config_path = f"{bert_model_path}/bert_config.json"
        checkpoint_path = f"{bert_model_path}/pytorch_model.bin"
        dict_path = f"{bert_model_path}/vocab.txt"
        self.bert = build_transformer_model(
            config_path=config_path, checkpoint_path=checkpoint_path, segment_vocab_size=0
        )
        self.global_pointer = GlobalPointer(hidden_size=768, heads=ner_vocab_size, head_size=ner_head_size)

    def forward(self, token_ids):
        sequence_output = self.bert([token_ids])  # [btz, seq_len, hdsz]
        logit = self.global_pointer(sequence_output, token_ids.gt(0).long())

        return logit

    #  def forward(self, token_ids_list):
    #      logit_list = []
    #
    #      for token_ids in token_ids_list:
    #          sequence_output = self.bert([token_ids])  # [btz, seq_len, hdsz]
    #          logit = self.global_pointer(sequence_output, token_ids.gt(0).long())
    #          logit_list.append(logit.unsqueze(0))
    #
    #      logit = torch.cat(logit_list)
    #
    #      return logit


def collate_fn(batch):
    batch_token_ids, batch_labels = [], []
    for token_ids, labels in batch:
        batch_token_ids.append(token_ids)
        batch_labels.append(labels)

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_labels = torch.tensor(sequence_padding(batch_labels, seq_dims=4), dtype=torch.long, device=device)
    return batch_token_ids, batch_labels


class MyLoss(MultilabelCategoricalCrossentropy):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_pred, y_true):
        #  print("y_pred.shape:", y_pred.shape)
        #  print("y_true.shape:", y_true.shape)

        y_pred = y_pred.view(y_pred.shape[0] * y_pred.shape[1], -1)  # [btz*ner_vocab_size, seq_len*seq_len]
        y_true = y_true.view(
            y_true.shape[0] * y_true.shape[1] * y_true.shape[2], -1
        )  # [btz*ner_vocab_size, seq_len*seq_len]

        return super().forward(y_pred, y_true)


def evaluate(model, dataloader, entities_id2label, relations_id2label, threshold=0):
    X, Y, Z = 0, 1e-10, 1e-10
    for x_true, label in dataloader:
        scores = model.predict(x_true)

        for i, score in enumerate(scores):
            R = set()
            for r_l, start, end in zip(*np.where(score.cpu() > threshold)):
                num_entities = len(entities_id2label)
                r = r_l // num_entities
                l = r_l % num_entities
                R.add((start, end, entities_id2label[l], relations_id2label[r]))

            T = set()
            for r, l, start, end in zip(*np.where(label[i].cpu() > threshold)):
                T.add((start, end, entities_id2label[l], relations_id2label[r]))
            X += len(R & T)
            Y += len(R)
            Z += len(T)

    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z

    eval_result = {
        'all': (f1, precision, recall)
    }
    return eval_result


#  def evaluate(model, dataloader, categories_id2label, threshold=0):
#      X, Y, Z = 0, 1e-10, 1e-10
#      for x_true, label_list in tqdm(dataloader, desc="eval", ncols=160):
#          scores_list = model.predict(x_true)
#          scores = scores_list[0]
#          label = label_list[0]
#
#          for i, score in enumerate(scores):
#              R = set()
#              for l, start, end in zip(*np.where(score.cpu() > threshold)):
#                  R.add((start, end, categories_id2label[l]))
#
#              T = set()
#              for l, start, end in zip(*np.where(label[i].cpu() > threshold)):
#                  T.add((start, end, categories_id2label[l]))
#              X += len(R & T)
#              Y += len(R)
#              Z += len(T)
#
#      f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
#
#      eval_result = {
#          'all': (f1, precision, recall)
#      }
#      return eval_result


class Evaluator(Callback):
    """评估与保存
    """

    def __init__(
        self, model, val_dataloader, entities_id2label, relations_id2label, best_f1=0., min_best=0.9, threshold=0
    ):
        self.model = model
        self.val_dataloader = val_dataloader
        self.entities_id2label = entities_id2label
        self.relations_id2label = relations_id2label
        self.best_f1 = best_f1
        self.min_best = 0.9

    def on_epoch_end(self, steps, epoch, logs=None):
        eval_result = evaluate(
            self.model, self.val_dataloader, self.entities_id2label, self.relations_id2label, threshold=0
        )
        f1, precision, recall = eval_result['all']
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.model.save_weights('best_model.pt')
            if f1 > self.min_best:
                self.model.save_weights(f'best_model_{self.best_f1:.5f}.pt')
        print(f'[val] f1: {f1:.5f}, p: {precision:.5f} r: {recall:.5f} best_f1: {self.best_f1:.5f}')
        # print(f"[val] {json.dumps(eval_result, ensure_ascii=False)}")
        for k, v in eval_result.items():
            if k == "total":
                continue
            v_list = [f"{x:.5f}" for x in v]
            print(f"\"{k}\": {v_list} ")

        logs.update({"f1": f1})
        logs.update({"best_f1": self.best_f1})


def build_model(args, entity_labels, relation_labels, bert_model_path, learning_rate=0, num_training_steps=0):
    ner_vocab_size = len(entity_labels) * len(relation_labels)
    ner_head_size = 64
    model = Model(bert_model_path, ner_vocab_size, ner_head_size).to(device)

    if learning_rate > 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = None

    if num_training_steps > 0:
        num_warmup_steps = 0  # int(num_training_steps * 0.05)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )
    else:
        scheduler = None

    model.compile(loss=MyLoss(), optimizer=optimizer, scheduler=scheduler)

    return model


def predict_text(args, model, text, tokenizer, entity_labels, relation_labels, repeat=1, threshold=0):

    entities_label2id = {label: i
                         for i, label in enumerate(entity_labels)}
    entities_id2label = {i: label
                         for i, label in enumerate(entity_labels)}
    relations_label2id = {label: i
                          for i, label in enumerate(relation_labels)}
    relations_id2label = {i: label
                          for i, label in enumerate(relation_labels)}

    true_tags = []
    token_ids, _ = encode_text(text, true_tags, entities_label2id, relations_label2id, args.max_length, tokenizer)
    batch_token_ids = [token_ids]
    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)

    scores_list = []
    for _ in range(repeat):
        scores = model.predict(batch_token_ids)
        scores_list.append(scores)

    def average_scores(scores_list):
        scores_list = [s.unsqueeze(0) for s in scores_list]
        scores = torch.cat(scores_list).mean(dim=0)
        return scores

    scores = average_scores(scores_list)
    sentences = [text]

    def scores_to_mentions(scores):
        tags_list = []
        for i, (score, sent_text) in enumerate(zip(scores, sentences)):
            if len(sent_text) == 0:
                continue
            tokens = tokenizer.tokenize(sent_text, maxlen=args.max_length)
            mapping = tokenizer.rematch(sent_text, tokens)
            # print(f"sent_text: len: {len(sent_text)}")
            # print(f"tokens: {len(tokens)}, {tokens}")
            # print(f"mapping: {len(mapping)}, {mapping}")

            # 用集合自动消除完全相同的实体标注
            R = {}
            for r, l, start, end in zip(*np.where(score.cpu() > threshold)):

                if r in relations_id2label and l in entities_id2label and start >= 1 and start < len(
                    mapping
                ) - 1 and end >= 1 and end < len(mapping) - 1 and start < end:
                    # print(f"l: {l}, start: {start}, end: {end}")
                    # print(f"mapping[start][0]: {mapping[start][0]}, mapping[end][-1] + 1: {mapping[end][-1] + 1}")
                    span_s = mapping[start][0]
                    span_e = mapping[end][-1]
                    k2 = sent_text[span_s:span_e + 1]
                    #  k2 = sent_text[mapping[start][0]:mapping[end][-1] + 1]
                    # print(start, end, entities_id2label[l], relations_id2label[r], k2)

                    cat_label = entities_id2label[l]
                    relation_label = relations_id2label[r]

                    if relation_label not in R:
                        R[relation_label] = set()

                    R[relation_label].add((span_s, span_e, cat_label, k2, sent_text))

            text_tags = []
            for relation_label, e_list in R.items():
                entity_tags = [{
                    'category': cat_label,
                    'start': start,
                    'mention': k2
                } for start, end, cat_label, k2, sent_text in e_list]
                entity_tags = sorted(entity_tags, key=lambda x: x['start'])
                text_tags.append({
                    'relation': relation_label,
                    'entities': entity_tags
                })

            tags_list.append({
                'text': sent_text,
                'tags': text_tags
            })

        return tags_list

    tags_list = scores_to_mentions(scores)

    return tags_list


#  def predict_sentences(args, model, sentences, tokenizer, categories_id2label, repeat=1, threshold=0):
#      # print(f"sentences: {sentences}")
#      batch_token_ids = []
#      for sent in sentences:
#          tokens = tokenizer.tokenize(sent, maxlen=max_length)
#          token_ids = tokenizer.tokens_to_ids(tokens)
#
#          batch_token_ids.append(token_ids)  # 前面已经限制了长度
#
#      batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
#
#      return predict_batch_tokens(args, model, batch_model_ids, categories_id2label, repeat=repeat, threshold=threshold)

#  def predict_batch_tokens(args, model, batch_token_ids, categories_id2label, repeat=1, threshold=0):
#      #  input = batch_token_ids.cpu().numpy()
#      # print(f"input.shape: {input.shape}")
#
#      scores_list = []
#      for _ in range(repeat):
#          scores = model.predict(batch_token_ids)
#          scores_list.append(scores)
#
#      def average_scores(scores_list):
#          scores_list = [s.unsqueeze(0) for s in scores_list]
#          scores = torch.cat(scores_list).mean(dim=0)
#          return scores
#
#      scores = average_scores(scores_list)
#
#      def scores_to_mentions(scores):
#          tags_list = []
#          for i, (score, sent_text) in enumerate(zip(scores, sentences)):
#              if len(sent_text) == 0:
#                  continue
#              tokens = tokenizer.tokenize(sent_text, maxlen=max_length)
#              mapping = tokenizer.rematch(sent_text, tokens)
#              # print(f"sent_text: len: {len(sent_text)}")
#              # print(f"tokens: {len(tokens)}, {tokens}")
#              # print(f"mapping: {len(mapping)}, {mapping}")
#
#              R = set()
#              for l, start, end in zip(*np.where(score.cpu() > threshold)):
#
#                  if l in categories_id2label and start >= 1 and start < len(mapping) - 1 and end >= 1 and end < len(
#                      mapping
#                  ) - 1 and start < end:
#                      # print(f"l: {l}, start: {start}, end: {end}")
#                      # print(f"mapping[start][0]: {mapping[start][0]}, mapping[end][-1] + 1: {mapping[end][-1] + 1}")
#                      span_s = mapping[start][0]
#                      span_e = mapping[end][-1]
#                      k2 = sent_text[span_s:span_e + 1]
#                      #  k2 = sent_text[mapping[start][0]:mapping[end][-1] + 1]
#                      # print(start, end, categories_id2label[l], k2)
#
#                      cat_label = categories_id2label[l]
#
#                      R.add((span_s, span_e, cat_label, k2, sent_text))
#
#              sent_tags = [{
#                  'category': cat_label,
#                  'start': start,
#                  'mention': k2
#              } for start, end, cat_label, k2, sent_text in R]
#              sent_tags = sorted(sent_tags, key=lambda x: x['start'])
#
#              #  mentions_list.append((sent_text, R))
#              tags_list.append({
#                  'text': sent_text,
#                  'tags': sent_tags
#              })
#
#          return tags_list
#
#      tags_list = scores_to_mentions(scores)
#
#      from .utils import merge_sent_tags_list
#      full_tags = merge_sent_tags_list(tags_list)
#
#      return full_tags, tags_list
