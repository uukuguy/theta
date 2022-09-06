#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json
from datetime import datetime
from tqdm import tqdm
from loguru import logger
try:
    import dill
except:
    import pickle as dill

import torch
from torch import nn

#  from transformers import (AutoConfig, AutoTokenizer)
#  from transformers import AutoModelForSequenceClassification
from transformers import (AutoConfig, AutoTokenizer)
from transformers import AutoModelForSequenceClassification
# from transformers import AutoModel

from .run_utils import BaseDataset, BaseTrainer, BaseRunner


def glue_label2id(glue_labels):
    label2id = {x: i
                for i, x in enumerate(glue_labels)}
    return label2id


def glue_id2label(glue_labels):
    id2label = {i: x
                for i, x in enumerate(glue_labels)}
    return id2label


class GlueDataset(BaseDataset):
    """
    """

    def __init__(self, data_args, data_generator, glue_labels, tokenizer):
        super(GlueDataset, self).__init__(
            data_args=data_args,
            data_generator=data_generator,
            label2id=glue_label2id(glue_labels),
            tokenizer=tokenizer,
        )

    def _encode_item(self, x):
        guid, text_a, text_b, labels = x

        # -------- input_ids, attention_mask, token_type_ids --------
        text_pair = [(text_a, text_b)] if text_b is not None else [text_a]
        encodings = self.tokenizer.batch_encode_plus(
            text_pair,
            padding=self.data_args.padding,
            max_length=self.data_args.max_length,
            add_special_tokens=True,
            truncation=True,
            return_offsets_mapping=True
        )
        input_ids = torch.from_numpy(np.array(encodings.input_ids, dtype=np.int64))[0]
        attention_mask = torch.from_numpy(np.array(encodings.attention_mask, dtype=np.int64))[0]
        token_type_ids = torch.from_numpy(np.array(encodings.token_type_ids, dtype=np.int64))[0]

        # -------- labels --------
        if labels is not None:
            if isinstance(labels, list):
                encoded_labels = [0] * len(self.label2id)
                for x in labels:
                    encoded_labels[self.label2id[x]] = 1
                labels = torch.from_numpy(np.array(encoded_labels, dtype=np.float32))
            else:
                encoded_labels = self.label2id[labels]
                labels = torch.from_numpy(np.array(encoded_labels, dtype=np.int64))

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': labels
        }

    @classmethod
    def collate_fn(cls, batch):
        stacked_batch = {}

        not_none_tensor_keys = ['input_ids', 'attention_mask', 'token_type_ids']
        maybe_none_tensor_keys = ['labels']
        not_tensor_keys = []

        # not None tensors
        for key in not_none_tensor_keys:
            key_batch = [e[key] for e in batch if e[key] is not None]
            #  logger.info(f"key: {key} key_batch: {key_batch}")
            batch_values = torch.stack(key_batch)
            stacked_batch[key] = batch_values
        # maybe None tensors
        for key in maybe_none_tensor_keys:
            key_batch = [e[key] for e in batch if e[key] is not None]
            if key_batch:
                batch_values = torch.stack(key_batch)
                stacked_batch[key] = batch_values
            else:
                stacked_batch[key] = None
        # not tensors
        for key in not_tensor_keys:
            key_batch = [e[key] for e in batch if e[key] is not None]
            stacked_batch[key] = key_batch

        return stacked_batch


#  def load_glue_dataset(data_args, data_generator, glue_labels, tokenizer, shuffle=False, split=False):
#      dataset = GlueDataset(data_args, data_generator, glue_labels, tokenizer)
#      dataset.load()
#      if shuffle:
#          dataset.shuffle()
#      if split:
#          splitted_datasets = dataset.split(data_args.split_ratios)
#          return splitted_datasets
#      else:
#          return dataset


# -------------------- Model --------------------
class SequenceClassificationModel(nn.Module):
    """

    """

    def __init__(self, model_name_or_path, num_labels, **kwargs):
        super(SequenceClassificationModel, self).__init__()
        self.num_labels = num_labels
        self.model_name_or_path = model_name_or_path

        #  self.init_weights()

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels, finetuning_task="glue")
        logger.info(f"{self.config}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, fast=True)

        #  self.load_from_config()
        self.load_from_pretrained(model_name_or_path)

    # def init_weights(self):
    #     self.classifier.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    #     self.classifier.bias.data.zero_()

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize the weights
        Derived from BertPreTrainedModel._init_weights() in modeling_bert.py of transformers.
        """
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _adjust_config(self):
        setattr(self.config, 'num_labels', self.num_labels)
        #  # default: 0.1
        #  setattr(self.config, 'attention_probs_dropout_prob', self.attention_probs_dropout_prob)
        #  # default: null
        #  setattr(self.config, 'classifier_dropout', self.dropout_prob)

    def load_from_config(self):
        self._adjust_config()
        self.bert = AutoModelForSequenceClassification.from_config(self.config)
        # self.bert = AutoModel.from_config(self.config)
        logger.warning(f"After load_from_config() : {self.config}")

    def load_from_pretrained(self, model_path=None):
        if model_path is None:
            model_path = self.model_name_or_path
        self._adjust_config()
        self.bert = AutoModelForSequenceClassification.from_pretrained(model_path, config=self.config)
        # self.bert = AutoModel.from_pretrained(model_path, config=self.config)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):

        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, return_dict=True
        )

        loss = outputs.loss
        logits = outputs.logits
        if labels is not None:
            return (loss, logits)
        else:
            return logits


# -------------------- Trainer --------------------
class GlueTrainer(BaseTrainer):
    """
    """


# -------------------- Runner --------------------
class GlueRunner(BaseRunner):

    def __init__(
        self,
        task_args,
        glue_labels,
        train_data_generator=None,
        test_data_generator=None,
        eval_data_generator=None,
    ):
        super(GlueRunner,
              self).__init__(task_args, glue_labels, train_data_generator, test_data_generator, eval_data_generator)

    @property
    def glue_labels(self):
        return self.labels_list

    @property
    def label2id(self):
        return glue_label2id(self.glue_labels)

    @property
    def id2label(self):
        return glue_id2label(self.glue_labels)

    def load_dataset(self, shuffle=False, split=False):
        dataset = GlueDataset(self.data_args, self.data_generator, self.glue_labels, self.model.tokenizer)
        return self._load_dataset(dataset, shuffle=shuffle, split=split)

    def configure_model(self):
        self.model = SequenceClassificationModel(
            model_name_or_path=self.model_args.model_name_or_path, num_labels=len(self.glue_labels)
        )
        return self.model

    def configure_optimizer(self):
        return super().configure_optimizer()

    def configure_scheduler_fn(self):
        return super().configure_scheduler_fn()

    def configure_loss_fn(self):
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn

    def do_submit(self):
        # -------------------- 载入模型推理结果 --------------------
        test_results = self.load_test_results()
        preds = test_results['preds']
        logits = test_results['logits']
        id2label = self.id2label

        # -------------------- 载入测试数据集 --------------------
        test_data = self.load_test_data()
        assert len(preds) == len(test_data), f"len(preds): {len(preds), len(test_data): {len(test_data)}}"

        # -------------------- 转换最终输出格式 --------------------
        # 转换最终输出格式
        final_results = []
        final_submissions = []
        for index, ((idx, text_a, text_b, _), pred) in enumerate(zip(test_data, preds)):
            label = id2label[pred]
            final_results.append({
                'idx': idx,
                'text_a': text_a,
                'text_b': text_b,
                'label': label
            })

            #  # TODO
            #  final_submissions.append({
            #      'idx': idx,
            #      'text_a': text_a,
            #      'text_b': text_b,
            #      'label': label
            #  })

        # -------------------- 保存最终结果 --------------------

        timestamp_filename = self.get_timestamp_filename()

        prediction_file = os.path.join(self.training_args.submissions_dir, f"prediction_{timestamp_filename}.json")
        json.dump(final_results, open(prediction_file, 'w'), ensure_ascii=False, indent=2)
        logger.warning(f"Saved {len(final_results)} lines in {prediction_file}")

        #  # TODO
        #  submission_file = os.path.join(self.training_args.submissions_dir, f"submission_{timestamp_filename}.json")
        #  json.dump(final_submissions, open(submission_file, 'w'), ensure_ascii=False, indent=2)
        #  logger.info(f"Saved {len(preds)} lines in {submission_file}")

        return {
            'prediction_file': prediction_file,
            'submission_file': submission_file
        }


def glue_tsv_data_generator(tsv_file):
    with open(jsonl_file) as fd:
        lines = fd.readlines()
        for i, line in enumerate(tqdm(lines, desc="{os.path.basename(tsv_file)}")):
            toks = line.strip().split('\t')
            if len(toks) != 4:
                logger.warning(f"tokens size must be 4: {line.strip()}")
                continue
            idx, text_a, text_b, label = toks
            yield idx, text_a, text_b, label


def glue_jsonl_data_generator(jsonl_file):
    with open(jsonl_file) as fd:
        lines = fd.readlines()
        for line in tqdm(lines, desc="{os.path.basename(jsonl_file)}"):
            d = json.loads(line.strip())
            idx = str(d['idx'])
            text_a = d['text_a']
            text_b = d['text_b']
            label = d['label']
            yield idx, text_a, text_b, label


def glue_json_data_generator(json_file):
    json_data = json.load(open(json_file))
    for d in tqdm(json_data, desc="{os.path.basename(json_file)}"):
        idx = str(d['idx'])
        text_a = d['text_a']
        text_b = d['text_b']
        label = d['label']
        yield idx, text_a, text_b, label
