#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from collections import OrderedDict
from typing import Type

import numpy as np
import torch
import torch.functional as F
import torch.nn as nn
from loguru import logger
from torch.nn import CrossEntropyLoss, MSELoss
from tqdm import tqdm

from theta.nlp.arguments import (DataArguments, ModelArguments, TaskArguments,
                                 TrainingArguments)
from theta.nlp.data.samples import GlueSamples

from ..tasks import BaseDataset, BaseTask, TaskData, TaskModel


# ------------------------------ Dataset ------------------------------
class GlueDataset(BaseDataset):
    """
    完成模型输入数据的编码工作
    """
    def __init__(self, *args, **kwargs):
        super(GlueDataset, self).__init__(*args, **kwargs)

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
            return_offsets_mapping=True)
        input_ids = torch.from_numpy(
            np.array(encodings.input_ids, dtype=np.int64))[0]
        attention_mask = torch.from_numpy(
            np.array(encodings.attention_mask, dtype=np.int64))[0]
        token_type_ids = torch.from_numpy(
            np.array(encodings.token_type_ids, dtype=np.int64))[0]

        # -------- labels --------
        if labels is not None:
            if isinstance(labels, list):
                encoded_labels = [0] * len(self.label2id)
                for x in labels:
                    encoded_labels[self.label2id[x]] = 1
            else:
                if labels:
                    encoded_labels = self.label2id[labels]
                else:
                    encoded_labels = 0
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
        for key in ['input_ids', 'attention_mask', 'token_type_ids', 'labels']:
            key_batch = [e[key] for e in batch if e[key] is not None]
            if key_batch:
                batch_values = torch.stack(key_batch)
                stacked_batch[key] = batch_values
            else:
                stacked_batch[key] = None
        return stacked_batch


# ------------------------------ TaskData ------------------------------
class GlueData(TaskData):
    """
    指定任务专属的Dataset类型，并提供训练、验证、测试集实例。
    """
    def __init__(self, *args, **kwargs):
        super(GlueData, self).__init__(*args, **kwargs)

    @property
    def train_dataset(self):
        if self._train_dataset is None:
            self._train_dataset = GlueDataset(
                self.data_args, self._splitted_train_samples.rows,
                self.label2id, self.tokenizer)
        return self._train_dataset

    @property
    def val_dataset(self):
        if self._val_dataset is None:
            self._val_dataset = GlueDataset(self.data_args,
                                            self._splitted_val_samples.rows,
                                            self.label2id, self.tokenizer)
        return self._val_dataset

    @property
    def test_dataset(self):
        if self._test_dataset is None:
            self._test_dataset = GlueDataset(self.data_args,
                                             self.test_samples.rows,
                                             self.label2id, self.tokenizer)
        return self._test_dataset


# ------------------------------ TaskModel ------------------------------
class GlueModel(TaskModel):
    """
    任务专属模型定义
    """
    def __init__(self, task_args, num_labels):
        super(GlueModel, self).__init__(**task_args.to_dict())
        self.num_labels = num_labels

        config = self.transformer_model.config
        if self._is_xlnet():
            from transformers.modeling_utils import SequenceSummary
            self.sequence_summary = SequenceSummary(config)
            self.logits_proj = nn.Linear(config.d_model, num_labels)
        else:
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.classifier = nn.Linear(config.hidden_size, num_labels)

    #  def forward(self, batch, batch_idx):
    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                labels=None):

        outputs = self.transformer_model(input_ids,
                                         attention_mask=attention_mask,
                                         token_type_ids=token_type_ids,
                                         return_dict=True)
        #  logger.warning(f"outputs[0].shape: {outputs[0].shape}")
        #  logger.warning(f"outputs: {outputs}")
        # logger.warning(f"outputs[1]: {outputs[1]}")

        if self._is_xlnet():
            output = outputs[0]
            output = self.sequence_summary(output)
            logits = self.logits_proj(output)
        elif self._is_electra():
            # ElectraConfig last_hidden_state
            pooled_output = outputs[0]

            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
        else:
            # BERT
            pooled_output = outputs.pooler_output

            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return (loss, logits)
        else:
            return logits

    def training_step(self, batch, batch_idx):
        outputs = self.forward(**batch)
        loss = outputs[0]

        self.log('train_loss', loss, on_step=True)
        #  self.log('lr', self.hparams.lr, on_step=True)

        return OrderedDict({
            "loss": loss,
        })

    def validation_step(self, batch, batch_idx):
        loss, logits = self.forward(**batch)

        labels = batch['labels']
        labels_hat = torch.argmax(logits, dim=1)
        correct_count = torch.sum(labels == labels_hat)
        if self.on_gpu:
            correct_cout = correct_count.cuda(loss.device.index)

        self.log('val_loss', loss, on_step=True)

        return OrderedDict({
            'val_loss': loss,
            'correct_count': correct_count,
            'batch_size': len(labels)
        })

    def validation_epoch_end(self, outputs):
        val_acc = sum([out["correct_count"]
                       for out in outputs]).float() / sum(out["batch_size"]
                                                          for out in outputs)
        val_loss = sum([out["val_loss"] for out in outputs]) / len(outputs)
        logger.info(f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

        self.log('val_loss', val_loss, on_epoch=True)
        self.log('val_acc', val_acc, on_epoch=True)

        eval_outputs = {'val_loss': val_loss, 'val_acc': val_acc}
        self.save_best_model(eval_outputs)

    def test_step(self, batch, batch_idx):
        logits = self.forward(**batch)

        preds = torch.argmax(logits, dim=1)

        return OrderedDict({'preds': preds, 'logits': logits})

    def test_epoch_end(self, outputs):

        final_preds = torch.cat([out['preds'] for out in outputs])
        logger.info(f"preds: {final_preds.shape}")
        final_preds = final_preds.detach().cpu().numpy().tolist()

        final_logits = torch.cat([out['logits'] for out in outputs])
        logger.info(f"logits: {final_logits.shape}")
        final_logits = final_logits.detach().cpu().numpy().tolist()

        self.test_results = {'preds': final_preds, 'logits': final_logits}


# ------------------------------ Task ------------------------------
class GlueTask(BaseTask):
    def __init__(self, *args, **kwargs):
        super(GlueTask, self).__init__(*args, **kwargs)

    def execute(self, *args, **kwargs):
        model_args = self.model_args
        data_args = self.data_args
        training_args = self.training_args
        remaining_args = self.remaining_args

        super(GlueTask, self).execute(*args, **kwargs)

    def generate_submission(self):
        logger.warning(f"GlueTask.generate_submission().")
