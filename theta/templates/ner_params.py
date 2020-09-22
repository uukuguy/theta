#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------- Params --------------------
from theta.modeling import (CommonParams, NerAppParams, NerParams, Params,
                            log_global_params)
#  from ner_task import ner_labels

experiment_params = NerAppParams(
    CommonParams(
        dataset_name="ner_task",
        experiment_name="Theta",
        train_file=None,
        eval_file=None,
        test_file=None,
        #  learning_rate=2e-5,
        learning_rate=1e-4,
        adam_epsilon=1e-6,
        weight_decay=0.0,
        train_max_seq_length=128,
        eval_max_seq_length=128,
        per_gpu_train_batch_size=16,
        per_gpu_eval_batch_size=16,
        per_gpu_predict_batch_size=16,
        seg_len=128,
        seg_backoff=64,
        num_train_epochs=5,
        fold=0,
        num_augements=0,
        enable_kd=False,
        #  enable_sda=True,
        #  sda_teachers=3,
        #  sda_stategy='recent_models',
        #  sda_empty_first=False,
        loss_type="CrossEntropyLoss",
        #  loss_type="FocalLoss",
        focalloss_gamma=2.0,
        model_type="bert",
        #  model_path="/opt/share/pretrained/pytorch/roberta-large-chinese",
        model_path="/opt/share/pretrained/pytorch/bert-base-chinese",
        fp16=True,
        best_index="f1",
        random_type=None,
        allow_overlap=False,
        #  max_train_examples=0,
        confidence=0.5,
        enable_nested_entities=False,
        seed=8864),
    NerParams(
        #  ner_labels=ner_labels,
        ner_type='pn', ),
)

experiment_params.debug()

if __name__ == '__main__':
    pass
