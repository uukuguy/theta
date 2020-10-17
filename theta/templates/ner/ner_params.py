#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------- Params --------------------
from theta.modeling import (CommonParams, NerAppParams, NerParams, Params,
                            log_global_params)

experiment_params = NerAppParams(
    CommonParams(
        dataset_name="ner_task",
        experiment_name="Theta",
        train_file=None,
        eval_file=None,
        test_file=None,
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
        num_augments=0,
        enable_kd=False,
        enable_sda=False,
        sda_teachers=3,
        sda_stategy='recent_models',
        sda_empty_first=False,
        loss_type="CrossEntropyLoss",
        #  loss_type="FocalLoss",
        focalloss_gamma=2.0,
        model_type="bert",
        model_path="/opt/share/pretrained/pytorch/bert-base-chinese",
        fp16=True,
        best_index="f1",
        random_type=None,
        allow_overlap=False,
        max_train_examples=0,
        confidence=0.3,
        enable_nested_entities=False,
        seed=8864),
    NerParams(ner_type='pn', ),
)

# @gpu.huawei
# docker pull ner_task:
"""
python -m theta --pull_deepcode \
    --dataset_name ner_task \
    --local_id 

python -m theta --exec_deepcode \
    --dataset_name ner_task \
    --local_id 
"""

conf_common_params = {
    'dataset_name': "ner_task",
    'experiment_name': "Theta",
    'fold': 0,
    'num_augments': 2,
    'num_train_epochs': 10,
    'train_max_seq_length': 512,
    'eval_max_seq_length': 512,
    'per_gpu_train_batch_size': 4,
    'per_gpu_eval_batch_size': 4,
    'per_gpu_predict_batch_size': 4,
    'seg_len': 510,
    'seg_backoff': 128,
    'model_path': "/opt/share/pretrained/pytorch/bert-base-chinese",
    'confidence': 0.3,
    'seed': 8864
}

conf_ner_params = {
    'ner_type': 'pn',
}

for k, v in conf_common_params.items():
    setattr(experiment_params.common_params, k, v)
for k, v in conf_ner_params.items():
    setattr(experiment_params.ner_params, k, v)
experiment_params.debug()

if __name__ == '__main__':
    pass