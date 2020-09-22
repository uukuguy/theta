#!/usr/bin/env python
# -*- coding: utf-8 -*-

from theta.modeling import (CommonParams, GlueAppParams, GlueParams, Params,
                            log_global_params)

experiment_params = GlueAppParams(
    CommonParams(
        dataset_name="glue_task",
        experiment_name="Theta",
        train_file=None,
        eval_file=None,
        test_file=None,
        learning_rate=2e-5,
        train_max_seq_length=200,
        eval_max_seq_length=200,
        predict_max_seq_length=200,
        per_gpu_train_batch_size=64,
        per_gpu_eval_batch_size=64,
        per_gpu_predict_batch_size=64,
        seg_len=0,
        seg_backoff=0,
        num_train_epochs=5,
        fold=0,
        num_augments=0,
        enable_kd=False,
        enable_sda=False,
        sda_teachers=3,
        sda_stategy="clone_models",
        sda_empty_first=True,
        #  loss_type="CircleLoss",
        #  loss_type="CrossEntropyLoss",
        #  loss_type = "BCEWithLogitsLoss",
        loss_type="FocalLoss",
        focalloss_gamma=2.0,
        model_type="bert",
        model_path=
        #  "/opt/share/pretrained/pytorch/roberta-wwm-large-ext-chinese",
        "/opt/share/pretrained/pytorch/bert-base-chinese",
        train_rate=0.9,
        fp16=False,
        best_index='f1',
    ),
    GlueParams())

# ----------------------------------------
# @gpu.huawei
# docker pull glue_task:
"""
python -m theta --pull_deepcode \
    --dataset_name glue_task \
    --local_id 

python -m theta --exec_deepcode \
    --dataset_name glue_task \
    --local_id 
"""

conf_common_params = {
    'dataset_name': "glue_task",
    'experiment_name': "Theta",
    'fold': 0,
    'num_train_epochs': 10,
    'train_max_seq_length': 128,
    'eval_max_seq_length': 128,
    'per_gpu_train_batch_size': 16,
    'per_gpu_eval_batch_size': 16,
    'per_gpu_predict_batch_size': 16,
    'model_path': "/opt/share/pretrained/pytorch/bert-base-chinese",
    'seed': 8864
}

conf_glue_params = {
}

for k, v in conf_common_params.items():
    setattr(experiment_params.common_params, k, v)
for k, v in conf_glue_params.items():
    setattr(experiment_params.glue_params, k, v)
experiment_params.debug()

if __name__ == '__main__':
    pass

