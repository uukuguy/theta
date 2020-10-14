#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------- Params --------------------
from theta.modeling import Params, CommonParams, SpoParams, SpoAppParams, log_global_params

experiment_params = SpoAppParams(
    CommonParams(
        dataset_name="spo_task",
        experiment_name="Theta",
        train_file=None,
        eval_file=None,
        test_file=None,
        #  learning_rate=2e-5,
        learning_rate=1e-4,
        #  adam_epsilon=1e-6,
        #  weight_decay=0.0,
        train_max_seq_length=256,
        eval_max_seq_length=256,
        per_gpu_train_batch_size=16,
        per_gpu_eval_batch_size=16,
        per_gpu_predict_batch_size=16,
        seg_len=254,
        seg_backoff=127,
        num_train_epochs=20,
        fold=0,
        num_augments=0,
        enable_kd=False,
        enable_sda=False,
        sda_teachers=3,
        loss_type="CrossEntropyLoss",
        model_type="bert",
        model_path="/opt/share/pretrained/pytorch/bert-base-chinese",
        fp16=True,
        best_index="f1",
        random_type=None,
        confidence=0.5,
        seed=8864),
    SpoParams(),
)

# @gpu.huawei
# docker pull spo_task:
"""
python -m theta --pull_deepcode \
    --dataset_name spo_task \
    --local_id 

python -m theta --exec_deepcode \
    --dataset_name spo_task \
    --local_id 
"""

conf_common_params = {
    'dataset_name': "spo_task",
    'experiment_name': "Theta",
    'fold': 0,
    'num_augments': 0,
    'num_train_epochs': 5,
    'train_max_seq_length': 256,
    'eval_max_seq_length': 256,
    'per_gpu_train_batch_size': 4,
    'per_gpu_eval_batch_size': 4,
    'per_gpu_predict_batch_size': 4,
    'seg_len': 254,
    'seg_backoff': 127,
    'model_path': "/opt/share/pretrained/pytorch/bert-base-chinese",
    'confidence': 0.3,
    'seed': 8864
}

conf_spo_params = {}

for k, v in conf_common_params.items():
    setattr(experiment_params.common_params, k, v)
for k, v in conf_spo_params.items():
    setattr(experiment_params.spo_params, k, v)
experiment_params.debug()

if __name__ == '__main__':
    pass
