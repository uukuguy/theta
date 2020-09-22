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

experiment_params.debug()

if __name__ == '__main__':
    pass
