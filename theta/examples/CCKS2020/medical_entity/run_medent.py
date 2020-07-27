#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, json, re
from tqdm import tqdm

from loguru import logger

# -------------------- Dataset --------------------
from medent import ner_labels, ner_connections, train_data_generator, test_data_generator, generate_submission

# -------------------- Params --------------------
from theta.modeling import Params, CommonParams, NerParams, NerAppParams, log_global_params

experiment_params = NerAppParams(
    CommonParams(
        dataset_name="medent",
        experiment_name="ccks2020_medical_entity",
        train_file='data/rawdata/ccks2020_2_task1_train/task1_train.txt',
        eval_file='data/rawdata/ccks2020_2_task1_train/task1_train.txt',
        test_file='data/rawdata/ccks2_task1_val/task1_no_val_utf8.txt',
        learning_rate=1e-4,
        adam_epsilon=1e-6,
        weight_decay=0.0,
        train_max_seq_length=512,
        eval_max_seq_length=512,
        per_gpu_train_batch_size=4,
        per_gpu_eval_batch_size=4,
        per_gpu_predict_batch_size=4,
        seg_len=510,
        seg_backoff=128,
        num_train_epochs=10,
        fold=0,
        num_augements=2,
        enable_kd=False,
        #  enable_sda=True,
        #  sda_teachers=3,
        #  sda_stategy='recent_models',
        #  sda_empty_first=False,
        loss_type="CrossEntropyLoss",
        model_type="bert",
        model_path=
        "/opt/share/pretrained/pytorch/roberta-wwm-large-ext-chinese",
        #  "/opt/share/pretrained/pytorch/bert-base-chinese",
        fp16=True,
        best_index="f1",
        random_type=None,
        allow_overlap=False,
        #  max_train_examples=0,
        confidence=0.5,
        enable_nested_entities=False,
        seed=6636),
    NerParams(
        ner_labels=ner_labels,
        ner_type='pn',
    ),
)

experiment_params.debug()

from theta.modeling.app import NerApp


class MyApp(NerApp):
    def __init__(self,
                 experiment_params: NerAppParams,
                 ner_labels,
                 ner_connections,
                 add_special_args=None):
        super(MyApp, self).__init__(experiment_params, ner_labels,
                                    ner_connections, add_special_args)

    def run(self,
            train_data_generator,
            test_data_generator,
            generate_submission=None,
            eval_data_generator=None):

        args = self.args

        if args.preapre_data:
            logger.info(f"Prepare data.")
        else:
            super(MyApp, self).run(train_data_generator, test_data_generator,
                                   generate_submission, eval_data_generator)


if __name__ == '__main__':

    # -------- Customized arguments --------
    def add_special_args(parser):
        parser.add_argument("--preapre_data",
                            action='store_true',
                            help="Preapre data.")
        return parser

    app = MyApp(experiment_params=experiment_params,
                ner_labels=ner_labels,
                ner_connections=ner_connections,
                add_special_args=add_special_args)

    app.run(train_data_generator=train_data_generator,
            test_data_generator=test_data_generator,
            generate_submission=generate_submission,
            eval_data_generator=train_data_generator)
