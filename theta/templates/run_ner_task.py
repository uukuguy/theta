#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, json, re
from tqdm import tqdm
from loguru import logger

# -------------------- Dataset --------------------
from ner_task import ner_labels, ner_connections, train_data_generator, test_data_generator, generate_submission

# -------------------- Params --------------------
from theta.modeling import Params, CommonParams, NerParams, NerAppParams, log_global_params

experiment_params = NerAppParams(
    CommonParams(dataset_name="ner_task",
                 experiment_name="theta_ner_task",
                 train_file=None,
                 eval_file=None,
                 test_file=None,
                 learning_rate=2e-5,
                 train_max_seq_length=256,
                 eval_max_seq_length=256,
                 per_gpu_train_batch_size=8,
                 per_gpu_eval_batch_size=8,
                 per_gpu_predict_batch_size=8,
                 seg_len=254,
                 seg_backoff=127,
                 num_train_epochs=5,
                 fold=0,
                 num_augements=0,
                 enable_kd=False,
                 enable_sda=False,
                 sda_teachers=3,
                 loss_type="CrossEntropyLoss",
                 model_type="bert",
                 model_path="/opt/share/pretrained/pytorch/bert-base-chinese",
                 fp16=True,
                 best_index="f1",
                 random_type=None,
                 seed=8864),
    NerParams(
        ner_labels=ner_labels,
        ner_type='span',
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
