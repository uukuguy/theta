#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tqdm import tqdm
from loguru import logger

# -------------------- Data --------------------
from spo_task import predicate_labels, train_data_generator, test_data_generator, generate_submission
from spo_params import experiment_params
experiment_params.spo_params.predicate_labels = predicate_labels

# -------------------- SpoApp --------------------
from theta.modeling.app import SpoApp


class MyApp(SpoApp):
    def __init__(self,
                 experiment_params,
                 predicate_labels,
                 add_special_args=None):
        super(MyApp, self).__init__(experiment_params, predicate_labels,
                                    add_special_args)

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


# -------------------- Main --------------------
if __name__ == '__main__':

    # -------- Customized arguments --------
    def add_special_args(parser):
        parser.add_argument("--preapre_data",
                            action='store_true',
                            help="Preapre data.")
        return parser

    app = MyApp(experiment_params,
                predicate_labels,
                add_special_args=add_special_args)

    app.run(train_data_generator,
            test_data_generator,
            generate_submission=generate_submission,
            eval_data_generator=train_data_generator)
