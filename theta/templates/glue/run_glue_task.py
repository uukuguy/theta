#!/usr/bin/env python
# -*- coding: utf-8 -*-

from loguru import logger
from tqdm import tqdm

# -------------------- Parameters and data --------------------
from glue_task import (generate_submission, glue_labels, train_data_generator,
                       test_data_generator)
from glue_params import experiment_params
experiment_params.glue_params.glue_labels = glue_labels

# -------------------- MyApp --------------------
from theta.modeling.app import GlueApp


class MyApp(GlueApp):
    def __init__(self, experiment_params, glue_labels, add_special_args=None):

        super(MyApp, self).__init__(experiment_params, glue_labels,
                                    add_special_args)

    def run(
        self,
        train_data_generator,
        test_data_generator,
        generate_submission=None,
        eval_data_generator=None,
    ):

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
                glue_labels=glue_labels,
                add_special_args=add_special_args)

    app.run(train_data_generator,
            test_data_generator,
            generate_submission=generate_submission,
            eval_data_generator=train_data_generator)
