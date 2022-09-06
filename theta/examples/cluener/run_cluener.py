#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from loguru import logger
from tqdm import tqdm

# -------------------- Parameters and data --------------------
from cluener import (generate_submission, ner_connections, ner_labels,
                     test_data_generator, train_data_generator)
from cluener_params import experiment_params
# -------------------- NerApp --------------------
from theta.modeling.app import NerApp

experiment_params.ner_params.ner_labels = ner_labels


class MyApp(NerApp):
    def __init__(self,
                 experiment_params,
                 ner_labels: list,
                 ner_connections: list,
                 add_special_args=None):

        super(MyApp, self).__init__(experiment_params, ner_labels,
                                    ner_connections, add_special_args)

    def run(
        self,
        train_data_generator,
        test_data_generator,
        generate_submission=None,
        eval_data_generator=None,
    ):

        args = self.args

        from theta.utils import init_theta
        init_theta(args)

        if args.train:
            do_train(args)
        if args.predict:
            do_predict(args)
        if args.submit:
            do_submit(args, generate_submission)

        if args.preapre_data:
            logger.info(f"Prepare data.")
        else:
            super(MyApp, self).run(train_data_generator, test_data_generator,
                                   generate_submission, eval_data_generator)


def do_train(args):

    # ------------------------------ dataflow ------------------------------
    from theta.nlp.dataflow import EntityDataFlow

    from cluener import train_data_generator
    from cluener import ner_labels as entity_labels

    dataflow = EntityDataFlow(data_generator=train_data_generator)
    total_samples = len(dataflow)
    num_train_samples = int(total_samples * 0.9)
    num_eval_samples = total_samples - num_train_samples
    train_dataflow = EntityDataFlow(data_list=dataflow[:num_train_samples])
    eval_dataflow = EntityDataFlow(data_list=dataflow[-num_eval_samples:])
    train_dataflow.info()
    eval_dataflow.info()

    # ------------------------------ args ------------------------------
    #  def add_modeling_args(parser):
    #      #  parser.add_argument("--soft_label", action="store_true", help="Soft label for ner span")
    #      parser.add_argument("--test_new", action="store_true", help="Test new")
    #      return parser
    #
    #  from theta.modeling.common_args import get_main_args
    #  args = get_main_args(add_modeling_args,
    #                       experiment_params=experiment_params,
    #                       special_args=None)
    #

    # ------------------------------ model ------------------------------
    from theta.modeling.token_utils import HFTokenizer
    tokenizer = HFTokenizer(
        "/opt/share/pretrained/pytorch/bert-base-chinese/vocab.txt")

    from theta.nlp.modeling import EntityExtractionModel
    model = EntityExtractionModel(args,
                                  entity_labels,
                                  tokenizer,
                                  tagging_type="pointer_sequence")

    #  from theta.nlp.trainers import PointerSequenceTrainer
    #  trainer = PointerSequenceTrainer(args, entity_labels)
    #
    #  trainer.train(args, all_train_features, all_eval_features)
    #

    #  from theta.nlp.taggers import PointerSequenceTagger
    #  tagger = PointerSequenceTagger(tokenizer=tokenizer, label2id=label2id)
    #  all_train_features = tagger.encode(train_dataflow, max_seq_length=64)
    #  all_eval_features = tagger.encode(eval_dataflow, max_seq_length=64)

    model.train(train_dataflow, eval_dataflow)


def do_predict(args):
    # ------------------------------ dataflow ------------------------------
    from theta.nlp.dataflow import EntityDataFlow

    from cluener import test_data_generator
    from cluener import ner_labels as entity_labels

    test_dataflow = EntityDataFlow(data_generator=test_data_generator)
    test_dataflow.info()

    # ------------------------------ model ------------------------------
    from theta.modeling.token_utils import HFTokenizer
    tokenizer = HFTokenizer(
        "/opt/share/pretrained/pytorch/bert-base-chinese/vocab.txt")

    from theta.nlp.modeling import EntityExtractionModel
    model = EntityExtractionModel(args,
                                  entity_labels,
                                  tokenizer,
                                  tagging_type="pointer_sequence")

    model.predict(test_dataflow)


def do_submit(args, generate_submission):
    submission_file = generate_submission(args)

    def archive_local_model(args, submission_file=None):
        import os, shutil
        if os.path.exists(args.local_dir):
            shutil.rmtree(args.local_dir)
        shutil.copytree(args.latest_dir, args.local_dir)
        logger.info(
            f"Archive local model({args.local_id}) {args.latest_dir} to {args.local_dir}"
        )

    archive_local_model(args, submission_file)


# -------------------- Main --------------------
if __name__ == '__main__':
    #  test_new()

    # -------- Customized arguments --------
    def add_special_args(parser):
        parser.add_argument("--preapre_data",
                            action='store_true',
                            help="Preapre data.")
        parser.add_argument("--train", action='store_true', help="Do train.")
        parser.add_argument("--predict",
                            action='store_true',
                            help="Do predict.")
        parser.add_argument("--submit", action='store_true', help="Do submit.")
        return parser

    app = MyApp(experiment_params,
                ner_labels=ner_labels,
                ner_connections=ner_connections,
                add_special_args=add_special_args)

    app.run(train_data_generator,
            test_data_generator,
            generate_submission=generate_submission,
            eval_data_generator=None)
