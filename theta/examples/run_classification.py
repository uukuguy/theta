#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json, random
from tqdm import tqdm
from loguru import logger
from pathlib import Path
import pandas as pd
import numpy as np

from theta.utils import init_theta, split_train_eval_examples
from theta.modeling.glue import GlueTrainer, load_model, InputExample, load_examples

fold = 0
seg_len = 0
seg_backoff = 0
train_sample_rate = 1.0
glue_labels = ['0', '1']

train_base_file = "./data/classifier_public_sentiment_train.tsv"
test_base_file = "./data/classifier_public_sentiment_test.tsv"


def get_result_file(args):
    return f"{args.output_dir}/{args.dataset_name}_result.tsv"


def get_submission_file(args):
    return f"{args.output_dir}/{args.dataset_name}_submission.tsv"


# -------------------- Data --------------------


def clean_text(text):
    text = text.strip().replace('\n', '')
    text = text.replace('\t', ' ')
    return text


# title, content各自截断长度
def get_title_content_seq_length(max_seq_length):
    title_seq_length = int(max_seq_length / 2)
    content_seq_length = max_seq_length - title_seq_length
    return title_seq_length, content_seq_length


def train_data_generator(args, train_file):
    title_seq_length, content_seq_length = get_title_content_seq_length(
        args.train_max_seq_length)

    df_train = pd.read_csv(train_file, sep='\t')

    total_examples = df_train.shape[0]
    num_sample_examples = int(total_examples * train_sample_rate)
    logger.warning(
        f"Sample {num_sample_examples}/{total_examples} ({train_sample_rate*100:.1f}%) train examples."
    )

    for i, row in tqdm(df_train.iterrows(), desc="train"):
        if i >= num_sample_examples:
            break

        guid = str(row.id)
        title = row.title
        title = clean_text(title)
        content = str(row.content)
        content = content if content != "nan" else ""
        content = clean_text(content)
        text = title[:title_seq_length] + content[-content_seq_length:]
        label = str(row.label)

        yield guid, text, None, label


def test_data_generator(args, test_file):
    title_seq_length, content_seq_length = get_title_content_seq_length(
        args.eval_max_seq_length)

    df_test = pd.read_csv(test_file, sep='\t')
    for i, row in tqdm(df_test.iterrows(), desc="test"):
        guid = str(row.id)
        title = row.title
        title = clean_text(title)
        content = str(row.content)
        content = content if content != "nan" else ""
        content = clean_text(content)
        text = title[:title_seq_length] + content[-content_seq_length:]

        yield guid, text, None, None


def load_train_eval_examples(args, train_base_file, seg_len=0, seg_backoff=0):
    train_base_examples = load_examples(args,
                                        train_data_generator,
                                        train_base_file,
                                        seg_len=seg_len,
                                        seg_backoff=seg_backoff)

    train_examples, eval_examples = split_train_eval_examples(
        train_base_examples,
        train_rate=args.train_rate,
        fold=fold,
        shuffle=True,
        random_state=args.seed)

    logger.info(
        f"Loaded {len(train_examples)} train examples, {len(eval_examples)} eval examples."
    )
    return train_examples, eval_examples


def load_test_examples(args, test_base_file, seg_len=0, seg_backoff=0):
    test_examples = load_examples(args,
                                  test_data_generator,
                                  test_base_file,
                                  seg_len=seg_len,
                                  seg_backoff=seg_backoff)

    return test_examples


# -------------------- Model --------------------

#  def build_model(args):
#      # -------- model --------
#      from theta.modeling.glue import load_pretrained_model
#      model = load_pretrained_model(args)
#      model.to(args.device)
#
#      # -------- optimizer --------
#      from transformers.optimization import AdamW
#      from theta.modeling.trainer import get_default_optimizer_parameters
#      optimizer_parameters = get_default_optimizer_parameters(
#          model, args.weight_decay)
#      optimizer = AdamW(optimizer_parameters,
#                        lr=args.learning_rate,
#                        correct_bias=False)
#
#      # -------- scheduler --------
#      from transformers.optimization import get_linear_schedule_with_warmup
#
#      scheduler = get_linear_schedule_with_warmup(
#          optimizer,
#          num_warmup_steps=args.total_steps * args.warmup_rate,
#          num_training_steps=args.total_steps)
#
#      return model, optimizer, scheduler

# -------------------- Trainer --------------------


class Trainer(GlueTrainer):
    def __init__(self, args, glue_labels):
        super(Trainer, self).__init__(args, glue_labels, build_model=None)

    #  def on_predict_end(self, args, test_dataloader):
    #      super(Trainer, self).on_predict_end(args, test_dataloader)
    #      logger.info(f"self.pred_results: {self.pred_results}")


# -------------------- Outputs --------------------


def save_predict_results(args, pred_results, pred_results_file, test_examples):
    with open(pred_results_file, 'w') as wr:
        wr.write(f"id\tlabel\n")
        for label, example in zip(pred_results, test_examples):
            ID = example.guid
            wr.write(f"{ID}\t{label}\n")
    logger.info(f"Predict results file saved to {pred_results_file}")


# -------------------- Main --------------------


def main(args):
    init_theta(args)

    if args.loss_type == 'FocalLoss':
        args.focalloss_gamma = 1.5
        args.focalloss_alpha = None

    trainer = Trainer(args, glue_labels)

    # --------------- train phase ---------------
    if args.do_train:
        train_examples, eval_examples = load_train_eval_examples(
            args, train_base_file)
        trainer.train(args, train_examples, eval_examples)

    # --------------- predict phase ---------------
    if args.do_eval:
        train_examples, eval_examples = load_train_eval_examples(
            args, train_base_file)

        model = load_model(args)
        trainer.evaluate(args, model, eval_examples)

    # --------------- predict phase ---------------
    if args.do_predict:
        test_examples = load_test_examples(args, test_base_file)

        model = load_model(args)
        trainer.predict(args, model, test_examples)
        save_predict_results(args, trainer.pred_results, get_result_file(args),
                             test_examples)


if __name__ == '__main__':
    from theta.modeling.glue.args import get_args
    args = get_args()
    main(args)
