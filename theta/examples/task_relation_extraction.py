#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, json
from tqdm import tqdm
from loguru import logger
from functools import partial
from dataclasses import dataclass
from copy import deepcopy
import numpy as np

try:
    import rich

    def print(*arg, **kwargs):
        rich.print(*arg, **kwargs)


except:
    pass

TASK_NAME = "task_entity_extraction"

script_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
print(f"script_path: {script_path}")

theta_path = os.environ.get("THETA_HOME", os.path.realpath(f"{script_path}/.."))
print(f"theta_path: {theta_path}")

if theta_path and theta_path not in sys.path:
    sys.path.insert(0, theta_path)

from theta.nlp.relation_extraction.dataset import NerDataset, get_default_tokenizer
from theta.nlp.relation_extraction.runner import (
    run_training,
    run_evaluating,
    run_predicting,
)
from theta.nlp.bert4torch.tokenizers import Tokenizer

data_path = os.path.realpath(f"{script_path}/data")
print(f"data_path: {data_path}")

#  bert_model_path = os.path.realpath(f"{script_path}/../pretrained/bert-base-chinese")
bert_model_path = "/opt/local/pretrained/bert-base-chinese"
print(f"bert_model_path: {bert_model_path}")

dict_path = f"{bert_model_path}/vocab.txt"
print(f"dict_path: {dict_path}")

tokenizer = get_default_tokenizer(dict_path)


class DictObject(object):
    __setitem__ = object.__setattr__
    __getitem__ = object.__getattribute__

    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def update(self, *args, **kwargs):
        for k, v in dict(**kwargs).items():
            self.__setattr__(k, v)

    def __str__(self):
        return f"{str(self.__dict__)}"


args = DictObject(
    **dict(
        SEED=42,
        max_length=256,
        # 是否将文本划分为多个短句
        do_split=False,
        # Training
        batch_size=16,
        learning_rate=2e-5,
        num_training_epochs=500,
        max_training_episodes=1,
        min_test=0.9,
        # Early Stopping
        earlystopping_monitor="best_f1",
        earlystopping_patience=10,
        earlystopping_mode="max",
        # Predicting
        repeat=1,
        # Data files
        train_file=f"{data_path}/train_a.json",
        val_file=f"{data_path}/train_a.json",
        test_file=f"{data_path}/train_a.json",
    )
)

# FIXME
entity_labels = []
relation_labels = []


# FIXME
def tag_text(idx, line):
    """
    根据原始数据的json数据，构建模型标准数据格式 idx, text, tags, others
    {
        'text': "text",
        'tags': [
            {
                'relation': "relation",
                'entities':[
                    {'category': "category", 'start': 0, 'mention': "mention"}
                    ......
                ]
            }
            ......
        ]
    }
    """
    json_data = json.loads(line)

    idx = json_data["idx"]
    text = json_data["text"]
    tags = json_data["tags"]

    for tag in tags:
        relation = tag["relation"]
        entities = tag["entities"]
        for ent in entities:
            c, s, m = ent["category"], ent["start"], ent["mention"]

    print("idx:", idx, "text:", text, "tags:", tags)
    return idx, text, tags, None


# FIXME
def build_final_result(d, real_tags):
    """
    根据预测结果构造成上层应用需要的输出格式
    """
    idx, text, true_tags, _ = d

    result = real_tags

    return result


# FIXME
def decode_text_tags(full_tags, sent_tags_list):
    """
    根据实体抽取的结果，构造成业务应用需要的组合标注形式

    full_tags: 整句的标注
    {
        'text': "text",
        'tags': [
            {
                'relation': "relation",
                'entities':[
                    {'category': "category", 'start': 0, 'mention': "mention"}
                    ......
                ]
            }
            ......
        ]
    }

    """

    real_tags = []

    text, tags = full_tags["text"], full_tags["tags"]
    if args.do_split:
        merged_tags = merge_sent_tags_list(sent_tags_list)
        full_text = merged_tags["text"]
        full_tags = merged_tags["tags"]

        tags = full_tags

    tags = sorted(tags, key=lambda x: x["entities"][0]["start"])

    real_tags = tags

    return real_tags


def predict_test_file(test_file, best_model_file, results_file="results.json"):
    test_data = [x for x in test_data_generator(test_file)]

    # [(full_tags, sent_tags_list)]
    predictions = run_predicting(
        args,
        entity_labels,
        relation_labels,
        test_data,
        best_model_file,
        bert_model_path,
        tokenizer,
    )

    def decode_predictions(predictions):
        real_tags_list = []

        for full_tags, sent_tags_list in predictions:
            real_tags = decode_text_tags(full_tags, sent_tags_list)
            real_tags_list.append(real_tags)

        return real_tags_list

    real_tags_list = decode_predictions(predictions)

    assert len(test_data) == len(real_tags_list)

    final_results = []
    for d, real_tags in zip(test_data, real_tags_list):
        result = build_final_result(d, real_tags)
        final_results.append(result)

    with open(results_file, "w") as wt:
        for d in final_results:
            line = json.dumps(d, ensure_ascii=False)
            wt.write(f"{line}\n")
    print(f"Saved {len(final_results)} results in {results_file}")

    return final_results


def data_generator(data_file, do_split=False):
    for idx, line in enumerate(open(data_file).readlines()):
        line = line.strip()

        d = tag_text(idx, line)

        yield d


def train_data_generator(train_file):
    for d in data_generator(train_file):
        yield d


def val_data_generator(val_file):
    for d in data_generator(val_file):
        yield d


def test_data_generator(test_file):
    for d in data_generator(test_file):
        yield d


def main(args):
    if args.do_train:
        partial_train_data_generator = partial(
            train_data_generator, train_file=args.train_file
        )
        train_dataset = NerDataset(
            args,
            partial_train_data_generator,
            entity_labels,
            relation_labels,
            tokenizer,
            do_split=args.do_split,
        )

        partial_val_data_generator = partial(val_data_generator, val_file=args.val_file)
        val_dataset = NerDataset(
            args,
            partial_val_data_generator,
            entity_labels,
            relation_labels,
            tokenizer,
            do_split=args.do_split,
        )

        run_training(
            args,
            entity_labels,
            relation_labels,
            train_dataset,
            val_dataset,
            bert_model_path,
        )

    if args.do_eval:
        partial_val_data_generator = partial(val_data_generator, val_file=args.val_file)
        val_dataset = NerDataset(
            args,
            partial_val_data_generator,
            entity_labels,
            relation_labels,
            tokenizer,
            do_split=args.do_split,
        )

        run_evaluating(
            args, entity_labels, relation_labels, val_dataset, bert_model_path
        )

    if args.do_predict:
        best_model_file = "best_model.pt"
        predict_test_file(args.test_file, best_model_file)


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", default=TASK_NAME, help="The task name.")

    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run evaluating."
    )
    parser.add_argument(
        "--do_predict", action="store_true", help="Whether to run predicting."
    )

    parser.add_argument(
        "--train_file", type=str, default=args.train_file, help="Train file"
    )
    parser.add_argument("--val_file", type=str, default=args.val_file, help="Val file")
    parser.add_argument(
        "--test_file", type=str, default=args.test_file, help="Test file"
    )

    parser.add_argument("--seed", type=int, default=42, help="SEED")
    parser.add_argument(
        "--output_dir", type=str, default="./outputs", help="The output data dir."
    )
    parser.add_argument(
        "--log_dir", type=str, default="./logs", help="The log data dir."
    )
    parser.add_argument(
        "--saved_models_dir",
        type=str,
        default="./outputs/saved_models",
        help="The saved models dir.",
    )

    cmd_args, unknown_args = parser.parse_known_args()

    os.makedirs(cmd_args.output_dir, exist_ok=True)
    os.makedirs(cmd_args.saved_models_dir, exist_ok=True)

    return cmd_args, unknown_args


if __name__ == "__main__":
    cmd_args, _ = get_args()
    args.update(**cmd_args.__dict__)
    main(args)
