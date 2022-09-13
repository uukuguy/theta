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

TASK_NAME = "cluener"

script_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
print(f"script_path: {script_path}")

theta_path = os.environ.get("THETA_HOME", os.path.realpath(f"{script_path}/.."))
print(f"theta_path: {theta_path}")

if theta_path and theta_path not in sys.path:
    sys.path.insert(0, theta_path)

from theta.nlp.entity_extraction.tagging import (
    TaskLabels,
    TaggedData,
    TaskTag,
)
from theta.nlp.entity_extraction.global_pointer.dataset import (
    TaskDataset,
    get_default_tokenizer,
)
from theta.nlp.entity_extraction.global_pointer.runner import (
    run_training,
    run_evaluating,
    run_predicting,
)

data_path = os.path.realpath(f"{script_path}/data")
print(f"data_path: {data_path}")

#  bert_model_path = os.path.realpath(f"{script_path}/../pretrained/bert-base-chinese")
bert_model_path = os.path.realpath(f"/opt/local/pretrained/bert-base-chinese")
print(f"bert_model_path: {bert_model_path}")

dict_path = f"{bert_model_path}/vocab.txt"
print(f"dict_path: {dict_path}")

tokenizer = get_default_tokenizer(dict_path)

# FIXME
entity_labels = [
    "address",
    "book",
    "company",
    "game",
    "government",
    "movie",
    "name",
    "organization",
    "position",
    "scene",
]
task_labels = TaskLabels(entity_labels=entity_labels)


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
        task_labels=task_labels,
        bert_model_path=bert_model_path,
        extract_threshold=0,
        SEED=42,
        max_length=256,
        # 是否将文本划分为多个短句
        do_split=False,
        # Training
        batch_size=16,
        learning_rate=2e-5,
        num_training_epochs=12,
        max_training_episodes=1,
        min_best=0.9,
        # Early Stopping
        earlystopping_monitor="best_f1",
        earlystopping_patience=5,
        earlystopping_mode="max",
        # Predicting
        repeat=1,
        # Data files
        train_file=f"{data_path}/train.json",
        val_file=f"{data_path}/dev.json",
        test_file=f"{data_path}/test.json",
    )
)


# FIXME
def tag_text(idx, line):
    """
    根据原始数据的json数据，构建模型标准数据格式 idx, text, tags, others
    tags:
    [
        {'category': "category", 'start': 0, 'mention': "mention"}
        ......
    ]
    """
    json_data = json.loads(line)

    guid = f"train-{idx}"
    text = json_data["text"]

    # -------------------- 训练数据json格式 --------------------
    #  {
    #      "text": "万通地产设计总监刘克峰；",
    #      "label": {
    #          "name": {
    #              "刘克峰": [[8, 10]]
    #          },
    #          "company": {
    #              "万通地产": [[0, 3]]
    #          },
    #          "position": {
    #              "设计总监": [[4, 7]]
    #          }
    #      }
    #  }

    tags = []
    labels = json_data.get("label", [])
    if labels:
        classes = labels.keys()
        for c in classes:
            c_labels = labels[c]
            #  logger.debug(f"c_labels:{c_labels}")
            for label, span in c_labels.items():
                s, e = span[0]
                m = text[s : e + 1]
                #  tags.append({"category": c, "start": s, "mention": m})
                tag = TaskTag(c=c, s=s, m=m)
                tags.append(tag)

    #  print("idx:", idx, "text:", text, "tags:", tags)
    return TaggedData(idx, text, tags, None)


# FIXME
def build_final_result(d, real_tags):
    """
    根据预测结果构造成上层应用需要的输出格式
    """
    idx, text, true_tags = d.idx, d.text, d.tags

    result = {"id": idx, "text": text, "label": real_tags}

    return result


# FIXME
def decode_text_tags(full_tags):
    """
    根据实体抽取的结果，构造成业务应用需要的组合标注形式

    full_tags: 整句的标注
    {'text': "text", 'tags': [{'category': "category", 'start': 0, 'mention': "mention"}]}

    sent_tags_list: 所有分句的标注
    [
        {'text': "text", 'tags': [{'category': "category", 'start': 0, 'mention': "mention"}]}
        ......
    ]
    """

    real_tags = {}

    tags = full_tags

    for tag in tags:
        c, s, m = tag.c, tag.s, tag.m

        if c not in real_tags:
            real_tags[c] = {}
        if m not in real_tags[c]:
            real_tags[c][m] = []
        real_tags[c][m].append([s, s + len(m) - 1])

    return real_tags


def predict_test_file(test_file, best_model_file, results_file="results.json"):
    test_data = [x for x in test_data_generator(test_file)]

    # [(full_tags, sent_tags_list)]
    predictions = run_predicting(args, test_data, best_model_file, tokenizer)

    def decode_predictions(predictions):
        real_tags_list = []

        for full_tags, sent_tags_list in predictions:
            real_tags = decode_text_tags(full_tags, sent_tags_list)
            real_tags_list.append(real_tags)

        return real_tags_list

    real_tags_list = decode_predictions(predictions)

    assert len(test_data) == len(real_tags_list)

    predictions_file = "./predictions.json"
    with open(predictions_file, "w") as wt:
        for d, real_tags in zip(test_data, real_tags_list):
            idx, text, true_tags = d.idx, d.text, d.tags
            pred = {
                "idx": idx,
                "text": text,
                "tags": [tag.to_json() for tag in real_tags],
            }
            line = f"{json.dumps(pred, ensure_ascii=False)}\n"
            wt.write(line)
    print(f"Saved {len(real_tags_list)} lines in {predictions_file}")

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
    lines = open(data_file).readlines()
    for idx, line in enumerate(lines):
        line = line.strip()

        d = tag_text(idx, line)
        if idx < 5 or idx > len(lines) - 5:
            print(d)

        if args.do_split:
            sentences, sent_tags_list = split_text_tags(text, tags)
            for sent_text, sent_tags in zip(sentences, sent_tags_list):
                if idx < 5 or idx > len(lines) - 5:
                    print(idx, sent_text, sent_tags, others)
                yield TaggedData(idx, sent_text, sent_tags, others)
        else:
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
        print(f"----- load train_dataset")
        partial_train_data_generator = partial(
            train_data_generator, train_file=args.train_file
        )
        train_dataset = TaskDataset(args, partial_train_data_generator, tokenizer)

        print(f"----- load val_dataset")
        partial_val_data_generator = partial(val_data_generator, val_file=args.val_file)
        val_dataset = TaskDataset(args, partial_val_data_generator, tokenizer)

        print(f"----- run_training()")
        run_training(args, train_dataset, val_dataset)

    if args.do_eval:
        partial_val_data_generator = partial(val_data_generator, val_file=args.val_file)
        val_dataset = TaskDataset(args, partial_val_data_generator, tokenizer)

        run_evaluating(args, val_dataset)

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
    parser.add_argument(
        "--bert_model_path",
        type=str,
        default=bert_model_path,
        help="The BERT models path.",
    )

    cmd_args, unknown_args = parser.parse_known_args()

    os.makedirs(cmd_args.output_dir, exist_ok=True)
    os.makedirs(cmd_args.saved_models_dir, exist_ok=True)

    return cmd_args, unknown_args


if __name__ == "__main__":
    cmd_args, _ = get_args()
    args.update(**cmd_args.__dict__)
    main(args)
