#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, json
from tqdm import tqdm
from loguru import logger
from functools import partial
from copy import deepcopy
import numpy as np
import random

random.seed(42)

try:
    import rich

    def print(*arg, **kwargs):
        rich.print(*arg, **kwargs)


except:
    pass

# FIXME
TASK_NAME = "relation_extraction"

script_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
print(f"script_path: {script_path}")

theta_path = os.environ.get("THETA_HOME", os.path.realpath(f"{script_path}/.."))
print(f"theta_path: {theta_path}")

if theta_path and theta_path not in sys.path:
    sys.path.insert(0, theta_path)

from theta.nlp import  get_default_tokenizer
from theta.nlp.relation_extraction import TaskLabels, TaggedData, TaskTag, SubjectTag, ObjectTag
from theta.nlp.relation_extraction.runner import run_training, run_evaluating, run_predicting

from theta.nlp.relation_extraction.gplinker import TaskDataset, Model, Evaluator

data_path = os.path.realpath(f"{script_path}/data")
print(f"data_path: {data_path}")

#  bert_model_path = os.path.realpath(f"{script_path}/../pretrained/bert-base-chinese")
bert_model_path = "/opt/local/pretrained/bert-base-chinese"
print(f"bert_model_path: {bert_model_path}")

dict_path = f"{bert_model_path}/vocab.txt"
print(f"dict_path: {dict_path}")

tokenizer = get_default_tokenizer(dict_path)

# FIXME

entity_labels = []
relation_labels = []
relations_map = {
    "predicate": ("subject", "object"),
}
task_labels = TaskLabels(
    entity_labels=entity_labels,
    relation_labels=relation_labels,
    relations_map=relations_map,
)


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
        task_name=TASK_NAME,
        task_labels=task_labels,
        bert_model_path=bert_model_path,
        extract_threshold=0,
        debug=False,
        SEED=42,
        max_length=500,
        # 是否将文本划分为多个短句
        do_split=False,
        # Training
        batch_size=8,
        learning_rate=2e-5,
        num_training_epochs=500,
        max_training_episodes=100,
        min_best=0.9,
        # Early Stopping
        earlystopping_monitor="best_f1",
        earlystopping_patience=10,
        earlystopping_mode="max",
        # Predicting
        repeat=3,
        # Data files
        train_file=f"{data_path}/rawdata/train-652346.json",
        val_file=f"{data_path}/rawdata/train-652346.json",
        test_file=f"{data_path}/rawdata/evalA-428907.json",
    )
)


def split_text(text, sep="。"):
    sentences = []

    text = text.replace(sep, f"\1{sep}")
    for sent in text.split(sep):
        sent = sent.replace("\1", sep)
        sentences.append(sent)

    #  assert "".join(sentences) == text, f'[{"".join(sentences)}] vs [{text}]'

    return sentences


def split_sentences(sentences, sep):
    final_sentences = []

    for sent in sentences:
        sents = split_text(sent, sep)
        final_sentences.extend(sents)

    return final_sentences


def split_text_tags(text, tags):
    sentences = split_text(text, sep="。")
    sentences = split_sentences(sentences, sep="；")

    relations = [tag["predicate"] for tag in tags]
    subject_tags = [tag["subject"] for tag in tags]
    object_tags = [tag["object"] for tag in tags]

    offset = 0
    sent_tags_list = []

    for sent_text in sentences:
        sent_s = offset
        sent_e = sent_s + len(sent_text)

        sent_tags = []

        #  for tag in tags:
        #      s, p, o = tag["subject"], tag["predicate"], tag["object"]
        #
        #      s_s = s["start"]
        #      s_e = s_s + len(s["mention"])
        #      o_s = o["start"]
        #      o_e = o_s + len(o["mention"])
        #
        #      if s_s >= sent_s and s_s <= sent_e and s_e >= sent_s and s_e <= sent_e:
        #          if o_s >= sent_s and o_s <= sent_e and o_e >= sent_s and o_e <= sent_e:
        #              s = deepcopy(s)
        #              o = deepcopy(o)
        #              s["start"] -= offset
        #              o["start"] -= offset
        #              sent_tags.append({"subject": s, "predicate": p, "object": o})
        #          else:
        #              print(
        #                  f"object {o} ({o_s}, {o_e}) tag {tag} not found in {sent_text}"
        #              )
        #
        #  sent_tags = sorted(sent_tags, key=lambda x: x["subject"]["start"])

        for tag in tags:
            s, p, o = tag.s, tag.p, tag.o

            s_s = tag.s.s
            s_e = s_s + len(tag.s.m)
            o_s = tag.o.s
            o_e = o_s + len(tag.o.m)

            if s_s >= sent_s and s_s <= sent_e and s_e >= sent_s and s_e <= sent_e:
                if o_s >= sent_s and o_s <= sent_e and o_e >= sent_s and o_e <= sent_e:
                    s = deepcopy(s)
                    o = deepcopy(o)

                    s.s -= offset
                    o.s -= offset

                    sent_tags.append({"subject": s, "predicate": p, "object": o})
                else:
                    print(
                        f"object {o} ({o_s}, {o_e}) tag {tag} not found in {sent_text}"
                    )

        sent_tags = sorted(sent_tags, key=lambda x: x.s.s)

        sent_tags_list.append(sent_tags)

        offset = sent_e

    return sentences, sent_tags_list


# FIXME
def tag_text(idx, line):
    """
    根据原始数据的json数据，构建模型标准数据格式 idx, text, tags, others
    """
    json_data = json.loads(line)

    idx = json_data["ID"]
    text = json_data["text"]
    spo_list = json_data.get("spo_list", [])

    tags = []
    for spo in spo_list:
        h, t, rel = spo["h"], spo["t"], spo["relation"]
        h_m, (h_s, h_e) = h["name"], h["pos"]
        t_m, (t_s, t_e) = t["name"], t["pos"]
        h_c, t_c = relations_map[rel]

        tag = TaskTag(
            s=SubjectTag(c=h_c, s=h_s, m=h_m),
            p=rel,
            o=ObjectTag(c=t_c, s=t_s, m=t_m),
        )

        tags.append(tag)

    tags = sorted(tags, key=lambda x: x.s.s)

    #  print("idx:", idx, "text:", text, "tags:", tags)
    return TaggedData(idx, text, tags, None)


# FIXME
def build_final_result(d, real_tags):
    """
    根据预测结果构造成上层应用需要的输出格式
    """
    idx, text, true_tags = d.idx, d.text, d.tags

    spo_list = []
    for real_tag in real_tags:
        #  print("real_tag:", real_tag)

        #  rel = real_tag["predicate"]
        #  h = real_tag["subject"]
        #  h_c, h_s, h_m = h["category"], h["start"], h["mention"]
        #  h_e = h_s + len(h_m)
        #
        #  t = real_tag["object"]
        #  t_c, t_s, t_m = t["category"], t["start"], t["mention"]
        #  t_e = t_s + len(t_m)

        rel = real_tag.p
        h = real_tag.s
        h_c, h_s, h_m = h.c, h.s, h.m
        h_e = h_s + len(h_m)

        t = real_tag.o
        t_c, t_s, t_m = t.c, t.s, t.m
        t_e = t_s + len(t_m)

        #  h_c, t_c = relations_map[rel]

        spo = {
            "h": {"name": h_m, "pos": [h_s, h_e]},
            "t": {"name": t_m, "pos": [t_s, t_e]},
            "relation": rel,
        }
        spo_list.append(spo)

    result = {"ID": idx, "text": text, "spo_list": spo_list}

    return result


# FIXME
def decode_text_tags(full_tags):
    """
    根据实体抽取的结果，构造成业务应用需要的组合标注形式
    """

    real_tags = []

    real_tags = full_tags

    return real_tags


def predict_test_file(test_file, task_model_file, results_file="results.json"):
    test_data = [x for x in test_data_generator(test_file)]

    # [(full_tags, sent_tags_list)]
    predictions = run_predicting(
        args,
        Model, Evaluator,
        test_data,
        task_model_file,
        tokenizer
    )

    def decode_predictions(predictions):
        real_tags_list = []

        for full_tags in predictions:
            real_tags = decode_text_tags(full_tags)
            real_tags_list.append(real_tags)

        return real_tags_list

    real_tags_list = decode_predictions(predictions)
    assert len(test_data) == len(real_tags_list)

    predictions_file = "./predictions.json"
    with open(predictions_file, "w") as wt:
        for d, real_tags in zip(test_data, real_tags_list):
            # print("real_tags:", real_tags)
            idx, text, true_tags = d.idx, d.text, d.tags
            pred = {
                "idx": idx,
                "text": text,
                "tags": [tag.to_json() for tag in real_tags],
            }
            # print("pred:", pred)
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
    for idx, line in enumerate(tqdm(lines, desc=os.path.basename(data_file))):
        line = line.strip()

        d = tag_text(idx, line)
        if idx < 5 or idx > len(lines) - 5:
            print(d)

        if args.do_split:
            #  _, text, tags, others = d
            text, tags, others = d.text, d.tags, d.others

            sentences, sent_tags_list = split_text_tags(text, tags)
            for sent_text, sent_tags in zip(sentences, sent_tags_list):

                if idx < 5 or idx > len(lines) - 5:
                    print(idx, sent_text, sent_tags, others)
                yield TaggedData(idx, sent_text, sent_tags, others)
        else:
            yield d


def prepare_raw_train_data(args, data_generator, train_ratio=0.8):
    raw_train_data = [d for d in data_generator(args.train_file)]
    raw_train_data = random.sample(raw_train_data, len(raw_train_data))
    num_train_samples = int(len(raw_train_data) * train_ratio)

    return raw_train_data, num_train_samples


raw_train_data, num_train_samples = prepare_raw_train_data(args, data_generator, train_ratio=0.8)

def train_data_generator(train_file):
    for d in raw_train_data[:num_train_samples]:
        # for d in raw_train_data:
        yield d


def val_data_generator(val_file):
    for d in raw_train_data[num_train_samples:]:
        # for d in raw_train_data:
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
        run_training(args, Model, Evaluator, train_dataset, val_dataset)

    if args.do_eval:
        partial_val_data_generator = partial(val_data_generator, val_file=args.val_file)
        val_dataset = TaskDataset(args, Model, Evaluator, partial_val_data_generator, tokenizer)

        run_evaluating(args, val_dataset)

    if args.do_predict:
        if args.task_model_file is None:
            task_model_file = "best_model.pt"
        else:
            task_model_file = args.task_model_file

        predict_test_file(args.test_file, task_model_file)


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", default=TASK_NAME, help="The task name.")

    parser.add_argument(
        "--debug", action="store_true", help="Whether to show debug messages."
    )

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
        help="The BERT model path.",
    )
    parser.add_argument(
        "--task_model_file",
        type=str,
        default="best_model.pt",
        help="The task model file.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )

    cmd_args, unknown_args = parser.parse_known_args()

    os.makedirs(cmd_args.output_dir, exist_ok=True)
    os.makedirs(cmd_args.saved_models_dir, exist_ok=True)

    if unknown_args:
        print("unknown_args:", unknown_args)

    return cmd_args, unknown_args


if __name__ == "__main__":
    cmd_args, _ = get_args()
    args.update(**cmd_args.__dict__)
    main(args)
