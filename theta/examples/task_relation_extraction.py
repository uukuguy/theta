#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, json
from tqdm import tqdm
import random
from copy import deepcopy

script_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
print(f"script_path: {script_path}")

# -------------------- Theta --------------------
# pip install -U theta

from theta.utils import DictObject

# from theta.nlp.entity_extraction import TaskLabels, TaggedData, TaskTag, SubjectTag, ObjectTag
# from theta.nlp.entity_extraction import TaskDataset, Model, Evaluator
# from theta.nlp.entity_extraction.runner import run_training, run_evaluating, run_predicting
# from theta.nlp.entity_extraction.utils import split_text_tags
# # 实体标签
# entity_labels = []
# task_labels = TaskLabels(
#     entity_labels=entity_labels,
# )

from theta.nlp.relation_extraction import TaskLabels, TaggedData, TaskTag, SubjectTag, ObjectTag
from theta.nlp.relation_extraction import TaskDataset, Model, Evaluator
from theta.nlp.relation_extraction.runner import run_training, run_evaluating, run_predicting
from theta.nlp.relation_extraction.utils import split_text_tags
# 关系头尾实体标签
entity_labels = []
# 关系标签
relation_labels = []
# 关系与头尾实体对应关系
relations_map = {
    "predicate": ("subject", "object"),
}
task_labels = TaskLabels(
    entity_labels=entity_labels,
    relation_labels=relation_labels,
    relations_map=relations_map,
)


# -------------------- Global Variables --------------------

TASK_NAME = "relation_extraction"

#  bert_model_path = os.path.realpath(f"{script_path}/pretrained/bert-base-chinese")
bert_model_path = "/opt/local/pretrained/bert-base-chinese"
print("bert_model_path:", bert_model_path)

data_path = os.path.realpath(f"{script_path}/data")
print("data_path:", data_path)

seed = 42
learning_rate = 2e-5
batch_size = 8
max_length = 512
do_split = False
min_best = 0.9
earlystopping_patience=10

args = DictObject(
    **dict(
        task_name=TASK_NAME,
        task_labels=task_labels,
        bert_model_path=bert_model_path,
        extract_threshold=0,
        debug=False,
        seed=seed,
        max_length=max_length,
        # 是否将文本划分为多个短句
        do_split=do_split,
        # Training
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_training_epochs=500,
        max_training_episodes=100,
        min_best=min_best,
        # Early Stopping
        earlystopping_monitor="best_f1",
        earlystopping_patience=earlystopping_patience,
        earlystopping_mode="max",
        # Predicting
        repeat=1,
        # Data files
        train_file=f"{data_path}/rawdata/train-652346.json",
        val_file=f"{data_path}/rawdata/train-652346.json",
        test_file=f"{data_path}/rawdata/evalA-428907.json",
        task_model_file="best_model.pt",
    )
)

random.seed(seed)


# -------------------- tag_text() --------------------
def tag_text(idx, line):
    """
    由原始文本行生成返回标准的TaggedData标注对象

    # 关系抽取任务标注样本数据结构
    @dataclass
    class TaggedData:
        idx: str = ""
        text: str = ""
        tags: List[TaskTag] = field(default_factory=list)
        metadata: Any = None  # 应用侧自行定义的附加标注信息

    # 关系抽取任务标注结构
    @dataclass
    class TaskTag:
        s: SubjectTag = None  # subject: 关系的主实体
        p: str = None  # predicate: 关系类别
        o: ObjectTag = None  # object: 关系的客实体

    # 实体标注结构
    @dataclass
    class EntityTag:
        c: str = None  # category: 实体类别
        s: int = -1  # start: 文本片断的起始位置
        m: str = None  # mention: 文本片断内容

    # 关系的主实体结构
    class SubjectTag(EntityTag):
        pass

    # 关系的客实体结构
    @dataclass
    class ObjectTag(EntityTag):
        pass

    """

    json_data = json.loads(line)


    idx = json_data['idx']
    text = json_data['text']
    spo_list = json_data['tags']

    tags = []
    for spo in spo_list:
        tag = TaskTag().from_json(spo)
        # # (h_c, h_s, h_m), rel, (t_c, t_s, t_m)
        # tag = TaskTag(
        #     s=SubjectTag(c=h_c, s=h_s, m=h_m),
        #     p=rel,
        #     o=ObjectTag(c=t_c, s=t_s, m=t_m),
        # )

        tags.append(tag)

    tags = sorted(tags, key=lambda x: x.subject.start)

    #  print("idx:", idx, "text:", text, "tags:", tags)
    return TaggedData(idx, text, tags, None)


# -------------------- build_final_result() --------------------
def build_final_result(tagged_text, decoded_tags):
    """
    根据预测结果构造生成应用需要的输出格式（对应一行原始文本）

    tagged_text: TaggedData # tag_text()函数生成的 TaggedData 标注对象
    decoded_tags: List[TaskTag] # decode_text_tags()函数返回的解码后的标注对象列表，通常是TaskTag对象列表，对象结构见tag_text()函数。

    """

    idx, text = tagged_text.idx, tagged_text.text
    # true_tags, metadata = tagged_text.tags, tagged_text.metadata

    spo_list = []
    for tag in decoded_tags:

        s = tag.subject
        s_c, s_s, s_m = s.category, s.start, s.mention
        s_e = s_s + len(s_m)


        o = tag.object
        o_c, o_s, o_m = o.category, o.start, o.mention
        o_e = o_s + len(o_m)

        p = tag.predicate
        #  s_c, o_c = relations_map[rel]
        assert (s_c, o_c) == relations_map[p]

        spo = {
            "subject": {"name": s_m, "pos": [s_s, s_e]},
            "predicate": p,
            "object": {"name": o_m, "pos": [o_s, o_e]},
        }
        spo_list.append(spo)

    result = {"ID": idx, "text": text, "spo_list": spo_list}

    return result


# -------------------- decode_text_tags() --------------------

def decode_text_tags(pred_tags):
    """
    模型预测出来的 TaskTag 对象，根据实际需要，可以在这里做解码变换，要求返回相同结构的TaskTag对象。
    无需解码时，直接返回pred_tags。
    """

    decoded_tags = pred_tags

    return decoded_tags


# -------------------- predict_test_file() --------------------
"""
"""

def predict_test_file(args,  tokenizer, results_file="results.json"):
    test_file = args.test_file
    task_model_file = args.task_model_file

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
        decoded_tags_list = []

        for pred_tags in predictions:
            decoded_tags = decode_text_tags(pred_tags)
            decoded_tags_list.append(decoded_tags)

        return decoded_tags_list

    decoded_tags_list = decode_predictions(predictions)
    assert len(test_data) == len(decoded_tags_list)

    # predictions_file = "./predictions.json"
    # with open(predictions_file, "w") as wt:
    #     for tagged_text, tags in zip(test_data, decoded_tags_list):
    #         idx, text, true_tags = tagged_text.idx, tagged_text.text
    #         # true_tags, metadata = tagged_text.tags, tagged_text.metadata
    #         pred = {
    #             "idx": idx,
    #             "text": text,
    #             "tags": [tag.to_json() for tag in tags],
    #         }
    #         line = f"{json.dumps(pred, ensure_ascii=False)}\n"
    #         wt.write(line)
    # print(f"Saved {len(decoded_tags_list)} lines in {predictions_file}")
    predictions_file = "./predictions.json"
    with open(predictions_file, "w") as wt:
        for tagged_text, tags in zip(test_data, decoded_tags_list):
            # idx, text, true_tags = tagged_text.idx, tagged_text.text
            # true_tags, metadata = tagged_text.tags, tagged_text.metadata
            # pred = {
            #     "idx": idx,
            #     "text": text,
            #     "tags": [tag.to_json() for tag in tags],
            # }
            # line = f"{json.dumps(pred, ensure_ascii=False)}\n"
            pred_tagged_text = deepcopy(tagged_text)
            pred_tagged_text.tags = tags
            line = pred_tagged_text.to_json()
            wt.write(f"{line}\n")
    print(f"Saved {len(decoded_tags_list)} lines in {predictions_file}")

    final_tags_list = []
    for tagged_text, decoded_tags in zip(test_data, decoded_tags_list):
        final_tags = build_final_result(tagged_text, decoded_tags)
        final_tags_list.append(final_tags)

    with open(results_file, "w") as wt:
        for d in final_tags:
            line = json.dumps(d, ensure_ascii=False)
            wt.write(f"{line}\n")
    print(f"Saved {len(final_tags_list)} results in {results_file}")

    return final_tags_list


def data_generator(data_file, do_split=False):
    lines = open(data_file).readlines()
    for idx, line in enumerate(tqdm(lines, desc=os.path.basename(data_file))):
        line = line.strip()

        tagged_text = tag_text(idx, line)
        if idx < 5 or idx > len(lines) - 5:
            print(tagged_text)

        if args.do_split:
            text, tags, others = tagged_text.text, tagged_text.tags, tagged_text.others

            sentences, sent_tags_list = split_text_tags(tagged_text.text, tagged_text.tags)
            for sent_text, sent_tags in zip(sentences, sent_tags_list):

                if idx < 5 or idx > len(lines) - 5:
                    print(idx, sent_text, sent_tags, others)
                yield TaggedData(idx, sent_text, sent_tags, others)
        else:
            yield tagged_text


def prepare_raw_train_data(args, data_generator, train_ratio=0.8):
    raw_train_data = [d for d in data_generator(args.train_file)]
    raw_train_data = random.sample(raw_train_data, len(raw_train_data))
    num_train_samples = int(len(raw_train_data) * train_ratio)

    return raw_train_data, num_train_samples


raw_train_data, num_train_samples = prepare_raw_train_data(args, data_generator, train_ratio=0.8)

def train_data_generator(train_file):
    for data in raw_train_data[:num_train_samples]:
        # for data in raw_train_data:
        yield data


def val_data_generator(val_file):
    for data in raw_train_data[num_train_samples:]:
        # for data in raw_train_data:
        yield data


def test_data_generator(test_file):
    for data in data_generator(test_file):
        yield data


def main(args):
    from functools import partial
    from theta.nlp import  get_default_tokenizer

    dict_path = f"{args.bert_model_path}/vocab.txt"
    print(f"dict_path: {dict_path}")
    tokenizer = get_default_tokenizer(dict_path)

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
            args.task_model_file = "best_model.pt"

        predict_test_file(args, tokenizer)


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

    # parser.add_argument(
    #     "--train_file", type=str, default=args.train_file, help="Train file"
    # )
    # parser.add_argument("--val_file", type=str, default=args.val_file, help="Val file")
    # parser.add_argument(
    #     "--test_file", type=str, default=args.test_file, help="Test file"
    # )

    # parser.add_argument("--seed", type=int, default=42, help="SEED")
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
    # parser.add_argument(
    #     "--bert_model_path",
    #     type=str,
    #     default=bert_model_path,
    #     help="The BERT model path.",
    # )
    # parser.add_argument(
    #     "--task_model_file",
    #     type=str,
    #     default="best_model.pt",
    #     help="The task model file.",
    # )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )

    for k, v in args.items():
        parser.add_argument(f"--{k}", default=v, help=k.replace('_', ' '))

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
