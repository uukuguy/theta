#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, json
from tqdm import tqdm
from loguru import logger
from theta.nlp.glue_utils import GlueRunner

# -------------------- Customize --------------------
glue_labels = []
train_data_file = "./data/train.tsv"
test_data_file = "./data/test.tsv"

from theta.nlp.glue_utils import (
    glue_tsv_data_generator,
    glue_json_data_generator,
    glue_jsonl_data_generator,
)

train_data_generator = glue_tsv_data_generator(train_data_file)
test_data_generator = glue_tsv_data_generator(test_data_file)

#  def train_data_generator():
#      yield idx, text_a, text_b, label

#  def test_data_generator():
#      yield idx, text, None, None


# -------------------- Runner --------------------
class AppRunner(GlueRunner):

    def do_submit(self):
        # -------------------- 载入模型推理结果 --------------------
        test_results = self.load_test_results()
        preds = test_results['preds']
        logits = test_results['logits']
        id2label = self.id2label

        # -------------------- 载入测试数据集 --------------------
        test_data = self.load_test_data()
        assert len(preds) == len(test_data), f"len(preds): {len(preds), len(test_data): {len(test_data)}}"

        # -------------------- 转换最终输出格式 --------------------
        # 转换最终输出格式
        final_results = []
        final_submissions = []
        for index, ((idx, text_a, text_b, _), pred) in enumerate(zip(test_data, preds)):
            label = id2label[pred]
            final_results.append({
                'idx': idx,
                'text_a': text_a,
                'text_b': text_b,
                'label': label
            })

            #  # TODO
            #  final_submissions.append({
            #      'idx': idx,
            #      'text_a': text_a,
            #      'text_b': text_b,
            #      'label': label
            #  })

        # -------------------- 保存最终结果 --------------------

        timestamp_filename = self.get_timestamp_filename()

        prediction_file = os.path.join(self.training_args.submissions_dir, f"prediction_{timestamp_filename}.json")
        json.dump(final_results, open(prediction_file, 'w'), ensure_ascii=False, indent=2)
        logger.warning(f"Saved {len(final_results)} lines in {prediction_file}")

        #  # TODO
        #  submission_file = os.path.join(self.training_args.submissions_dir, f"submission_{timestamp_filename}.json")
        #  json.dump(final_submissions, open(submission_file, 'w'), ensure_ascii=False, indent=2)
        #  logger.info(f"Saved {len(preds)} lines in {submission_file}")

        return {
            'prediction_file': prediction_file,
            'submission_file': submission_file
        }

    #  def configure_model(self):
    #      return super.configure_model()

    #  def configure_optimizer(self):
    #      return super.configure_optimizer()

    #  def configure_scheduler_fn(self):
    #      return super.configure_scheduler_fn()

    #  def configure_loss_fn(self):
    #      return super.configure_loss_fn()

    def before_execute(self):
        pass

    def after_execute(self):
        if self.do_something:
            print("do_something() Done.")


# -------------------- Arguments --------------------

# Customezied Arguments
from theta.nlp.arguments import TrainingArguments

from dataclasses import dataclass


@dataclass
class CustomTrainingArguments(TrainingArguments):
    """
    """
    from dataclasses import dataclass, field

    # TODO 自定义需要的命令行参数
    do_something: bool = field(default=False, metadata={"help": "Do something"})


def get_task_args():
    from theta.nlp.arguments import TaskArguments
    return TaskArguments.parse_args(training_args_cls=CustomTrainingArguments)


# -------------------- Main --------------------
def main():
    task_args = get_task_args()
    runner = AppRunner(
        task_args,
        glue_labels=glue_labels,
        train_data_generator=train_data_generator,
        test_data_generator=test_data_generator,
    )
    runner.execute()
