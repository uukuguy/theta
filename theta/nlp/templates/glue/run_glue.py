#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, re, json

from tqdm import tqdm
from loguru import logger

#  os.environ['PYTHONPATH'] = os.path.abspath(os.path.curdir)
if 'THETA_HOME' in os.environ:
    import sys
    sys.path.insert(0, os.environ['THETA_HOME'])
from theta.nlp.arguments import TaskArguments, TrainingArguments
from theta.nlp.data.samples import GlueSamples
from theta.nlp.tasks import GlueData, GlueTask

from glue_data import glue_labels, prepare_samples


class MyTask(GlueTask):
    def __init__(self, *args, **kwargs):
        super(MyTask, self).__init__(*args, **kwargs)

    def execute(self, *args, **kwargs):
        model_args = self.model_args
        data_args = self.data_args
        training_args = self.training_args
        remaining_args = self.remaining_args

        # TODO 在此处响应自定义命令行参数
        if training_args.do_something:
            """
            Do something
            """
            pass

        super(MyTask, self).execute(*args, **kwargs)

    # TODO 将模型推理结果转换成任务最终输出格式
    def generate_submission(self):
        """
        """

        # -------------------- 载入模型推理结果 --------------------
        test_results = self.load_test_results()
        preds = test_results['preds']
        logits = test_results['logits']
        id2label = self.runner.id2label

        # -------------------- 载入测试数据集 --------------------
        self.data.load_test_data()
        test_samples = self.data.test_samples

        assert len(preds) == len(
            test_samples
        ), f"len(preds): {len(preds), len(test_samples): {len(test_samples)}}"

        # -------------------- 转换最终输出格式 --------------------
        # 转换最终输出格式
        final_results = []
        #  final_submissions = []
        for index, ((idx, text_a, text_b, _),
                    pred) in enumerate(zip(test_samples, preds)):
            label = id2label[pred]
            #  final_results.append(f"{guid},{label}\n")
            final_results.append({
                'idx': idx,
                'text_a': text_a,
                'text_b': text_b,
                'labels': label
            })

            #  final_submissions.append({
            #      'idx': idx,
            #      'text_a': text_a,
            #      'text_b': text_b,
            #      'labels': label
            #  })

        # -------------------- 保存最终结果 --------------------

        submission_file = self.get_latest_submission_file(ext="json")
        prediction_file = re.sub("submission_", "prediction_", submission_file)

        json.dump(final_results,
                  open(prediction_file, 'w'),
                  ensure_ascii=False,
                  indent=2)
        logger.warning(
            f"Saved {len(final_results)} lines in {prediction_file}")

        #  json.dump(final_submissions,
        #            open(submission_file, 'w'),
        #            ensure_ascii=False,
        #            indent=2)
        #  logger.info(f"Saved {len(preds)} lines in {submission_file}")

        return prediction_file, submission_file


def get_task_args():
    """
    """
    from dataclasses import dataclass, field

    @dataclass
    class CustomTrainingArguments(TrainingArguments):
        """
        """
        # TODO 自定义需要的命令行参数
        do_something: bool = field(default=False,
                                   metadata={"help": "Do something"})

    task_args = TaskArguments.parse_args(
        training_args_cls=CustomTrainingArguments)

    return task_args


def run():
    task_args = get_task_args()

    # -------------------- Data --------------------
    train_samples, test_samples = prepare_samples()
    task_data = GlueData(task_args.data_args, train_samples, test_samples)

    # -------------------- Task --------------------
    task = MyTask(task_args, task_data, glue_labels)
    task.execute()


if __name__ == '__main__':
    run()
