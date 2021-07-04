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
from theta.nlp.data.samples import EntitySamples
from theta.nlp.tasks import NerData, NerTask

from ner_data import ner_labels, prepare_samples


class MyTask(NerTask):
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
        #  logits = test_results['logits']
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
        final_submissions = []
        for e in tqdm(test_samples):
            guid, text, _, _ = e
            #  logger.warning(f"{guid}: {text}")
            spans = preds[guid]

            tags = []
            for c, s, e in spans:
                m = text[s:e + 1]
                if len(m) == 0:
                    continue
                tags.append({
                    'category': id2label[c],
                    'start': s,
                    'mention': m
                })
            tags = sorted(tags, key=lambda x: x['start'])
            #  rich.print(tags)
            final_results.append({'idx': guid, 'text': text, 'tags': tags})

        # -------------------- 保存最终结果 --------------------

        submission_file = self.get_latest_submission_file(ext="json")
        predication_file = re.sub("submission_", "prediction_",
                                  submission_file)
        json.dump(final_results,
                  open(predication_file, 'w'),
                  ensure_ascii=False,
                  indent=2)
        logger.warning(
            f"Saved {len(final_results)} lines in {predication_file}")

        with open(submission_file, 'w') as wt:
            for submission in final_submissions:
                line = json.dumps(submission, ensure_ascii=False)
                wt.write(f"{line}\n")

        logger.warning(
            f"Saved {len(final_submissions)} lines in {submission_file}")


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
    task_data = NerData(task_args.data_args, train_samples, test_samples)

    # -------------------- task --------------------
    task = MyTask(task_args, task_data, ner_labels)
    task.execute()


if __name__ == '__main__':
    run()
