#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
from copy import deepcopy
from typing import Type, Union

import dill
import numpy as np
import pytorch_lightning as pl
import torch
import torch.functional as F
import torch.nn as nn
from loguru import logger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from theta.nlp.arguments import (DataArguments, ModelArguments, TaskArguments,
                                 TrainingArguments,
                                 create_instance_from_arguments,
                                 generate_method_kwargs_from_arguments)
from transformers import (AdamW, AutoConfig, AutoModel, AutoTokenizer,
                          ElectraConfig, XLNetConfig)
from transformers.optimization import (
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup)

from ...utils import seed_everything


# ------------------------------ Dataset ------------------------------
class BaseDataset(object):
    """
        Dataset负责完成传入模型数据的编码工作
        编码data_generator生成的数据
    """
    def __init__(self, data_args, data_generator, label2id, tokenizer):
        super(BaseDataset, self).__init__()
        self.data_args = data_args
        self.label2id = label2id
        self.tokenizer = tokenizer

        self.encoded_data_list = [
            self._encode_item(x)
            for x in tqdm(data_generator(), desc='Encoding')
        ]

    def _encode_item(self, x):
        raise NotImplementedError

    def __iter__(self):
        for x in self.encoded_data_list:
            yield x

    def __getitem__(self, idx):
        return self.encoded_data_list[idx]

    def __len__(self):
        return len(self.encoded_data_list)


# ------------------------------ TaskData ------------------------------
class TaskData:
    """
    TaskData负责完成Samples -> Dataset的转换
    对模型提供train_dataset、val_dataset、test_dataset
    """
    def __init__(self,
                 data_args: Type[DataArguments],
                 train_samples,
                 test_samples,
                 tokenizer=None):
        self.data_args = data_args
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.tokenizer = tokenizer

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

        assert train_samples.label2id == test_samples.label2id
        self.label2id = train_samples.label2id

    def load_train_data(self):
        self.train_samples.load_samples()

        all_train_samples = self.train_samples.shuffle()
        self._splitted_train_samples, self._splitted_val_samples = all_train_samples.split(
            ratios=self.data_args.split_ratios)
        logger.info(f"total train samples: {len(self.train_samples)}")
        logger.info(f"train samples: {len(self._splitted_train_samples)}")
        logger.info(f"val samples: {len(self._splitted_val_samples)}")

    def load_test_data(self):
        self.test_samples.load_samples()
        logger.info(f"test samples: {len(self.test_samples)}")


def save_args(args, model_path):
    os.makedirs(model_path, exist_ok=True)
    args_file = os.path.join(model_path, "training_args.json")
    logger.info(f"Save args in {args_file}")
    json.dump(
        {
            k: v
            for k, v in args.__dict__.items() if v is None
            or type(v) in [bool, str, int, float, dict, list, tuple]
        },
        open(args_file, 'w'),
        ensure_ascii=False,
        indent=2)


# ------------------------------ TaskModel ------------------------------
class TransformerModel():
    def __init__(self, model_name_or_path, tokenizer=None):
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        logger.info(f"{self.config}")

        if tokenizer is None:
            if model_name_or_path not in ['clue/roberta_chinese_pair_large']:
                tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                          fast=True)
            else:
                # self.config.vocab_size == 21128
                tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese",
                                                          fast=True)
        self.tokenizer = tokenizer

        self.model_name_or_path = model_name_or_path
        self.model = None

    def load_model(self, model_path):
        self.model = AutoModel.from_pretrained(model_path)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

    def save_model(self, model_path):
        os.makedirs(model_path, exist_ok=True)
        model_to_save = (self.model.module
                         if hasattr(self.model, "module") else self.model)
        model_to_save.save_pretrained(model_path)

        self.tokenizer.save_vocabulary(os.path.abspath(model_path) + '/')

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def __call__(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        return outputs


class TaskModel(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super(TaskModel, self).__init__()
        self.save_hyperparameters()

        tokenizer = kwargs.get('tokenizer', None)
        model_name_or_path = self.hparams.model_name_or_path
        if model_name_or_path is None:
            model_name_or_path = os.path.join(self.hparams.task_dir,
                                              "checkpoint")
        self.transformer_model = TransformerModel(model_name_or_path,
                                                  tokenizer)

        self.warmup_steps = None
        self.total_steps = None

        self.best_score = 0.0 if self.hparams.greater_is_better else float(
            'inf')

    @property
    def tokenizer(self):
        return self.transformer_model.tokenizer

    def _is_xlnet(self):
        config = self.transformer_model.config
        if isinstance(config, XLNetConfig):
            return True
        else:
            return False
        #  return 'XLNetLMHeadModel' in config.architectures
    def _is_electra(self):
        if isinstance(self.transformer_model.config, ElectraConfig):
            return True
        else:
            return False

    def load_model(self, model_path):
        pl_model_checkpoint_file = f"{model_path}/pl_model.ckpt"
        if os.path.exists(pl_model_checkpoint_file):
            logger.info(f"Load PL module from {pl_model_checkpoint_file}.")
            checkpoint = dill.load(open(pl_model_checkpoint_file, 'rb'))
            self.load_state_dict(checkpoint['state_dict'])

        logger.info(f"Load transformer model from {model_path}")
        self.transformer_model.load_model(model_path)

    def save_model(self, model_path):
        os.makedirs(model_path, exist_ok=True)

        pl_model_checkpoint_file = f"{model_path}/pl_model.ckpt"

        # FIXME
        dill.dump({'state_dict': self.state_dict()},
                  open(pl_model_checkpoint_file, 'wb'))
        logger.warning(f"Save PL module in {pl_model_checkpoint_file}")

        self.transformer_model.save_model(model_path)

        logger.warning(f"Save transformer model in {model_path}")

    def configure_optimizers(self):
        param_optimizer = list(self.transformer_model.model.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate":
                self.hparams.weight_decay
            },
            {
                "params": [
                    p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate":
                0.0
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.learning_rate)

        #  return optimizer

        assert self.warmup_steps is not None
        assert self.total_steps is not None

        # Inspired from https://github.com/PyTorchLightning/pytorch-lightning/issues/328
        #  def lr_exp_warmup(steps):
        #      if steps < self.warmup_steps:
        #          lr_scale = 0.1**(self.warmup_steps - steps)
        #      else:
        #          lr_scale = 0.95**steps
        #
        #      return lr_scale
        #
        #  scheduler = LambdaLR(optimizer, lr_lambda=lr_exp_warmup)

        #  schedule_fn = get_linear_schedule_with_warmup
        #  schedule_fn = get_cosine_schedule_with_warmup
        schedule_fn = get_cosine_with_hard_restarts_schedule_with_warmup
        scheduler = schedule_fn(optimizer,
                                num_warmup_steps=self.warmup_steps,
                                num_training_steps=self.total_steps)

        scheduler_dict = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }

        return ([optimizer], [scheduler_dict])

    #  def training_step_end(self, batch_parts_outputs):
    #      if self.trainer.global_step > 0 and self.trainer.global_step % self.hparams.eval_steps == 0:
    #          self.trainer.run_evaluation()

    def on_train_start(self):
        self.transformer_model.train()

    def on_validation_start(self, model):
        self.transformer_model.eval()

    def on_validation_end(self, model):
        self.transformer_model.train()

    def on_test_start(self, model):
        self.transformer_model.eval()

    def on_test_end(self, model):
        self.transformer_model.train()

    def save_best_model(self, eval_outputs: dict):
        if self.hparams.metric_for_best_model in eval_outputs:
            curr_score = eval_outputs[self.hparams.metric_for_best_model]

            is_best = False
            if self.hparams.greater_is_better:
                if curr_score > self.best_score:
                    is_best = True
            else:
                if curr_score < self.best_score:
                    is_best = True

            if is_best:
                logger.warning(
                    f"Best {self.hparams.metric_for_best_model}: {curr_score:.4f} / {self.best_score:.4f}"
                )
                self.best_score = curr_score

                self.save_model(
                    os.path.join(self.hparams.task_dir, "checkpoint"))
            else:
                logger.info(
                    f"Eval result: {curr_score:.4f} / {self.best_score:.4f}")


# ------------------------------ Task ------------------------------


class BaseTask():
    """
    实验任务(ExperimentTask)
    定义一次任务中必须包含的数据、模型、参数。
    支持任务执行、加载复现。

    task_args = TaskArguments.parse_args()
    task_data = ExpData(data_args=task_args.data_args)
    task_model = ExpModel(model_args=task_args.model_args,
                          num_labels=len(glue_labels))

    task = ExpTask(args=task_args,
                   data=task_data,
                   model=task_model)

    task.execute(glue_labels)

    """
    def __init__(self, args: Type[TaskArguments], data: Type[TaskData],
                 model: Type[TaskModel]):
        self.args = args
        self.data = data
        self.model = model

        self.data.tokenizer = self.model.tokenizer

        os.environ['TOKENIZERS_PARALLELISM'] = "true"
        seed_everything(self.args.training_args.seed)  #固定随机种子
        #  self.args.training_args.device = torch.device(
        #      'cuda' if torch.cuda.is_available() else 'cpu')

        #  self.tokenizer = AutoTokenizer.from_pretrained(
        #      self.model_args.model_name_or_path, fast=True)
        #  self.data.tokenizer = self.tokenizer

    @property
    def rootdir(self):
        return self.args.rootdir

    @property
    def data_args(self):
        return self.args.data_args

    @property
    def model_args(self):
        return self.args.model_args

    @property
    def training_args(self):
        return self.args.training_args

    @property
    def remaining_args(self):
        return self.args.remaining_args

    @property
    def checkpoint_path(self):
        return os.path.join(self.training_args.task_dir, "checkpoint")

    @property
    def test_results_file(self):
        return os.path.join(self.training_args.task_dir, "test_results.pkl")

    def load_test_results(self, test_results_file=None):
        if test_results_file is None:
            test_results_file = self.test_results_file
        logger.info(f"Load test result from {test_results_file}")
        test_results = dill.load(open(test_results_file, 'rb'))
        return test_results

    def dump_test_results(self, test_results, test_results_file=None):
        if test_results_file is None:
            test_results_file = self.test_results_file
        dill.dump(test_results, open(test_results_file, 'wb'))
        logger.info(f"Dump test results to {test_results_file}.")

    @property
    def train_dataloader(self):
        train_dataset = self.data.train_dataset
        for index in random.sample(range(len(train_dataset)), 1):
            logger.info(
                f"Sample {index} of the training set: {train_dataset[index]}.")

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.training_args.per_device_train_batch_size,
            collate_fn=train_dataset.collate_fn,
            pin_memory=True,
            num_workers=8)
        return train_dataloader

    @property
    def val_dataloader(self):
        val_dataset = self.data.val_dataset
        for index in random.sample(range(len(val_dataset)), 1):
            logger.info(
                f"Sample {index} of the val set: {val_dataset[index]}.")

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.training_args.per_device_eval_batch_size,
            collate_fn=val_dataset.collate_fn,
            pin_memory=True,
            num_workers=8)
        return val_dataloader

    @property
    def test_dataloader(self):
        test_dataset = self.data.test_dataset
        for index in random.sample(range(len(test_dataset)), 1):
            logger.info(
                f"Sample {index} of the test set: {test_dataset[index]}.")

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.training_args.per_device_test_batch_size,
            collate_fn=test_dataset.collate_fn,
            pin_memory=True,
            num_workers=8)
        return test_dataloader

    def generate_submission(self):
        logger.warning(
            f"Call generate_submission() implemented in base class Task.")

    def execute(self, *args, **kwargs):

        model_args = self.model_args
        data_args = self.data_args
        training_args = self.training_args
        remaining_args = self.remaining_args

        # ------------------------------ do_train ------------------------------
        if training_args.do_train:
            model_path = model_args.model_name_or_path
            self.model.load_model(model_path)
            self.data.load_train_data()

            # warmup_steps
            if self.training_args.warmup_steps is None:
                if self.training_args.max_steps is not None:
                    self.training_args.warmup_steps = int(
                        self.training_args.max_steps / 10)
                else:
                    epoch_steps = int(
                        len(self.data.train_dataset) /
                        self.training_args.per_device_train_batch_size)
                    self.training_args.warmup_steps = int(
                        epoch_steps * self.training_args.max_epochs / 10)

            max_epochs = self.training_args.max_epochs
            if max_epochs:
                total_steps = int(
                    len(self.data.train_dataset) / self.training_args.
                    per_device_train_batch_size) * max_epochs
            else:
                total_steps = self.training_args.max_steps

            self.model.warmup_steps = self.training_args.warmup_steps
            self.model.total_steps = total_steps

            logger.warning(f"warmup_steps: {self.training_args.warmup_steps}")
            logger.warning(f"total_steps: {total_steps}")

            #  trainer = create_instance_from_arguments(
            #      pl.Trainer, self.training_args.to_dict())
            trainer_kwargs = generate_method_kwargs_from_arguments(
                pl.Trainer,
                method="__init__",
                args=self.training_args.to_dict())

            trainer_kwargs['precision'] = 16 if self.training_args.fp16 else 32
            # float: precent, int: steps
            trainer_kwargs['val_check_interval'] = 1.0
            trainer_kwargs['checkpoint_callback'] = False

            trainer = pl.Trainer(**trainer_kwargs)

            trainer.fit(self.model, self.train_dataloader, self.val_dataloader)

        # ------------------------------ do_eval ------------------------------
        if training_args.do_eval:
            self.model.load_model(self.checkpoint_path)
            self.data.load_train_data()

            val_dataloader = self.val_dataloader
            #  eval_results = do_eval(trainer, val_dataset, data_args)

        # ------------------------------ do_predict ------------------------------
        if training_args.do_predict:
            self.model.load_model(self.checkpoint_path)
            self.data.load_test_data()

            test_dataloader = self.test_dataloader

            trainer_kwargs = generate_method_kwargs_from_arguments(
                pl.Trainer,
                method="__init__",
                args=self.training_args.to_dict())

            trainer_kwargs['precision'] = 16 if self.training_args.fp16 else 32

            trainer = pl.Trainer(**trainer_kwargs)

            trainer.test(self.model, test_dataloaders=test_dataloader)

            self.dump_test_results(self.model.test_results)

        if training_args.do_submit:
            self.generate_submission()
