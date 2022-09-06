#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json, random
from tqdm import tqdm
from loguru import logger
from torch import nn
from transformers import AdamW
from functools import partial


# -------------------- Dataset --------------------
class BaseDataset(object):
    """
        Dataset负责完成传入模型数据的编码工作
        编码data_generator生成的数据
    """

    def __init__(self, data_args, data_generator, label2id, tokenizer):
        super(BaseDataset, self).__init__()

        self.data_args = data_args
        self.data_generator = data_generator
        self.label2id = label2id
        self.tokenizer = tokenizer
        self.encoded_data_list = []
        self.data_ids = []
        self.data_indices = []

    def load(self):
        for x in tqdm(self.data_generator(), desc="Encoding"):
            encoded = self._encode_item(x)
            self.encoded_data_list.append(encoded)
            self.data_ids.append(x[0])
        assert len(self.encoded_data_list) == len(self.data_ids)
        self.data_indices = list(range(self.__len__()))

    def _encode_item(self, x):
        raise NotImplementedError

    def __iter__(self):
        for i in self.data_indices:
            yield self.encoded_data_list[i]

    def __getitem__(self, idx):
        return self.encoded_data_list[self.data_indices[idx]]

    def __len__(self):
        return len(self.encoded_data_list)

    def shuffle(self):
        random.shuffle(self.data_indices)

    def split(self, ratios):
        if isinstance(ratios, float):
            ratios = [ratios]

        s = sum(ratios)
        assert s <= 1.0, f"split ratios: {ratios} sum: {s}"

        if s < 1.0:
            ratios.apend(1.0 - s)

        total_samples = len(self.encoded_data_list)
        segs = [int(total_samples * x) for r in ratios[:-1]]
        segs = [0] + segs

        splitted_datasets = []
        for i, s in enumerate(segs[:-1]):
            e = segs[i + 1]

            s_dataset = self.__class__(self.data_args, self.data_generator, self.label2id, self.tokenizer)
            s_dataset.encoded_data_list = self.encoded_data_list[s:e]
            s_dataset.data_ids = self.data_ids[s:e]
            s_dataset.data_indices = list(range(e - s))
            splitted_datasets.append(s_dataset)

        assert len(splitted_datasets) == len(ratios)

        return splitted_datasets


# -------------------- Optimizer --------------------


def configure_optimizer(traning_args, model, optim_cls=AdamW):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta"]
    optimizer_grouped_parameters = [{
        "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        "weight_decay_rate": traing_args.weight_decay
    }, {
        "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        "weight_decay_rate": 0.0
    }]
    optimizer = optim_cls(optimizer_grouped_parameters, lr=training_args.learning_rate)

    return optimizer


# -------------------- Scheduler Function --------------------
def configure_scheduler_fn(training_args):
    from transformers.optimization import (
        get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup,
        get_cosine_with_hard_restarts_schedule_with_warmup
    )

    scheduler_fn = partial(
        get_linear_schedule_with_warmup,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=training_args.total_steps,
    )

    return scheduler_fn


# -------------------- Loss Function --------------------
def configure_loss_fn(training_args):
    loss_fn = nn.CrossEntropyLoss()
    return loss_fn


# -------------------- Trainer --------------------
from .pytorch_accelerated import Trainer
from .pytorch_accelerated.callbacks import TrainerCallback


class SaveBestModelCallback(TrainerCallback):
    """
    """

    def __init__(self):
        self.pbar = None

    def on_train_epoch_start(self, trainer, **kwargs):
        self.pbar = tqdm(
            total=len(trainer._train_dataloader),
            disable=not trainer._accelerator.is_local_main_process,
        )

    def on_train_step_end(self, trainer, **kwargs):
        self.pbar.update(1)

    def on_train_epoch_end(self, trainer, **kwargs):
        self.pbar.close()
        time.sleep(0.01)

    def on_eval_epoch_start(self, trainer, **kwargs):
        self.pbar = tqdm(
            total=len(trainer._eval_dataloader),
            disable=not trainer._accelerator.is_local_main_process,
        )

    def on_eval_step_end(self, trainer, **kwargs):
        self.pbar.update(1)

    def on_eval_epoch_end(self, trainer, **kwargs):
        self.pbar.close()
        time.sleep(0.01)


class BaseTrainer(Trainer):
    from .pytorch_accelerated.callbacks import (
        CallbackHandler,
        LogMetricsCallback,
        PrintProgressCallback,
        TerminateOnNaNCallback,
        StopTrainingError,
        ProgressBarCallback,
        MoveModulesToDeviceCallback,
    )

    DEFAULT_CALLBACKS = (
        MoveModulesToDeviceCallback,
        TerminateOnNaNCallback,
        PrintProgressCallback,
        ProgressBarCallback,
        LogMetricsCallback,
    )

    def __init__(
        self,
        training_args,
        model,
        optimizer=None,
        scheduler_fn=None,
        loss_func=None,
        callbacks=DEFAULT_CALLBACKS,
        run_history=None,
    ):
        self.training_args = training_args

        if optimizer is None:
            optimizer = configure_optimizer(training_args, model.model)
        if scheduler_fn is None:
            scheduler_fn = configure_scheduler_fn(training_args)
        if loss_fn is None:
            loss_fn = configure_loss_fn(training_args)

        super(BaseTrainer, self).__init__(
            model,
            optimizer=optimizer,
            scheduler_fn=scheduler_fn,
            loss_func=loss_func,
            collate_fn=GlueDataset.collate_fn,
            callbacks=callbacks,
            run_history=run_history
        )

    def on_train_step_start(self):
        pass


# -------------------- Runner --------------------
class BaseRunner:

    def __init__(
        self,
        task_args,
        labels_list,
        train_data_generator=None,
        test_data_generator=None,
        eval_data_generator=None,
    ):
        self.task_args = task_args
        self.labels_list = labels_list
        self.train_data_generator = train_data_generator
        self.test_data_generator = test_data_generator
        self.eval_data_generator = eval_data_generator

    @property
    def model_args(self):
        return self.task_args.model_args

    @property
    def data_args(self):
        return self.task_args.data_args

    @property
    def training_args(self):
        return self.task_args.training_args

    @property
    def remaining_args(self):
        return self.args.remaining_args

    @property
    def latest_path(self):
        return self.training_args.latest_path

    @property
    def checkpoint_path(self):
        return self.model_args.checkpoint_path

    def load_dataset(self, shuffle=False, split=False):
        raise NotImplementedError

    def _load_dataset(self, dataset, shuffle=False, split=False):
        dataset.load()
        if shuffle:
            dataset.shuffle()
        if split:
            splitted_datasets = dataset.split(data_args.split_ratios)
            return splitted_datasets
        else:
            return dataset

    def configure_model(self):
        raise NotImplementedError

    def configure_optimizer(self):
        param_optimizer = list(self.model.bert.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [{
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay_rate": self.training_args.weight_decay
        }, {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay_rate": 0.0
        }]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.training_args.learning_rate)

        return optimizer

    def configure_scheduler_fn(self):
        from transformers.optimization import (
            get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup,
            get_cosine_with_hard_restarts_schedule_with_warmup
        )

        scheduler_fn = partial(
            get_linear_schedule_with_warmup,
            num_warmup_steps=self.training_args.warmup_steps,
            num_training_steps=self.training_args.total_steps,
        )

        return scheduler_fn

    def configure_loss_fn(self):
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn

    def _build_trainer(self):
        optimizer = self.configure_optimizer()
        scheduler_fn = self.configure_scheduler_fn()
        loss_fn = self.configure_loss_fn()
        trainer = GlueTrainer(
            self.training_args,
            self.model,
            optimizer=optimizer,
            scheduler_fn=scheduler_fn,
            loss_func=loss_fn,
        )

        return trainer

    def do_train(self):
        """
        """
        self.model = self.configure_model()
        train_dataset, eval_dataset = self.load_dataset(shuffle=True, split=True)

        self.training_args.total_steps = int(
            len(train_dataset) / (self.training_args.per_device_train_batch_size * gradient_accumulation_steps)
        )
        if self.warmup_steps is None:
            self.warmup_steps = int(self.training_args.total_steps * self.warmup_rate)

        self.optimizer = self.configure_optimizer()
        self.scheduler_fn = self.configure_scheduler_fn()
        self.loss_fn = self.configure_loss_fn()

        trainer = self._build_trainer()
        trainer.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            num_epochs=self.training_args.num_epochs,
            per_device_batch_size=self.training_args.per_device_train_batch_size
        )

        logger.info(f"do_train() Done.")

    def do_eval(self):
        """
        """
        self.model = self.configure_model()
        train_dataset, eval_dataset = self.load_dataset(shuffle=True, split=True)

        trainer = self._build_trainer()
        trainer.evaluate(
            dataset=eval_dataset,
            per_device_batch_size=training_args.per_device_eval_batch_size,
        )

        logger.info(f"do_eval() Done.")

    def do_predict(self):
        """
        """

        test_dataset = self.load_dataset(shuffle=False, split=False)

        trainer = self._build_trainer()
        trainer.predict(
            dataset=test_dataset,
            per_device_batch_size=training_args.per_device_test_batch_size,
        )

        logger.info(f"do_predict() Done.")

    def do_submit(self):
        raise NotImplementedError

    def before_execute(self):
        pass

    def after_execute(self):
        pass

    def execute(self):
        """
        """
        self.before_execute()

        if self.task_args.do_train:
            self.do_train()

        if self.task_args.do_eval:
            self.do_eval()

        if self.task_args.do_predict:
            self.do_predict()

        if self.task_args.do_submit:
            self.do_submit()

        self.after_execute()

    @property
    def test_results_file(self):
        return os.path.join(self.training_args.task_dir, "test_results.pkl")
        #  return os.path.join(self.training_args.latest_path, "test_results.pkl")

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

        return test_results_file

    def load_test_data(self):
        return [x for x in self.test_data_generator()]

    def get_timestamp_filename(self):
        task_id = self.training_args.task_id
        ts = datetime.now().strftime("%Y%m%d.%H%M%S")
        timestamp_filename = f"{task_id}_{ts}"
        return timestamp_filename
