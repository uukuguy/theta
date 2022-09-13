#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn

from tensorboardX import SummaryWriter
tb_writer = SummaryWriter(log_dir='./tensorboard_logs')  # prepare summary writer

script_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

from ..bert4torch.models import build_transformer_model, BaseModel
from ..bert4torch.losses import MultilabelCategoricalCrossentropy
from ..bert4torch.optimizers import get_linear_schedule_with_warmup
from ..bert4torch.utils import sequence_padding, Callback, get_pool_emb

# from ...bert4torch.layers import GlobalPointer
from ..bert4torch.layers import EfficientGlobalPointer as GlobalPointer

from .tagging import TaskLabels, TaskTag, TaggedData
from .dataset import encode_text, encode_sentences

device = "cuda" if torch.cuda.is_available() else "cpu"


class Model(BaseModel):
    def __init__(self, bert_model_path, num_classes, pool_method='cls') -> None:
        config_path = f"{bert_model_path}/bert_config.json"
        checkpoint_path = f"{bert_model_path}/pytorch_model.bin"
        dict_path = f"{bert_model_path}/vocab.txt"
        super().__init__()

        self.pool_method = pool_method
        self.num_classes = num_classes

        self.bert = build_transformer_model(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            # segment_vocab_size=0,
            with_pool=True,
        )

        self.dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(self.bert.configs['hidden_size'], num_classes)


    def forward(self, token_ids, segment_ids):
        hidden_states, pooling = self.bert([token_ids, segment_ids])
        pooled_output = get_pool_emb(hidden_states, pooling, token_ids.gt(0).long(), self.pool_method)
        output = self.dropout(pooled_output)
        output = self.dense(output)
        return output


    @classmethod
    def collate_fn(batch):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for _, (token_ids, segment_ids), label in batch:
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
        batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long, device=device)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
        return [batch_token_ids, batch_segment_ids], batch_labels.flatten()



class MyLoss(nn.CrossEntropyLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_pred, y_true):
        # # [btz*heads, seq_len*seq_len]
        # y_true = y_true.view(y_true.shape[0] * y_true.shape[1], -1)
        # # [btz*heads, seq_len*seq_len]
        # y_pred = y_pred.view(y_pred.shape[0] * y_pred.shape[1], -1)
        return super().forward(y_pred, y_true)

def evaluate(model, dataloader, task_labels, threshold=0):
    total, right = 0., 0.
    pbar = tqdm(desc="eval", ncols=100)
    for x_true, y_true in dataloader:
        y_pred = model.predict(x_true).argmax(axis=1)
        total += len(y_true)
        right += (y_true == y_pred).sum().item()

        val_acc = right / total if total else 0.0

        pbar.update()
        pbar.set_postfix({'val_acc': f"{val_acc:.5f}"})

    eval_result = {"all": val_acc}
    return eval_result


class Evaluator(Callback):
    """评估与保存"""

    def __init__(
        self,
        model,
        val_dataloader,
        task_labels,
        best_acc=0.0,
        min_best=0.9,
        threshold=0,
    ):
        self.model = model
        self.val_dataloader = val_dataloader
        self.task_labels = task_labels
        self.best_acc = best_acc
        self.min_best = min_best
        self.threshold = threshold

    def do_evaluate(self):
        return evaluate(self.model, self.valid_dataloader, self.task_labels, threshold=self.threshold)

    def on_batch_end(self, global_step, batch, logs=None):
        if global_step % 10 == 0:
            tb_writer.add_scalar(f"train/loss", logs['loss'], global_step)
            eval_result = self.do_evaluate()
            val_acc = eval_result["all"]
            tb_writer.add_scalar(f"val/acc", val_acc, global_step)

    def on_epoch_end(self, steps, epoch, logs=None):
        eval_result = self.do_evaluate()

        acc = eval_result["all"]
        if acc > self.best_acc:
            self.best_acc = acc
            self.model.save_weights("best_model.pt")
            if acc > self.min_best:
                self.model.save_weights(f"best_model_{self.best_acc:.5f}.pt")
        print(
            f"[val] acc: {acc:.5f}, best_acc: {self.best_acc:.5f}"
        )
        for k, v in eval_result.items():
            if k == "total":
                continue
            v_list = [f"{x:.5f}" for x in v]
            print(f'"{k}": {v_list} ')

        logs.update({"acc": acc})
        logs.update({"best_acc": self.best_acc})


def build_model(args, num_training_steps=0):
    bert_model_path = args.bert_model_path
    learning_rate = args.learning_rate
    num_classes = len(args.task_labels.labels)

    model = Model(bert_model_path, num_classes=num_classes).to(device)

    if learning_rate > 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = None

    if num_training_steps > 0:
        num_warmup_steps = 0  # int(num_training_steps * 0.05)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    else:
        scheduler = None

    model.compile(loss=nn.CrossEntropyLoss, optimizer=optimizer, scheduler=scheduler, metrics=['accuracy'])

    return model


def predict_text(args, model, text, tokenizer, threshold=0):
    repeat = args.repeat
    id2label = args.task_labels.id2label

    true_tags = []
    ((token_ids, segment_ids), label) = encode_text(
        text, true_tags, args.task_labels, args.max_length, tokenizer
    )

    batch_token_ids, batch_segment_ids = [token_ids], [segment_ids]
    batch_token_ids = torch.tensor(
        sequence_padding(batch_token_ids), dtype=torch.long, device=device
    )
    batch_segment_ids = torch.tensor(
        sequence_padding(batch_segment_ids), dtype=torch.long, device=device
    )

    logits_list = []
    for _ in range(repeat):
        logit = model.predict([batch_token_ids, batch_segment_ids])
        logits_list.append(logit)

    def average_logits(logits_list):
        logits_list = [s.unsqueeze(0) for s in logits_list]
        logit = torch.cat(logits_list).mean(dim=0)
        return logit

    logit = average_logits(logits_list)

    y_pred = torch.argmax(torch.softmax(logit, dim=-1)).cpu().numpy()
    label = id2label[y_pred]

    return label
