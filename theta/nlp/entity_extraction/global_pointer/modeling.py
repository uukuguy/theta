#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json
import numpy as np
from tqdm import tqdm
import torch

#  from sched import scheduler
#  from tensorboardX import SummaryWriter
#  writer = SummaryWriter(log_dir='./tensorboard_log')  # prepare summary writer

try:
    import rich

    def print(*arg, **kwargs):
        rich.print(*arg, **kwargs)


except:
    pass

script_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

from ...bert4torch.models import build_transformer_model, BaseModel
from ...bert4torch.losses import MultilabelCategoricalCrossentropy
from ...bert4torch.optimizers import get_linear_schedule_with_warmup
from ...bert4torch.utils import sequence_padding, Callback

# from ...bert4torch.layers import GlobalPointer
from ...bert4torch.layers import EfficientGlobalPointer as GlobalPointer

from ..tagging import TaskLabels, TaskTag, TaggedData
from .dataset import encode_text, encode_sentences

device = "cuda" if torch.cuda.is_available() else "cpu"


class Model(BaseModel):
    def __init__(self, bert_model_path, heads, head_size) -> None:
        config_path = f"{bert_model_path}/bert_config.json"
        checkpoint_path = f"{bert_model_path}/pytorch_model.bin"
        dict_path = f"{bert_model_path}/vocab.txt"
        super().__init__()

        self.bert = build_transformer_model(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            segment_vocab_size=0,
        )
        self.global_pointer = GlobalPointer(
            hidden_size=768, heads=heads, head_size=head_size
        )

    def forward(self, token_ids):
        sequence_output = self.bert([token_ids])  # [btz, seq_len, hdsz]
        logit = self.global_pointer(sequence_output, token_ids.gt(0).long())

        return logit

    @classmethod
    def collate_fn(cls, batch):
        batch_token_ids, batch_labels = [], []
        for _, _, token_ids, labels in batch:
            batch_token_ids.append(token_ids)
            batch_labels.append(labels)

        batch_token_ids = torch.tensor(
            sequence_padding(batch_token_ids), dtype=torch.long, device=device
        )
        batch_labels = torch.tensor(
            sequence_padding(batch_labels, seq_dims=3), dtype=torch.long, device=device
        )
        return batch_token_ids, batch_labels


class MyLoss(MultilabelCategoricalCrossentropy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_pred, y_true):
        # [btz*heads, seq_len*seq_len]
        y_true = y_true.view(y_true.shape[0] * y_true.shape[1], -1)
        # [btz*heads, seq_len*seq_len]
        y_pred = y_pred.view(y_pred.shape[0] * y_pred.shape[1], -1)

        return super().forward(y_pred, y_true)


def evaluate(model, dataloader, task_labels, threshold=0):
    entities_id2label = task_labels.entities_id2label

    X, Y, Z = 0, 1e-10, 1e-10
    #  for (x_true, label), ds in zip(dataloader, dataloader.dataset):
    pbar = tqdm(desc="eval", ncols=100)
    #  for (encoded_text, encoded_label), ds in zip(dataloader, dataloader.dataset):
    for ds in dataloader.dataset:
        #  scores = model.predict(encoded_text)
        tagged_data, (tokens, mapping), token_ids, encoded_label = ds
        text, true_tags = tagged_data.text, tagged_data.tags

        encoded_label = [encoded_label]
        batch_token_ids = [token_ids]
        batch_token_ids = torch.tensor(
            sequence_padding(batch_token_ids), dtype=torch.long, device=device
        )
        scores = model.predict(batch_token_ids)
        scores = [o.cpu().numpy() for o in scores]  # [heads, seq_len, seq_len]

        for i, score in enumerate(scores):
            R = set()
            for l, start, end in zip(*np.where(score > threshold)):
                #  R.add((start, end, entities_id2label[l]))
                c = entities_id2label[l]
                s = mapping[start][0]
                e = mapping[end][-1] + 1
                m = text[s:e]
                R.add((c, s, m))
                #  R.add(TaskTag(c=c, s=s, m=m))

            T = set()
            #  for l, start, end in zip(*np.where(encoded_label[i] > threshold)):
            for l, start, end in zip(*np.where(encoded_label[i] > threshold)):
                #  T.add((start, end, entities_id2label[l]))
                c = entities_id2label[l]
                s = mapping[start][0]
                e = mapping[end][-1] + 1
                m = text[s:e]
                T.add((c, s, m))
                #  T.add(TaskTag(c=c, s=s, m=m))

            if R != T:
                print("========================================")
                print(text)
                print("----------------------------------------")
                print(
                    f"T: {sorted([TaskTag(t[0], t[1], t[2]) for t in T], key=lambda x: x.s)}"
                )
                print(
                    f"R: {sorted([TaskTag(r[0], r[1], r[2]) for r in R], key=lambda x: x.s)}"
                )

            X += len(R & T)
            Y += len(R)
            Z += len(T)

        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        pbar.update()
        pbar.set_description(
            "f1: %.5f, precision: %.5f, recall: %.5f" % (f1, precision, recall)
        )

    eval_result = {"all": (f1, precision, recall)}
    return eval_result


class Evaluator(Callback):
    """评估与保存"""

    def __init__(
        self,
        model,
        val_dataloader,
        task_labels,
        best_f1=0.0,
        min_best=0.9,
        threshold=0,
    ):
        self.model = model
        self.val_dataloader = val_dataloader
        self.task_labels = task_labels
        self.best_f1 = best_f1
        self.min_best = min_best
        self.threshold = threshold

    def on_epoch_end(self, steps, epoch, logs=None):
        eval_result = evaluate(
            self.model, self.val_dataloader, self.task_labels, threshold=self.threshold
        )
        f1, precision, recall = eval_result["all"]
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.model.save_weights("best_model.pt")
            if f1 > self.min_best:
                self.model.save_weights(f"best_model_{self.best_f1:.5f}.pt")
        print(
            f"[val] f1: {f1:.5f}, p: {precision:.5f} r: {recall:.5f} best_f1: {self.best_f1:.5f}"
        )
        # print(f"[val] {json.dumps(eval_result, ensure_ascii=False)}")
        for k, v in eval_result.items():
            if k == "total":
                continue
            v_list = [f"{x:.5f}" for x in v]
            print(f'"{k}": {v_list} ')

        logs.update({"f1": f1})
        logs.update({"best_f1": self.best_f1})


def build_model(args, num_training_steps=0):
    bert_model_path = args.bert_model_path
    learning_rate = args.learning_rate
    #  num_training_steps = args.num_training_steps
    entity_labels = args.task_labels.entity_labels

    heads = len(entity_labels)
    head_size = 64
    model = Model(bert_model_path, heads=heads, head_size=head_size).to(device)

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

    model.compile(loss=MyLoss(), optimizer=optimizer, scheduler=scheduler)

    return model


def predict_text(args, model, text, tokenizer, threshold=0):
    repeat = args.repeat
    entities_id2label = args.task_labels.entities_id2label

    true_tags = []
    ((tokens, mapping), token_ids, labels) = encode_text(
        text, true_tags, args.task_labels, args.max_length, tokenizer
    )

    batch_token_ids = [token_ids]
    batch_token_ids = torch.tensor(
        sequence_padding(batch_token_ids), dtype=torch.long, device=device
    )

    scores_list = []
    for _ in range(repeat):
        scores = model.predict(batch_token_ids)
        scores_list.append(scores)

    def average_scores(scores_list):
        scores_list = [s.unsqueeze(0) for s in scores_list]
        scores = torch.cat(scores_list).mean(dim=0)
        return scores

    scores = average_scores(scores_list)
    sentences = [text]

    def scores_to_mentions(scores):
        tags_list = []
        for i, (score, sent_text) in enumerate(zip(scores, sentences)):
            if len(sent_text) == 0:
                continue
            tokens = tokenizer.tokenize(sent_text, maxlen=args.max_length)
            mapping = tokenizer.rematch(sent_text, tokens)
            # print(f"sent_text: len: {len(sent_text)}")
            # print(f"tokens: {len(tokens)}, {tokens}")
            # print(f"mapping: {len(mapping)}, {mapping}")

            # 用集合自动消除完全相同的实体标注
            R = set()
            for l, start, end in zip(*np.where(score.cpu() > threshold)):

                if (
                    l in entities_id2label
                    and start >= 1
                    and start < len(mapping) - 1
                    and end >= 1
                    and end < len(mapping) - 1
                    and start < end
                ):
                    # print(f"l: {l}, start: {start}, end: {end}")
                    # print(f"mapping[start][0]: {mapping[start][0]}, mapping[end][-1] + 1: {mapping[end][-1] + 1}")
                    span_s = mapping[start][0]
                    span_e = mapping[end][-1]
                    k2 = sent_text[span_s : span_e + 1]
                    #  k2 = sent_text[mapping[start][0]:mapping[end][-1] + 1]
                    # print(start, end, entities_id2label[l], k2)

                    cat_label = entities_id2label[l]

                    R.add((span_s, span_e, cat_label, k2, sent_text))

            sent_tags = [
                #  {"category": cat_label, "start": start, "mention": k2}
                TaskTag(c=cat_label, s=start, m=k2)
                for start, end, cat_label, k2, sent_text in R
            ]
            sent_tags = sorted(sent_tags, key=lambda x: x.s)

            tags_list.append(sent_tags)

        return tags_list

    tags_list = scores_to_mentions(scores)
    tags = tags_list[0]

    #  from .utils import merge_sent_tags_list
    #
    #  full_tags = merge_sent_tags_list(tags_list)

    #  return full_tags, tags_list

    return tags
