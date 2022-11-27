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

# from ...bert4torch.layers import GlobalPointer

#  from .dataset import encode_text, encode_sentences
from ...bert4torch.layers import EfficientGlobalPointer as GlobalPointer
from ...bert4torch.utils import sequence_padding, Callback
from ...bert4torch.optimizers import get_linear_schedule_with_warmup
from ...bert4torch.losses import (
    MultilabelCategoricalCrossentropy,
    SparseMultilabelCategoricalCrossentropy,
)
from ...bert4torch.models import build_transformer_model, BaseModel

#  from dataset_a_b_x_y import masks_a, masks_b, masks_x, masks_y
#  from .dataset import max_length, categories_label2id, categories_id2label

device = "cuda" if torch.cuda.is_available() else "cpu"


def encode_text(
    text, tags, entities_label2id, relations_label2id, max_length, tokenizer
):
    def search(pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i : i + n] == pattern:
                return i
        return -1

    token_ids, segment_ids = tokenizer.encode(text, maxlen=max_length)

    tokens = tokenizer.tokenize(text, maxlen=max_length)
    assert len(tokens) == len(
        token_ids
    ), f"tokens: {len(tokens)}, {tokens}, token_ids: {len(token_ids)}, {token_ids}"
    mapping = tokenizer.rematch(text, tokens)
    start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
    end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
    # 整理三元组 {s: [(o, p)]}
    spoes = set()
    for tag in tags:
        s, p, o = tag["subject"]["mention"], tag["predicate"], tag["object"]["mention"]

        s = tokenizer.encode(s)[0][1:-1]
        p = relations_label2id[p]
        o = tokenizer.encode(o)[0][1:-1]

        sh, oh = tag["subject"]["start"], tag["object"]["start"]
        if sh in start_mapping and oh in start_mapping:
            sh0 = start_mapping[sh]
            oh0 = start_mapping[oh]
        else:
            sh0 = -1
            oh0 = -1

        sh = search(s, token_ids)
        oh = search(o, token_ids)

        if sh != -1 and oh != -1 and sh0 != -1 and oh0 != -1:
            if sh != sh0 or oh != oh0:
                if (
                    tokens[sh : sh + len(s)] != tokens[sh0 : sh0 + len(s)]
                    or tokens[oh : oh + len(o)] != tokens[oh0 : oh0 + len(o)]
                ):

                    print("-------------------")
                    #  print("tokens:", tokens)
                    print(
                        "subject:",
                        tag["subject"]["mention"],
                        "object:",
                        tag["object"]["mention"],
                    )
                    print(
                        "search:",
                        (sh, oh),
                        tokens[sh : sh + len(s)],
                        tokens[oh : oh + len(o)],
                    )
                    print(
                        "my:",
                        (sh0, oh0),
                        tokens[sh0 : sh0 + len(s)],
                        tokens[oh0 : oh0 + len(o)],
                    )
        sh, oh = sh0, oh0

        if sh != -1 and oh != -1:
            spoes.add((sh, sh + len(s) - 1, p, oh, oh + len(o) - 1))
    # 构建标签
    entity_labels = [set() for _ in range(2)]
    head_labels = [set() for _ in range(len(relations_label2id))]
    tail_labels = [set() for _ in range(len(relations_label2id))]
    for sh, st, p, oh, ot in spoes:
        entity_labels[0].add((sh, st))
        entity_labels[1].add((oh, ot))
        head_labels[p].add((sh, oh))
        tail_labels[p].add((st, ot))
    for label in entity_labels + head_labels + tail_labels:
        if not label:  # 至少要有一个标签
            label.add((0, 0))  # 如果没有则用0填充
    # [subject/object=2, 实体个数, 实体起终点]
    entity_labels = sequence_padding([list(l) for l in entity_labels])
    # [关系个数, 该关系下subject/object配对数, subject/object起点]
    head_labels = sequence_padding([list(l) for l in head_labels])
    # [关系个数, 该关系下subject/object配对数, subject/object终点]
    tail_labels = sequence_padding([list(l) for l in tail_labels])

    return (
        tokens,
        mapping,
        token_ids,
        segment_ids,
        entity_labels,
        head_labels,
        tail_labels,
    )


def encode_sentences(
    text_list, tags_list, entities_label2id, relations_label2id, max_length, tokenizer
):

    tokens_list, mappings_list = [], []
    token_ids_list, segment_ids_list = [], []
    entity_labels_list, head_labels_list, tail_labels_list = [], [], []
    for text, tags in zip(text_list, tags_list):

        (
            tokens,
            mapping,
            token_ids,
            segment_ids,
            entity_labels,
            head_labels,
            tail_labels,
        ) = encode_text(
            text, tags, entities_label2id, relations_label2id, max_length, tokenizer
        )

        tokens_list.append(tokens)
        mappings_list.append(mapping)
        token_ids_list.append(token_ids)
        segment_ids_list.append(segment_ids)
        entity_labels_list.append(entity_labels)
        head_labels_list.append(head_labels)
        tail_labels_list.append(tail_labels)

    return (
        tokens_list,
        mappings_list,
        token_ids_list,
        segment_ids_list,
        entity_labels_list,
        head_labels_list,
        tail_labels_list,
    )


class Model(BaseModel):

    # heads = len(predicate2id)
    def __init__(self, bert_model_path, heads, head_size=64) -> None:
        config_path = f"{bert_model_path}/bert_config.json"
        checkpoint_path = f"{bert_model_path}/pytorch_model.bin"
        dict_path = f"{bert_model_path}/vocab.txt"
        super().__init__()
        self.bert = build_transformer_model(config_path, checkpoint_path)
        self.entity_output = GlobalPointer(
            hidden_size=768, heads=2, head_size=head_size
        )
        self.head_output = GlobalPointer(
            hidden_size=768,
            heads=heads,
            head_size=head_size,
            RoPE=False,
            tril_mask=False,
        )
        self.tail_output = GlobalPointer(
            hidden_size=768,
            heads=heads,
            head_size=head_size,
            RoPE=False,
            tril_mask=False,
        )

    def forward(self, inputs):
        hidden_states = self.bert(inputs)  # [btz, seq_len, hdsz]
        mask = inputs[0].gt(0).long()

        # [btz, heads, seq_len, seq_len]
        entity_output = self.entity_output(hidden_states, mask)
        # [btz, heads, seq_len, seq_len]
        head_output = self.head_output(hidden_states, mask)
        # [btz, heads, seq_len, seq_len]
        tail_output = self.tail_output(hidden_states, mask)
        return entity_output, head_output, tail_output

    @classmethod
    def collate_fn(cls, batch):
        batch_token_ids, batch_segment_ids = [], []
        batch_entity_labels, batch_head_labels, batch_tail_labels = [], [], []
        for (
            _,
            _,
            _,
            token_ids,
            segment_ids,
            entity_labels,
            head_labels,
            tail_labels,
        ) in batch:
            idx, text, tags, others = b

            (
                tokens,
                mapping,
                token_ids,
                segment_ids,
                entity_labels,
                head_labels,
                tail_labels,
            ) = encode_text(
                text, tags, entities_label2id, relations_label2id, max_length, tokenizer
            )

            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_entity_labels.append(entity_labels)
            batch_head_labels.append(head_labels)
            batch_tail_labels.append(tail_labels)

        batch_token_ids = torch.tensor(
            sequence_padding(batch_token_ids), dtype=torch.long, device=device
        )
        batch_segment_ids = torch.tensor(
            sequence_padding(batch_segment_ids), dtype=torch.long, device=device
        )
        # batch_entity_labels: [btz, subject/object=2, 实体个数, 实体起终点]
        # batch_head_labels: [btz, 关系个数, 该关系下subject/object配对数, subject/object起点]
        # batch_tail_labels: [btz, 关系个数, 该关系下subject/object配对数, subject/object终点]
        batch_entity_labels = torch.tensor(
            sequence_padding(batch_entity_labels, seq_dims=2),
            dtype=torch.float,
            device=device,
        )
        batch_head_labels = torch.tensor(
            sequence_padding(batch_head_labels, seq_dims=2),
            dtype=torch.float,
            device=device,
        )
        batch_tail_labels = torch.tensor(
            sequence_padding(batch_tail_labels, seq_dims=2),
            dtype=torch.float,
            device=device,
        )

        return [batch_token_ids, batch_segment_ids], [
            batch_entity_labels,
            batch_head_labels,
            batch_tail_labels,
        ]


class MyLoss(SparseMultilabelCategoricalCrossentropy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_preds, y_trues):
        """y_preds: [Tensor], shape为[btz, heads, seq_len ,seq_len]"""
        loss_list = []
        for y_pred, y_true in zip(y_preds, y_trues):
            #  print("y_pred.shape", y_pred.shape, "y_true.shape", y_true.shape)
            shape = y_pred.shape
            # 乘以seq_len是因为(i, j)在展开到seq_len*seq_len维度对应的下标是i*seq_len+j
            y_true = (
                y_true[..., 0] * shape[2] + y_true[..., 1]
            )  # [btz, heads, 实体起终点的下标]
            # [btz, heads, seq_len*seq_len]
            y_pred = y_pred.reshape(shape[0], -1, np.prod(shape[2:]))
            loss = super().forward(y_pred, y_true.long())
            loss = torch.mean(torch.sum(loss, dim=1))
            loss_list.append(loss)
        return {
            "loss": sum(loss_list) / 3,
            "entity_loss": loss_list[0],
            "head_loss": loss_list[1],
            "tail_loss": loss_list[2],
        }


#  def collate_fn(batch):
#      batch_token_ids, batch_segment_ids = [], []
#      batch_entity_labels, batch_head_labels, batch_tail_labels = [], [], []
#      for (
#          _,
#          _,
#          _,
#          token_ids,
#          segment_ids,
#          entity_labels,
#          head_labels,
#          tail_labels,
#      ) in batch:
#          batch_token_ids.append(token_ids)
#          batch_segment_ids.append(segment_ids)
#          batch_entity_labels.append(entity_labels)
#          batch_head_labels.append(head_labels)
#          batch_tail_labels.append(tail_labels)
#
#      batch_token_ids = torch.tensor(
#          sequence_padding(batch_token_ids), dtype=torch.long, device=device
#      )
#      batch_segment_ids = torch.tensor(
#          sequence_padding(batch_segment_ids), dtype=torch.long, device=device
#      )
#      # batch_entity_labels: [btz, subject/object=2, 实体个数, 实体起终点]
#      # batch_head_labels: [btz, 关系个数, 该关系下subject/object配对数, subject/object起点]
#      # batch_tail_labels: [btz, 关系个数, 该关系下subject/object配对数, subject/object终点]
#      batch_entity_labels = torch.tensor(
#          sequence_padding(batch_entity_labels, seq_dims=2),
#          dtype=torch.float,
#          device=device,
#      )
#      batch_head_labels = torch.tensor(
#          sequence_padding(batch_head_labels, seq_dims=2),
#          dtype=torch.float,
#          device=device,
#      )
#      batch_tail_labels = torch.tensor(
#          sequence_padding(batch_tail_labels, seq_dims=2),
#          dtype=torch.float,
#          device=device,
#      )
#
#      return [batch_token_ids, batch_segment_ids], [
#          batch_entity_labels,
#          batch_head_labels,
#          batch_tail_labels,
#      ]


#  def extract_spoes(text, id2predicate, threshold=0):
#      """抽取输入text所包含的三元组"""
#      tokens = tokenizer.tokenize(text, maxlen=maxlen)
#      mapping = tokenizer.rematch(text, tokens)
#      token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)


def extract_spoes(
    model, text, tokens, mapping, token_ids, segment_ids, id2predicate, threshold=0
):

    start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
    end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}

    token_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    segment_ids = torch.tensor([segment_ids], dtype=torch.long, device=device)
    outputs = model.predict([token_ids, segment_ids])
    outputs = [o[0].cpu().numpy() for o in outputs]  # [heads, seq_len, seq_len]
    # 抽取subject和object
    subjects, objects = set(), set()
    outputs[0][:, [0, -1]] -= float("inf")
    outputs[0][:, :, [0, -1]] -= float("inf")
    for l, h, t in zip(*np.where(outputs[0] > threshold)):
        if l == 0:
            subjects.add((h, t))
        else:
            objects.add((h, t))
    # 识别对应的predicate
    #  spoes = set()
    spoes_set = set()
    spoes = []
    for sh, st in subjects:
        if sh not in start_mapping and st not in end_mapping:
            continue
        for oh, ot in objects:
            if oh not in start_mapping and ot not in end_mapping:
                continue
            p1s = np.where(outputs[1][:, sh, oh] > threshold)[0]
            p2s = np.where(outputs[2][:, st, ot] > threshold)[0]
            ps = set(p1s) & set(p2s)
            for p in ps:
                #  spoes.add(
                #      (
                #          text[mapping[sh][0] : mapping[st][-1] + 1],
                #          id2predicate[p],
                #          text[mapping[oh][0] : mapping[ot][-1] + 1],
                #          tokens[sh : st + 1],
                #          tokens[oh : ot + 1],
                #      )
                #  )

                s, p, o = (
                    text[mapping[sh][0] : mapping[st][-1] + 1],
                    id2predicate[p],
                    text[mapping[oh][0] : mapping[ot][-1] + 1],
                )
                if (s, p, o) not in spoes_set:
                    spoes_set.add((s, p, o))
                    spoes.append(
                        (
                            text[mapping[sh][0] : mapping[st][-1] + 1],
                            id2predicate[p],
                            text[mapping[oh][0] : mapping[ot][-1] + 1],
                            tokens[sh : st + 1],
                            tokens[oh : ot + 1],
                        )
                    )

    return list(spoes)


class SPO(tuple):
    """用来存三元组的类
    表现跟tuple基本一致，只是重写了 __hash__ 和 __eq__ 方法，
    使得在判断两个三元组是否等价时容错性更好。
    """

    def __init__(self, spo):
        self.spox = (
            tuple(spo[0]),
            spo[1],
            tuple(spo[2]),
        )

    def __hash__(self):
        return self.spox.__hash__()

    def __eq__(self, spo):
        return self.spox == spo.spox


#  class SPO(tuple):
#      """用来存三元组的类
#      表现跟tuple基本一致，只是重写了 __hash__ 和 __eq__ 方法，
#      使得在判断两个三元组是否等价时容错性更好。
#      """
#
#      def __init__(self, spo):
#          self.spox = (
#              tuple(tokenizer.tokenize(spo[0])),
#              spo[1],
#              tuple(tokenizer.tokenize(spo[2])),
#          )
#
#      def __hash__(self):
#          return self.spox.__hash__()
#
#      def __eq__(self, spo):
#          return self.spox == spo.spox
#


def evaluate(model, dataloader, entities_id2label, relations_id2label, threshold=0):
    """评估函数，计算f1、precision、recall"""
    X, Y, Z = 0, 1e-10, 1e-10
    #  f = open("dev_pred.json", "w", encoding="utf-8")
    pbar = tqdm()
    for (x_true, label), ds in zip(dataloader, dataloader.dataset):
        #  print("x_true:", x_true, "ds:", ds)
        #  print("label", label)
        text, tokens, mapping, token_ids, segment_ids = ds[:5]

        R = set(
            [
                SPO((spo[3], spo[1], spo[4]))
                for spo in extract_spoes(
                    model,
                    text,
                    tokens,
                    mapping,
                    token_ids,
                    segment_ids,
                    relations_id2label,
                )
            ]
        )
        T = set([SPO(spo) for spo in label])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        pbar.update()
        pbar.set_description(
            "f1: %.5f, precision: %.5f, recall: %.5f" % (f1, precision, recall)
        )
        #  s = json.dumps(
        #      {
        #          "text": text,
        #          "spo_list": list(T),
        #          "spo_list_pred": list(R),
        #          "new": list(R - T),
        #          "lack": list(T - R),
        #      },
        #      ensure_ascii=False,
        #      indent=2,
        #  )
        #  f.write(s + "\n")
    pbar.close()
    #  f.close()

    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z

    eval_result = {"all": (f1, precision, recall)}
    return eval_result


class Evaluator(Callback):
    """评估与保存"""

    def __init__(
        self,
        model,
        val_dataloader,
        entities_id2label,
        relations_id2label,
        best_f1=0.0,
        min_best=0.9,
        threshold=0,
    ):
        self.model = model
        self.val_dataloader = val_dataloader
        self.entities_id2label = entities_id2label
        self.relations_id2label = relations_id2label
        self.best_f1 = best_f1
        self.min_best = 0.9

    def on_epoch_end(self, steps, epoch, logs=None):
        eval_result = evaluate(
            self.model,
            self.val_dataloader,
            self.entities_id2label,
            self.relations_id2label,
            threshold=0,
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


def build_model(
    args,
    entity_labels,
    relation_labels,
    bert_model_path,
    learning_rate=0,
    num_training_steps=0,
):
    heads = len(relation_labels)
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


def predict_text(
    args, model, text, tokenizer, entity_labels, relation_labels, repeat=1, threshold=0
):

    entities_label2id = {label: i for i, label in enumerate(entity_labels)}
    entities_id2label = {i: label for i, label in enumerate(entity_labels)}
    relations_label2id = {label: i for i, label in enumerate(relation_labels)}
    relations_id2label = {i: label for i, label in enumerate(relation_labels)}

    #  true_tags = []
    #  token_ids, segment_ids, entity_labels, head_labels, tail_labels = encode_text(
    #      text, true_tags, entities_label2id, relations_label2id, args.max_length, tokenizer
    #  )
    #
    #  batch_token_ids = [token_ids]

    tags_list = extract_spoes(text, relations_id2label, threshold=0)

    #  scores_list = []
    #  for _ in range(repeat):
    #      scores = model.predict(batch_token_ids)
    #      scores_list.append(scores)
    #
    #  def average_scores(scores_list):
    #      scores_list = [s.unsqueeze(0) for s in scores_list]
    #      scores = torch.cat(scores_list).mean(dim=0)
    #      return scores
    #
    #  scores = average_scores(scores_list)
    #  sentences = [text]
    #
    #  def scores_to_mentions(scores):
    #      tags_list = []
    #      for i, (score, sent_text) in enumerate(zip(scores, sentences)):
    #          if len(sent_text) == 0:
    #              continue
    #          tokens = tokenizer.tokenize(sent_text, maxlen=args.max_length)
    #          mapping = tokenizer.rematch(sent_text, tokens)
    #          # print(f"sent_text: len: {len(sent_text)}")
    #          # print(f"tokens: {len(tokens)}, {tokens}")
    #          # print(f"mapping: {len(mapping)}, {mapping}")
    #
    #          # 用集合自动消除完全相同的实体标注
    #          R = {}
    #          for r, l, start, end in zip(*np.where(score.cpu() > threshold)):
    #
    #              if r in relations_id2label and l in entities_id2label and start >= 1 and start < len(
    #                  mapping
    #              ) - 1 and end >= 1 and end < len(mapping) - 1 and start < end:
    #                  # print(f"l: {l}, start: {start}, end: {end}")
    #                  # print(f"mapping[start][0]: {mapping[start][0]}, mapping[end][-1] + 1: {mapping[end][-1] + 1}")
    #                  span_s = mapping[start][0]
    #                  span_e = mapping[end][-1]
    #                  k2 = sent_text[span_s:span_e + 1]
    #                  #  k2 = sent_text[mapping[start][0]:mapping[end][-1] + 1]
    #                  # print(start, end, entities_id2label[l], relations_id2label[r], k2)
    #
    #                  cat_label = entities_id2label[l]
    #                  relation_label = relations_id2label[r]
    #
    #                  if relation_label not in R:
    #                      R[relation_label] = set()
    #
    #                  R[relation_label].add((span_s, span_e, cat_label, k2, sent_text))
    #
    #          text_tags = []
    #          for relation_label, e_list in R.items():
    #              entity_tags = [{
    #                  'category': cat_label,
    #                  'start': start,
    #                  'mention': k2
    #              } for start, end, cat_label, k2, sent_text in e_list]
    #              entity_tags = sorted(entity_tags, key=lambda x: x['start'])
    #              text_tags.append({
    #                  'relation': relation_label,
    #                  'entities': entity_tags
    #              })
    #
    #          tags_list.append({
    #              'text': sent_text,
    #              'tags': text_tags
    #          })
    #
    #      return tags_list

    #  tags_list = scores_to_mentions(scores)

    return tags_list
