#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json
import numpy as np
from tqdm import tqdm
import torch

#  from sched import scheduler
from tensorboardX import SummaryWriter
tb_writer = SummaryWriter(log_dir='./tensorboard_logs')  


try:
    import rich

    def print(*arg, **kwargs):
        rich.print(*arg, **kwargs)


except:
    pass

script_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))


# from ...bert4torch.layers import GlobalPointer
from ...bert4torch.layers import EfficientGlobalPointer as GlobalPointer
from ...bert4torch.utils import sequence_padding, Callback
from ...bert4torch.optimizers import get_linear_schedule_with_warmup
from ...bert4torch.losses import (
    # MultilabelCategoricalCrossentropy,
    SparseMultilabelCategoricalCrossentropy,
)
from ...bert4torch.models import build_transformer_model, BaseModel, BaseModelDDP

from ..tagging import TaskLabels, TaskTag, SubjectTag, ObjectTag, TaggedData
from .dataset import encode_text, encode_sentences

def get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=1, last_epoch= -1):
    import math
    from torch.optim.lr_scheduler import LambdaLR
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

device = "cuda" if torch.cuda.is_available() else "cpu"


class Model(BaseModel):
    def __init__(self, bert_model_path, heads, head_size=64) -> None:
        config_path = f"{bert_model_path}/bert_config.json"
        checkpoint_path = f"{bert_model_path}/pytorch_model.bin"
        dict_path = f"{bert_model_path}/vocab.txt"

        print("bert_model_path:", bert_model_path)
        print("config_path:", config_path)
        print("checkpoint_path:", checkpoint_path)
        print("dict_path:", dict_path)
        super().__init__()

        self.bert = build_transformer_model(config_path, checkpoint_path)
        self.entity_output = GlobalPointer(
            hidden_size=768, heads=2, head_size=head_size
        )
        self.head_output = GlobalPointer(
            hidden_size=768,
            heads=heads,
            head_size=head_size,
            RoPE=True,
            tril_mask=False,
        )
        self.tail_output = GlobalPointer(
            hidden_size=768,
            heads=heads,
            head_size=head_size,
            RoPE=True,
            tril_mask=False,
        )

        # FIXME
        self.head_tail_output = GlobalPointer(
            hidden_size=768,
            heads=heads,
            head_size=head_size,
            RoPE=True,
            tril_mask=False,
        )
        self.tail_head_output = GlobalPointer(
            hidden_size=768,
            heads=heads,
            head_size=head_size,
            RoPE=True,
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

        # FIXME
        # [btz, heads, seq_len, seq_len]
        head_tail_output = self.head_tail_output(hidden_states, mask)
        # [btz, heads, seq_len, seq_len]
        tail_head_output = self.tail_head_output(hidden_states, mask)

        return entity_output, head_output, tail_output, head_tail_output, tail_head_output

    @staticmethod
    def collate_fn(batch):
        batch_token_ids, batch_segment_ids = [], []
        batch_entity_labels, batch_head_labels, batch_tail_labels = [], [], []
        # FIXME
        batch_head_tail_labels, batch_tail_head_labels = [], []

        for (
            _,
            _,
            (token_ids, segment_ids),
            (entity_labels, head_labels, tail_labels, head_tail_labels, tail_head_labels),
        ) in batch:
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)

            batch_entity_labels.append(entity_labels)
            batch_head_labels.append(head_labels)
            batch_tail_labels.append(tail_labels)
            # FIXME
            batch_head_tail_labels.append(head_tail_labels)
            batch_tail_head_labels.append(tail_head_labels)


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

        #FIXME
        batch_head_tail_labels = torch.tensor(
            sequence_padding(batch_head_tail_labels, seq_dims=2),
            dtype=torch.float,
            device=device,
        )
        batch_tail_head_labels = torch.tensor(
            sequence_padding(batch_tail_head_labels, seq_dims=2),
            dtype=torch.float,
            device=device,
        )

        #  print("batch_entity_labels.shape:", batch_entity_labels.shape)
        #  print("batch_head_labels.shape:", batch_head_labels.shape)
        #  print("batch_tail_labels.shape:", batch_tail_labels.shape)
        return [batch_token_ids, batch_segment_ids], [
            batch_entity_labels,
            batch_head_labels,
            batch_tail_labels,
            # FIXME
            batch_head_tail_labels,
            batch_tail_head_labels,
        ]

    @staticmethod
    def build_model(args, num_training_steps=0):

        bert_model_path = args.bert_model_path
        learning_rate = args.learning_rate
        #  num_training_steps = args.num_training_steps
        relation_labels = args.task_labels.relation_labels

        heads = len(relation_labels)
        head_size = 64
        model = Model(bert_model_path, heads=heads, head_size=head_size).to(device)
        # 指定DDP模型使用多GPU，master_rank为指定用于打印训练过程的local_rank
        if args.local_rank >= 0:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = BaseModelDDP(
                model,
                master_rank=0,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=False,
            )

        if learning_rate > 0:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        else:
            optimizer = None

        if num_training_steps > 0:
            num_warmup_steps = args.num_warmup_steps  # int(num_training_steps * 0.05)
            # scheduler = get_linear_schedule_with_warmup(
            #     optimizer,
            #     num_warmup_steps=num_warmup_steps,
            #     num_training_steps=num_training_steps,
            # )
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=args.num_cycles,
            )
        else:
            scheduler = None

        # model.compile(loss=MyLoss(), optimizer=optimizer, scheduler=scheduler, adversarial_train={'name': 'fgm'})
        model.compile(loss=MyLoss(neg_weight=0.6), optimizer=optimizer, scheduler=scheduler)

        return model


    @staticmethod
    def predict_text(args, model, text, tokenizer, repeat=1, threshold=0):

        relations_map = args.task_labels.relations_map

        #  tokens = tokenizer.tokenize(text, maxlen=args.max_length)
        #  mapping = tokenizer.rematch(text, tokens)
        #
        #  token_ids, segment_ids = tokenizer.encode(text, maxlen=args.max_length)
        #  token_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        #  segment_ids = torch.tensor([segment_ids], dtype=torch.long, device=device)

        tags = []
        (
            (tokens, mapping),
            (token_ids, segment_ids),
            (entity_labels, head_labels, tail_labels, 
            # FIXME
            head_tail_labels, tail_head_labels,
            ),
        ) = encode_text(text, tags, args.task_labels, args.max_length, tokenizer)

        token_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        segment_ids = torch.tensor([segment_ids], dtype=torch.long, device=device)

        # outputs = model.predict([token_ids, segment_ids])

        entity_logits_list = []
        head_logits_list = []
        tail_logits_list = []
        # FIXME
        head_tail_logits_list = []
        tail_head_logits_list = []

        for _ in range(repeat):
            # scores = model.predict(batch_token_ids)
            entity_logit, head_logit, tail_logit, head_tail_logit, tail_head_logit = model.predict([token_ids, segment_ids])
            entity_logits_list.append(entity_logit)
            head_logits_list.append(head_logit)
            tail_logits_list.append(tail_logit)
            # FIXME
            head_tail_logits_list.append(head_tail_logit)
            tail_head_logits_list.append(tail_head_logit)

        # print("logits_list:", logits_list)

        def average_logits_list(logits_list):
            logits_list = [s.unsqueeze(0) for s in logits_list]
            logits_list = torch.cat(logits_list).mean(dim=0)
            return logits_list

        entity_logits_list = average_logits_list(entity_logits_list)
        head_logits_list = average_logits_list(head_logits_list)
        tail_logits_list = average_logits_list(tail_logits_list)
        # FIXME
        head_tail_logits_list = average_logits_list(head_tail_logits_list)
        tail_head_logits_list = average_logits_list(tail_head_logits_list)

        outputs = (entity_logits_list, head_logits_list, tail_logits_list, head_tail_logits_list, tail_head_logits_list)
        outputs = [o[0].cpu().numpy() for o in outputs]  # [heads, seq_len, seq_len]

        relations_id2label = args.task_labels.relations_id2label
        spoes, _ = extract_spoes(
            text, mapping, outputs, relations_id2label, threshold=threshold
        )

        #  spoes, _ = extract_spoes(
        #      model, text, tokens, mapping, token_ids, segment_ids, relations_id2label
        #  )

        tags = []
        for spo in spoes:
            s_m, rel, o_m, s_s, o_s = spo
            s_c, o_c = relations_map[rel]
            #  tag = {
            #      "subject": {"category": s_c, "start": s_s, "mention": s_m},
            #      "predicate": rel,
            #      "object": {"category": o_c, "start": o_s, "mention": o_m},
            #  }
            tag = TaskTag(
                s=SubjectTag(c=s_c, s=s_s, m=s_m),
                p=rel,
                o=ObjectTag(c=o_c, s=o_s, m=o_m),
            )
            tags.append(tag)
        #  tags = sorted(tags, key=lambda x: x["subject"]["start"])
        tags = sorted(tags, key=lambda x: x.s.s)

        return tags


class MultilabelCategoricalCrossentropy(torch.nn.Module):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1， 1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred不用加激活函数，尤其是不能加sigmoid或者softmax！预测
         阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解本文。
    参考：https://kexue.fm/archives/7359
    """
    def __init__(self, neg_weight, **kwargs):
        self.neg_weight = neg_weight
        super().__init__(**kwargs)
    def forward(self, y_pred, y_true):
        """ y_true ([Tensor]): [..., num_classes]
            y_pred ([Tensor]): [..., num_classes]
        """
        y_pred = (1-2*y_true) * y_pred
        y_pred_pos = y_pred - (1-y_true) * 1e12
        y_pred_neg = y_pred - y_true * 1e12

        y_pred_pos = torch.cat([y_pred_pos, torch.zeros_like(y_pred_pos[..., :1])], dim=-1)
        y_pred_neg = torch.cat([y_pred_neg, torch.zeros_like(y_pred_neg[..., :1])], dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        return (pos_loss + neg_loss).mean()
        # return ((pos_loss * (1-self.neg_weight) + neg_loss * self.neg_weight) * 2).mean()

class MyLoss(MultilabelCategoricalCrossentropy):
    def __init__(self, neg_weight, **kwargs):
        super().__init__(neg_weight=neg_weight, **kwargs)

    def forward(self, y_preds, y_trues):
        """y_preds: [Tensor], shape为[btz, heads, seq_len ,seq_len]"""
        loss_list = []
        for y_pred, y_true in zip(y_preds, y_trues):
            shape = y_true.shape

            new_true = np.zeros((shape[0], shape[1], y_pred.shape[2], y_pred.shape[2]))
            #  print("new_true:", new_true.shape)
            y_true = y_true.cpu().numpy().astype(int)
            #  print("y_true:", y_true.shape, y_true)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        x = y_true[i, j, k][0]
                        y = y_true[i, j, k][1]
                        #  print("x", x, "y", y)
                        if x > 0 and y > 0:
                            new_true[i, j, x, y] = 1.0
            y_true = torch.tensor(new_true.astype(float)).to(device)

            y_true = y_true.view(
                y_true.shape[0] * y_true.shape[1], -1
            )  # [btz*heads, seq_len*seq_len]
            y_pred = y_pred.view(
                y_pred.shape[0] * y_pred.shape[1], -1
            )  # [btz*heads, seq_len*seq_len]

            loss = super().forward(y_pred, y_true.long())
            #  print("loss.shape:", loss.shape, "loss", loss)
            #  loss = torch.mean(torch.sum(loss, dim=1))
            loss_list.append(loss)

        # assert len(loss_list) == 3
        # entity_loss, head_loss, tail_loss = loss_list
        # # loss = sum(loss_list) / 3
        # # loss = (entity_loss * 0.5 + head_loss * 0.25 + tail_loss * 0.25)
        # loss = (entity_loss * 0.6 + head_loss * 0.2 + tail_loss * 0.2)

        assert len(loss_list) == 5
        entity_loss, head_loss, tail_loss, head_tail_loss, tail_head_loss = loss_list
        loss = sum(loss_list) / 5
        # loss = (entity_loss * 0.5 + head_loss * 0.25 + tail_loss * 0.25)
        # loss = (entity_loss * 0.6 + head_loss * 0.2 + tail_loss * 0.2)

        return {
            "loss": loss,
            "entity_loss": entity_loss,
            "head_loss": head_loss,
            "tail_loss": tail_loss,
            "head_tail_loss": head_tail_loss,
            "tail_head_loss": tail_head_loss,
        }
        # return {
        #     "loss": sum(loss_list) / 3,
        #     "entity_loss": loss_list[0],
        #     "head_loss": loss_list[1],
        #     "tail_loss": loss_list[2],
        # }


class MyLoss_Sparse(SparseMultilabelCategoricalCrossentropy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_preds, y_trues):
        """y_preds: [Tensor], shape为[btz, heads, seq_len ,seq_len]"""
        loss_list = []
        for y_pred, y_true in zip(y_preds, y_trues):
            #  print(f"-------------------")
            #  print(
            #      "---> origin:",
            #      "y_pred.shape",
            #      y_pred.shape,
            #      "y_true.shape",
            #      y_true.shape,
            #  )
            shape = y_pred.shape
            # 乘以seq_len是因为(i, j)在展开到seq_len*seq_len维度对应的下标是i*seq_len+j
            # [btz, heads, 实体起终点的下标]
            y_true = y_true[..., 0] * shape[2] + y_true[..., 1]
            # [btz, heads, seq_len*seq_len]
            y_pred = y_pred.reshape(shape[0], -1, np.prod(shape[2:]))
            #  print("y_pred", y_pred, "y_true", y_true)
            #  print("y_pred.shape", y_pred.shape, "y_true.shape", y_true.shape)
            loss = super().forward(y_pred, y_true.long())
            loss = torch.mean(torch.sum(loss, dim=1))
            loss_list.append(loss)
        return {
            "loss": sum(loss_list) / 3,
            "entity_loss": loss_list[0],
            "head_loss": loss_list[1],
            "tail_loss": loss_list[2],
        }


def extract_spoes(text, mapping, outputs, relations_id2label, threshold=0):

    # 抽取subject和object
    subjects, objects = set(), set()
    outputs[0][:, [0, -1]] -= float("inf")
    outputs[0][:, :, [0, -1]] -= float("inf")
    # for l, h, t in zip(*np.where(outputs[0] > threshold / 2 if threshold < 0 else threshold * 2)):
    for l, h, t in zip(*np.where(outputs[0] > threshold)):
        if l == 0:
            subjects.add((h, t))
        else:
            objects.add((h, t))
    subjects_list = [text[mapping[h][0] : mapping[t][-1] + 1] for h, t in subjects]
    objects_list = [text[mapping[h][0] : mapping[t][-1] + 1] for h, t in objects]
    #  print("")
    #  print(f"found_subjects: {subjects_list}")
    #  print(f"found_objects: {objects_list}")

    # 识别对应的predicate
    #  spoes = set()
    spoes_set = set()
    spoes = []
    #  found_objects = set()
    for sh, st in subjects:
        s_m = text[mapping[sh][0] : mapping[st][-1] + 1]
        for j, (oh, ot) in enumerate(objects):
            o_m = text[mapping[oh][0] : mapping[ot][-1] + 1]
            #  if j in found_objects:
            #      continue
            p1s = np.where(outputs[1][:, sh, oh] > threshold)[0]
            p2s = np.where(outputs[2][:, st, ot] > threshold)[0]
            # FIXME
            p3s = np.where(outputs[3][:, sh, ot] > threshold)[0]
            p4s = np.where(outputs[4][:, oh, st] > threshold)[0]

            #  if len(p1s) > 0 or len(p2s) > 0:
            #      print(
            #          f"s_m: {s_m}, o_m: {o_m}, head_labels: {outputs[1][:, sh, oh]}, tail_labels: {outputs[2][:, st, ot]}"
            #      )

            # ps = set(p1s) & set(p2s)
            ps = set(p1s) & set(p2s) & set(p3s) & set(p4s)

            for p in ps:
                s, p, o = (
                    text[mapping[sh][0] : mapping[st][-1] + 1],
                    relations_id2label[p],
                    text[mapping[oh][0] : mapping[ot][-1] + 1],
                )
                if (s, p, o) not in spoes_set:
                    spoes_set.add((s, p, o))
                    spoes.append(
                        #  (s, p, o, token_ids[sh : st + 1], token_ids[oh : ot + 1])
                        (s, p, o, mapping[sh][0], mapping[oh][0])
                    )
                    #  print(f"found_spo: ({s}, {p}, {o})")
                    #  found_objects.add(j)

    return list(spoes), (subjects_list, objects_list)


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
        debug=False,
    ):
        self.model = model
        self.val_dataloader = val_dataloader
        self.task_labels = task_labels
        self.best_f1 = best_f1
        self.min_best = min_best
        self.threshold = threshold
        self.debug = debug
    
    def do_evaluate(self):
        eval_result = Evaluator.evaluate(
            self.model, self.val_dataloader, self.task_labels, threshold=self.threshold, debug=self.debug
        )
        return eval_result

    def on_batch_end(self, global_step, batch, logs=None):
        if global_step % 10 == 0:
            tb_writer.add_scalar(f"train/loss", logs['loss'], global_step)

            tb_writer.add_scalar(f"val/best_f1", logs.get('best_f1', 0.0), global_step)
            tb_writer.add_scalar(f"val/f1", logs.get('f1', 0.0), global_step)
            tb_writer.add_scalar(f"val/precision", logs.get('precision', 0.0), global_step)
            tb_writer.add_scalar(f"val/recall", logs.get('recall', 0.0), global_step)
            # eval_result = self.do_evaluate()
            # val_f1, val_p, val_r = eval_result["all"]
            # tb_writer.add_scalar(f"val/f1", val_f1, global_step)
            # tb_writer.add_scalar(f"val/precision", val_p, global_step)
            # tb_writer.add_scalar(f"val/recall", val_r, global_step)


    def on_epoch_end(self, steps, epoch, logs=None):
        eval_result = self.do_evaluate()
        f1, precision, recall = eval_result["all"]
        if f1 > self.best_f1:
            print(f"Best f1: {f1:.5f}/{self.best_f1:.5f}")
            self.best_f1 = f1
            self.model.save_weights("best_model.pt")
            if f1 > self.min_best:
                best_checkpoint_file = f"best_checkpoint_epoch_{epoch}_f1_{self.best_f1:.5f}.pt"
                self.model.save_weights(best_checkpoint_file)
                print(f"Saved best model file in {best_checkpoint_file}")
        print(
            f"[val] f1: {f1:.5f}, p: {precision:.5f} r: {recall:.5f} best_f1: {self.best_f1:.5f}"
        )
        # print(f"[val] {json.dumps(eval_result, ensure_ascii=False)}")
        for k, v in eval_result.items():
            if k == "total":
                continue
            v_list = [f"{x:.5f}" for x in v]
            print(f'"{k}": {v_list} ')

        logs.update({"best_f1": self.best_f1})
        logs.update({"f1": f1})
        logs.update({"precision": precision})
        logs.update({"recall": recall})

    @staticmethod
    def evaluate(model, dataloader, task_labels, threshold=0, debug=False):
        """评估函数，计算f1、precision、recall"""
        X, Y, Z = 0, 1e-10, 1e-10
        bad_cases = 0
        #  f = open("dev_pred.json", "w", encoding="utf-8")
        pbar = tqdm(desc="eval", ncols=100)
        #  for (x_true, label), ds in zip(dataloader, dataloader.dataset):
        for ds in dataloader.dataset:
            tagged_data, (tokens, mapping), (token_ids, segment_ids), encoded_label = ds

            idx, text, true_tags = tagged_data.idx, tagged_data.text, tagged_data.tags

            token_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
            segment_ids = torch.tensor([segment_ids], dtype=torch.long, device=device)

            outputs = model.predict([token_ids, segment_ids])
            outputs = [o[0].cpu().numpy() for o in outputs]  # [heads, seq_len, seq_len]

            relations_id2label = task_labels.relations_id2label
            spoes, (subjects_list, objects_list) = extract_spoes(
                text, mapping, outputs, relations_id2label, threshold=threshold
            )
            #  spoes = extract_spoes(
            #      model, text, tokens, mapping, token_ids, segment_ids, relations_id2label, threshold=threshold
            #  )

            T = set([SPO((tag.s.m, tag.p, tag.o.m)) for tag in true_tags])
            R = set([SPO((spo[0], spo[1], spo[2])) for spo in spoes])


            X += len(R & T)
            Y += len(R)
            Z += len(T)
            f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z

            if debug and R != T:
                if bad_cases < 5:
                    print("========================================")
                    print("idx:", idx, "text:", text[:50])
                    print("----------------------------------------")
                    print("found subjects:", subjects_list)
                    print("found objects:", objects_list)
                    print(f"T: {T}")
                    print(f"R: {R}")

                bad_cases += 1

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



