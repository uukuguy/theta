#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json, re

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

script_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

from ..bert4torch.models import build_transformer_model
from ..bert4torch.utils import sequence_padding, Callback
from ..bert4torch.optimizers import get_linear_schedule_with_warmup
from ..bert4torch.tokenizers import Tokenizer
from .mlm_dataset import TrainingDatasetRoBERTa


task_name = "roberta_mlm"

bert_model_path="/opt/local/pretrained/bert-base-chinese"
config_path = f"{bert_model_path}/bert_config.json"
checkpoint_path = f"{bert_model_path}/pytorch_model.bin"  # 如果从零训练，就设为None
dict_path = f"{bert_model_path}/vocab.txt"  

model_saved_path = "./bert_model.ckpt"

batch_size = 8
mask_rate=0.15
sequence_length = 512  # 文本长度
max_file_num = 5000  # 最大保存的文件个数

learning_rate = 0.00176
weight_decay_rate = 0.01  # 权重衰减
num_warmup_steps = 3125
num_train_steps = 125000
steps_per_epoch = 10000
grad_accum_steps = 16  # 大于1即表明使用梯度累积
epochs = num_train_steps * grad_accum_steps // steps_per_epoch

device = "cuda" if torch.cuda.is_available() else "cpu"

local_rank=-1
torch.cuda.set_device(local_rank)
device=torch.device('cuda', local_rank)
torch.distributed.init_process_group(backend='nccl')

tokenizer = Tokenizer(dict_path, do_lower_case=True)

all_texts = []
for file in [
    "../data/rawdata/train-652346.json",
    "../data/rawdata/evalA-428907.json",
]:
    with open(file) as fd:
        lines = fd.readlines()
        lines = [line.strip() for line in lines]
        for line in lines:
            json_data = json.loads(line)
            text = json_data["text"]
            all_texts.extend(re.findall(".*?[\n。]+", text))



# 读取数据集，构建数据张量
class MyDataset(Dataset):
    def __init__(self, tokenizer, word_segment, mask_rate=0.15, sequence_length=512):
        super(MyDataset, self).__init__()

        self.TD = TrainingDatasetRoBERTa(
            all_texts, tokenizer, word_segment, mask_rate=mask_rate, sequence_length=sequence_length
        )

    def __getitem__(self, index):
        return self.TD.map_features[str(index)]

    def __len__(self):
        return len(self.TD.map_features)



def collate_fn(batch):
    batch_token_ids, batch_labels = [], []
    for item in batch:
        batch_token_ids.append(item["input_ids"])
        batch_labels.append(item["masked_lm_labels"])

    batch_token_ids = torch.tensor(
        sequence_padding(batch_token_ids), dtype=torch.long, device=device
    )
    batch_labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
    return [batch_token_ids], batch_labels


import jieba

jieba.initialize()
def word_segment(text):
    return jieba.lcut(text)


def get_train_dataloader():
    from torch.utils.data.distributed import DistributedSampler

    train_dataset = MyDataset(tokenizer, word_segment, mask_rate=mask_rate, sequence_length=sequence_length),
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    return train_dataloader
train_dataloader = get_train_dataloader()

model = build_transformer_model(
    config_path, checkpoint_path, segment_vocab_size=0, with_mlm=True
).to(device)

# weight decay
param_optimizer = list(model.named_parameters())
no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [
            p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
        ],
        "weight_decay": weight_decay_rate,
    },
    {
        "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]


class MyLoss(nn.CrossEntropyLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, output, batch_labels):
        y_preds = output[-1]
        y_preds = y_preds.reshape(-1, y_preds.shape[-1])
        return super().forward(y_preds, batch_labels.flatten())


# 定义使用的loss和optimizer，这里支持自定义
optimizer = optim.Adam(
    optimizer_grouped_parameters, lr=learning_rate, weight_decay=weight_decay_rate
)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
)
model.compile(loss=MyLoss(ignore_index=0), optimizer=optimizer, scheduler=scheduler)

if os.path.exists(model_saved_path):
    print(f"Load model weights from {model_saved_path}")
    model.load_weights(model_saved_path)

class ModelCheckpoint(Callback):
    """自动保存最新模型"""

    def on_dataloader_end(self, logs=None):
        # 重新生成dataloader
        model.train_dataloader = get_train_dataloader()

    def on_epoch_end(self, global_step, epoch, logs=None):
        model.save_weights(model_saved_path)


if __name__ == "__main__":
    # 保存模型
    checkpoint = ModelCheckpoint()

    # 模型训练
    model.fit(
        train_dataloader,
        steps_per_epoch=steps_per_epoch,
        grad_accumulation_steps=grad_accum_steps,
        epochs=epochs,
        callbacks=[checkpoint],
    )
