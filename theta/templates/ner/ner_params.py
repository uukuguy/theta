#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------- Params --------------------
from theta.modeling import NerAppParams
experiment_params = NerAppParams()

# ----------------------------------------
LR = 2e-5
ADAM_EPS = 1e-8
N_AUGS = 0
N_EPOCHS = 10
MAX_SEQ_LENGTH = 256
BATCH_SIZE = 16
SEG_LEN = MAX_SEQ_LENGTH - 2
SEG_BACKOFF = 127
ENABLE_KD = False
MODEL_PATH = "/opt/share/pretrained/pytorch/bert-base-chinese"
CONFIDENCE = 0.0
LOSS_TYPE = "CrossEntropyLoss"
FOCALLOSS_GAMMA = 2.0
ALLOW_OVERLAP = False
NER_TYPE = "span"
SOFT_LABEL = False
CC = None

SEED = 8864
FOLD = 0

# ----------------------------------------
conf_common_params = {
    'learning_rate': LR,
    'adam_epsilon': ADAM_EPS,
    'fold': FOLD,
    'num_augments': N_AUGS,
    'enable_kd': ENABLE_KD,
    'num_train_epochs': N_EPOCHS,
    'train_max_seq_length': MAX_SEQ_LENGTH,
    'eval_max_seq_length': MAX_SEQ_LENGTH,
    'per_gpu_train_batch_size': BATCH_SIZE,
    'per_gpu_eval_batch_size': BATCH_SIZE,
    'per_gpu_predict_batch_size': BATCH_SIZE,
    'seg_len': SEG_LEN,
    'seg_backoff': SEG_BACKOFF,
    'model_path': MODEL_PATH,
    'confidence': CONFIDENCE,
    'loss_type': LOSS_TYPE,
    'focalloss_gamma': FOCALLOSS_GAMMA,
    'allow_overlap': ALLOW_OVERLAP,
    'cc': CC,
    'seed': SEED
}
conf_ner_params = {'ner_type': NER_TYPE, 'soft_label': SOFT_LABEL}

for k, v in conf_common_params.items():
    setattr(experiment_params.common_params, k, v)
for k, v in conf_ner_params.items():
    setattr(experiment_params.ner_params, k, v)
experiment_params.debug()

if __name__ == '__main__':
    pass
