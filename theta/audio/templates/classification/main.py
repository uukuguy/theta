#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import datetime
import random
from tqdm import tqdm
from loguru import logger

import numpy as np
import pandas as pd
from dataset import BirdDataset
from models import get_model
from trainer import fit, predict
from utils import seed_everything, save_model_weights, count_parameters

from params import NUM_CLASSES

#  TODAY = str(datetime.date.today())
#  CP_TODAY = f"./checkpoints/{TODAY}/"
#
#  if not os.path.exists(CP_TODAY):
#      os.makedirs(CP_TODAY)


class Config:
    """
    Parameter used for training
    """
    # General
    seed = 2021
    verbose = 1
    verbose_eval = 1
    epochs_eval_min = 0  #25
    save = True

    # k-fold
    k = 5
    random_state = 42
    selected_folds = [0]

    # Model
    #  selected_model = "resnest50_fast_1s1x64d"
    # 0.82891
    selected_model = "resnext101_32x8d_wsl"
    # 0.81424
    #  selected_model = 'resnext50_32x4d'

    #  selected_model = "resnest50"
    #  selected_model = "resnest101"
    #  selected_model = "resnest200"
    #  selected_model = "resnest269"

    # 0.80648
    #  selected_model = "efficientnet-b3"
    # 0.79743
    #  selected_model = "efficientnet-b5"
    #  selected_model = "efficientnet-b8"

    use_conf = False
    use_extra = False

    # Training
    batch_size = 64
    epochs = 30 if use_extra else 40
    lr = 1e-3
    warmup_prop = 0.05
    val_bs = 64

    if "101" in selected_model or "b5" in selected_model or "b6" in selected_model:
        batch_size = batch_size // 2
        lr = lr / 2

    mixup_proba = 0.5
    alpha = 5

    name = "double"


class AudioParams:
    """
    Parameters used for the audio data
    """
    sr = 32000
    duration = 5

    # Melspectrogram
    n_mels = 128
    fmin = 20
    fmax = 16000


def train(config, df_train, df_val, fold):
    """
    Trains and validate a model

    Arguments:
        config {Config} -- Parameters
        df_train {pandas dataframe} -- Training metadata
        df_val {pandas dataframe} -- Validation metadata
        fold {int} -- Selected fold

    Returns:
        np array -- Validation predictions
    """

    print(f"    -> {len(df_train)} training birds")
    print(f"    -> {len(df_val)} validation birds")

    model = get_model(config.selected_model, num_classes=NUM_CLASSES).cuda()
    model.zero_grad()

    train_dataset = BirdDataset(df_train,
                                AudioParams,
                                use_conf=config.use_conf,
                                name="train")
    val_dataset = BirdDataset(df_val, AudioParams, train=False, name="val")

    n_parameters = count_parameters(model)
    print(f"    -> {n_parameters} trainable parameters\n")

    pred_val = fit(model,
                   train_dataset,
                   val_dataset,
                   epochs=config.epochs,
                   batch_size=config.batch_size,
                   val_bs=config.val_bs,
                   lr=config.lr,
                   warmup_prop=config.warmup_prop,
                   alpha=config.alpha,
                   mixup_proba=config.mixup_proba,
                   verbose_eval=config.verbose_eval,
                   epochs_eval_min=config.epochs_eval_min,
                   fold=fold)

    #  if config.save:
    #      save_model_weights(
    #          model,
    #          f"{config.selected_model}_{config.name}_{fold}.pt",
    #          cp_folder=CP_TODAY,
    #      )

    return pred_val


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--seed", default=Config.seed)

    args = parser.parse_args()

    return args


def main():

    args = get_args()

    seed_everything(args.seed)

    if args.do_train:
        os.makedirs("./checkpoints", exist_ok=True)

        from audio_data import train_data_generator
        all_train_data = [(label, wav_file)
                          for label, wav_file in train_data_generator()]
        all_train_data = random.sample(all_train_data, len(all_train_data))

        num_train_samples = int(len(all_train_data) * 0.9)
        train_data = all_train_data[:num_train_samples]
        val_data = all_train_data[num_train_samples:]

        df_train = pd.DataFrame(train_data,
                                columns=['ebird_code', 'file_path'])
        logger.info(f"df_train: {df_train.shape}")
        df_val = pd.DataFrame(val_data, columns=['ebird_code', 'file_path'])
        logger.info(f"df_val: {df_val.shape}")

        fold = 0
        pred_val = train(Config, df_train, df_val, fold=fold)

    if args.do_predict:
        model = get_model(Config.selected_model,
                          num_classes=NUM_CLASSES).cuda()
        from utils import load_model_weights
        model_file = "./checkpoints/best_model_0.pt"
        model = load_model_weights(model, model_file)

        from audio_data import test_data_generator
        test_data = [(None, wav_file) for _, wav_file in test_data_generator()]
        df_test = pd.DataFrame(test_data, columns=['ebird_code', 'file_path'])
        test_dataset = BirdDataset(df_test,
                                   AudioParams,
                                   use_conf=Config.use_conf,
                                   train=False,
                                   name="test")
        preds = predict(model, test_dataset, batch_size=64)
        logger.info(f"preds: {preds.shape}")
        preds_file = "preds"
        np.save(preds_file, preds)
        logger.info(f"Saved {preds_file}")

        labels = np.argmax(preds, axis=-1)

        submission_file = "./submission.csv"
        with open(submission_file, "w") as wt:
            for (_, wav_file), label in tqdm(zip(test_data, labels),
                                             desc="Submission"):
                wav_file = os.path.basename(wav_file)
                wt.write(f"{wav_file} B{label:03d}\n")
        logger.info(f"Saved {len(labels)} lines in {submission_file}")


if __name__ == '__main__':
    main()
