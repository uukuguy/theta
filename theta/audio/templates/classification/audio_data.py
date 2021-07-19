#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
from glob import glob
from tqdm import tqdm, trange
from loguru import logger


def train_data_generator():
    for i in trange(100, desc="train"):
        wav_dir = f"./data/rawdata/train_data/train_data/B{i:03d}"
        wav_files = glob(f"{wav_dir}/*.wav")
        wav_files = sorted(wav_files)
        label = f"B{i:03d}"
        #  n_samples = 0
        for wav_file in wav_files:
            yield label, wav_file

    for i in trange(100, desc='val'):
        wav_dir = f"./data/rawdata/dev_data/B{i:03d}"
        wav_files = glob(f"{wav_dir}/*.wav")
        wav_files = sorted(wav_files)
        label = f"B{i:03d}"
        for wav_file in wav_files:
            yield label, wav_file


#  def val_data_generator():
#      for i in trange(100, desc='val'):
#          wav_dir = f"./data/rawdata/dev_data/B{i:03d}"
#          wav_files = glob(f"{wav_dir}/*.wav")
#          wav_files = sorted(wav_files)
#          label = f"B{i:03d}"
#          for wav_file in wav_files:
#              for _ in range(3):
#                  yield label, wav_file


def test_data_generator():
    wav_files = glob(f"./data/rawdata/test_data/test_data_blind_name/*.wav")
    wav_files = sorted(wav_files)
    for wav_file in tqdm(wav_files, desc="test"):
        yield None, wav_file
