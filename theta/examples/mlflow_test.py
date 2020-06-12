#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from random import random, randint

from mlflow import log_metric, log_param, log_artifacts

from dataclasses import dataclass


@dataclass
class Parameters:
    learning_rate: float = 2e-5
    max_seq_length: int = 512
    dataset_name: str = "default"
    fold: int = 0


global_parameters = Parameters()

if __name__ == "__main__":
    print("Running mlflow_tracking.py")

    log_param("param1", randint(0, 100))

    log_param("global_parameers", global_parameters)
    log_metric("foo", random())
    log_metric("foo", random() + 1)
    log_metric("foo", random() + 2)

    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    with open("outputs/test.txt", "w") as f:
        f.write("hello world!")

    log_artifacts("outputs")
