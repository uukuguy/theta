#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .arguments import TaskArguments


def get_default_args():
    task_args = TaskArguments.parse_args()
    return task_args

def get_default_tokenizer(dict_path):
    from .bert4torch.tokenizers import Tokenizer

    tokenizer = Tokenizer(dict_path, do_lower_case=True)

    return tokenizer