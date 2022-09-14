#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    import rich

    def print(*arg, **kwargs):
        rich.print(*arg, **kwargs)


except:
    pass

class DictObject(object):
    __setitem__ = object.__setattr__
    __getitem__ = object.__getattribute__

    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def update(self, *args, **kwargs):
        for k, v in dict(**kwargs).items():
            self.__setattr__(k, v)

    def __str__(self):
        return f"{str(self.__dict__)}"

# from .utils import init_theta, seed_everything, init_random, init_cuda, seg_generator
# from .utils import softmax, sigmoid, load_json_file, show_statistical_distribution, dataframe_to_examples, acc_and_f1, simple_accuracy, to_numpy, create_logger
# from .utils import batch_generator, slide_generator, split_train_eval_examples, shuffle_list, get_list_size, list_to_list
# from .utils import get_pred_results_file, get_submission_file
# from .utils import tokenize_en, generate_token_offsets, get_token_index, match_tokenized_to_untokenized
# from .utils import DataClassBase, remove_duplicate_entities, merge_entities
# from .dict_utils import DictObject
# from .file_reader import jsonl_reader, json_reader, tsv_reader
