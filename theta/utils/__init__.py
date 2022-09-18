#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.utils.data.distributed import DistributedSampler
from contextlib import contextmanager

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


def get_device(local_rank=-1):
    import torch
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


class DistributedSamplerWithLoop(DistributedSampler):
    def __init__(self, dataset, batch_size, **kwargs):
        super().__init__(dataset, **kwargs)
        self.batch_size = batch_size

    def __iter__(self):
        indices = list(super().__iter__())
        remainder = 0 if len(indices) % self.batch_size == 0 else self.batch_size - len(indices) % self.batch_size
        # DistributedSampler already added samples from the beginning to make the number of samples a round multiple
        # of the world size, so we skip those.
        start_remainder = 1 if self.rank < len(self.dataset) % self.num_replicas else 0
        indices += indices[start_remainder : start_remainder + remainder]
        return iter(indices)

def build_dataloader(dataset, batch_size, collate_fn, shuffle=False, local_rank=-1) :
    from torch.utils.data import DataLoader

    if local_rank == -1:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
    else:

        train_sampler = DistributedSamplerWithLoop(dataset, batch_size=batch_size)
        dataloader = DataLoader(
            dataset,
            sampler=train_sampler,
            batch_size=batch_size,
            collate_fn=collate_fn,
        )
    return dataloader

def dist_process_index(local_rank=-1):
    if local_rank != -1:
        return torch.distributed.get_rank()
    else:
        return 0

def local_process_index(local_rank=-1):
    if local_rank != -1:
        return local_rank
    else:
        return 0

@contextmanager
def main_process_first(local_rank, local=True, desc="work"):
    main_process_desc = "main process"
    if local:
        is_main_process = local_process_index(local_rank) == 0
        main_process_desc = "main local process"
    else:
        is_main_process = dist_process_index(local_rank) == 0

    try:
        if not is_main_process:
            # tell all replicas to wait
            # logger.debug(f"{self.process_index}: waiting for the {main_process_desc} to perform {desc}")
            torch.distributed.barrier()
        yield
    finally:
        if is_main_process:
            # the wait is over
            # logger.debug(f"{self.process_index}: {main_process_desc} completed {desc}, releasing all replicas")
            torch.distributed.barrier()

# from .utils import init_theta, seed_everything, init_random, init_cuda, seg_generator
# from .utils import softmax, sigmoid, load_json_file, show_statistical_distribution, dataframe_to_examples, acc_and_f1, simple_accuracy, to_numpy, create_logger
# from .utils import batch_generator, slide_generator, split_train_eval_examples, shuffle_list, get_list_size, list_to_list
# from .utils import get_pred_results_file, get_submission_file
# from .utils import tokenize_en, generate_token_offsets, get_token_index, match_tokenized_to_untokenized
# from .utils import DataClassBase, remove_duplicate_entities, merge_entities
# from .dict_utils import DictObject
# from .file_reader import jsonl_reader, json_reader, tsv_reader
