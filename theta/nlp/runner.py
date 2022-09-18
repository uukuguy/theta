#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from tqdm import tqdm, trange

import torch
from torch.utils.data import DataLoader

try:
    import rich

    def print(*arg, **kwargs):
        rich.print(*arg, **kwargs)

except:
    pass

from ..utils import get_device, build_dataloader, torch_distributed_zero_first
from .bert4torch.utils import seed_everything, EarlyStopping, Callback
# from .utils import check_tags

script_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

# device = "cuda" if torch.cuda.is_available() else "cpu"

# -------- run_training() --------

class DistributedTrainingEpoch(Callback):
    def __init__(self, local_rank, dist_sampler=None):
        self.local_rank = local_rank
        self.dist_sampler=dist_sampler
    
    def on_epoch_begin(self, global_step, epoch, logs=None):
        if self.local_rank >= 0:
            self.dist_sampler.set_epoch(epoch)


def run_training(args, Model, Evaluator, train_dataset, val_dataset):
    seed_everything(args.seed)
    device = get_device(local_rank=args.local_rank)

    print("args.debug:", args.debug)

    num_training_epochs = args.num_training_epochs
    max_training_episodes = args.max_training_episodes
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    earlystopping_monitor = args.earlystopping_monitor
    earlystopping_patience = args.earlystopping_patience
    earlystopping_mode = args.earlystopping_mode
    best_model_file = args.best_model_file

    last_best_f1 = 0.0

    if args.local_rank in [-1, 0]:
        if os.path.exists(best_model_file):
            os.remove(best_model_file)

    if args.local_rank >= 0:
        max_training_episodes = 1
    with trange(max_training_episodes, desc=f"Episode", ncols=160) as pbar:
        for ep in pbar:
            if args.local_rank >= 0:
                torch.distributed.init_process_group('nccl')
                torch.distributed.barrier()

            # -------- Data --------
            train_dataloader = build_dataloader(train_dataset,
                                                batch_size=batch_size,
                                                collate_fn=Model.collate_fn,
                                                shuffle=True,
                                                local_rank=args.local_rank)
            val_dataloader = build_dataloader(val_dataset,
                                            batch_size=batch_size,
                                            collate_fn=Model.collate_fn,
                                            shuffle=False,
                                            local_rank=args.local_rank)


            num_training_steps = (len(train_dataloader) /
                                  batch_size) * num_training_epochs
            model = Model.build_model(args, num_training_steps)
            if os.path.exists(best_model_file):
                print(
                    f"Load model weights from best_model_file {os.path.realpath(best_model_file)}"
                )
                model.load_weights(best_model_file)
            else:
                last_model_file = "last_model.pt"
                if os.path.exists(last_model_file):
                    print(
                        f"Load model weights from last_model_file {os.path.realpath(last_model_file)}"
                    )
                    model.load_weights(last_model_file)

            callbacks = []

            if args.local_rank in [-1, 0]:
                evaluator = Evaluator(
                    model,
                    val_dataloader,
                    task_labels=args.task_labels,
                    best_f1=last_best_f1,
                    min_best=args.min_best,
                    threshold=args.extract_threshold,
                    debug=args.debug,
                )
                callbacks.append(evaluator)

            if earlystopping_patience >= 0:
                early_stopping = EarlyStopping(
                    monitor=earlystopping_monitor,
                    patience=earlystopping_patience,
                    verbose=0,
                    mode=earlystopping_mode,
                )
                callbacks.append(early_stopping)

            dist_training_epoch = DistributedTrainingEpoch(local_rank=args.local_rank, dist_sampler=train_dataloader.sampler)
            callbacks.append(dist_training_epoch)

            model.fit(
                train_dataloader,
                epochs=num_training_epochs,
                steps_per_epoch=None,
                callbacks=callbacks,
            )

            torch.cuda.empty_cache()
            model = None

            if args.local_rank >= 0:
                torch.distributed.barrier()
                torch.distributed.destroy_process_group()

            if args.local_rank in [-1, 0]:
                if evaluator.best_f1 > last_best_f1:
                    last_best_f1 = evaluator.best_f1
                else:
                    print(
                        f"Episode best f1: {evaluator.best_f1:.5f} is not larger than last_best_f1: {last_best_f1:.5f}. Done."
                    )
                    break

            pbar.set_postfix({"best_f1": f"{last_best_f1:.5f}"})
            pbar.update(1)

            args.learning_rate /= 2


def run_evaluating(args, Model, Evaluator, val_dataset):
    device = get_device(local_rank=args.local_rank)

    val_dataloader = build_dataloader(val_dataset,
                                      batch_size=args.batch_size,
                                      collate_fn=Model.collate_fn,
                                      shuffle=False,
                                      local_rank=args.local_rank)

    model = Model.build_model(args)

    best_model_file = args.best_model_file
    model.load_weights(best_model_file)

    eval_result = Evaluator.evaluate(model,
                                     val_dataloader,
                                     args.task_labels,
                                     threshold=args.extract_threshold,
                                     debug=args.debug)
    print(f"eval_result: {eval_result}")


def run_predicting(args, Model, Evaluator, test_data, task_model_file,
                   tokenizer):
    device = get_device(local_rank=args.local_rank)

    final_results = []
    X0, Y0, Z0 = 0, 0, 0

    model = Model.build_model(args)
    print(f"Load weights from {task_model_file}")
    model.load_weights(task_model_file)

    for d in tqdm(test_data, desc="predict"):
        idx, full_text, true_tags = d.idx, d.text, d.tags

        full_tags = Model.predict_text(args,
                                       model,
                                       full_text,
                                       tokenizer,
                                       repeat=args.repeat,
                                       threshold=args.extract_threshold)
        # print("full_tags:", full_tags)

        if False:
            # if true_tags:
            #  print(f"full_text: {full_text}")
            #  print(f"full_tags: {full_tags}")
            #  print(f"true_tags: {true_tags}")
            check_results = check_tags(full_text, full_tags["tags"], true_tags)
            total_result = check_results["total"]
            diff_list, (X, Y, Z), (f1, p, r) = total_result
            X0 += X
            Y0 += Y
            Z0 += Z
            if X != Z or Y != Z:
                print(f"----------------------------------------")
                print("idx:", idx, "text:", full_text)
                print("identical:", diff_list["identical"])
                print("added:", diff_list["added"])
                print("removed:", diff_list["removed"])
                print("")

        final_results.append(full_tags)

    if Z0 > 0 and Y0 > 0:
        f1, p, r = 2 * X0 / (Y0 + Z0), X0 / Y0, X0 / Z0
        print(f"P: {p:.5f}, R: {r:.5f}, F1: {f1:.5f}")

    # print("final_results[:5]:", final_results[:5])
    return final_results
