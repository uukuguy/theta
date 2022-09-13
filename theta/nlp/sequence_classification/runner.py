#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from tqdm import tqdm, trange

import torch
from torch.utils.data import DataLoader


from ..bert4torch.utils import seed_everything, EarlyStopping, sequence_padding
from .tagging import TaskLabels, TaskTag, TaggedData
from .dataset import encode_text, encode_sentences
from .modeling import build_model, Model, Evaluator, evaluate
from .modeling import predict_text

script_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------- run_training() --------


def run_training(args, train_dataset, val_dataset):
    seed_everything(args.seed)

    num_training_epochs = args.num_training_epochs
    max_training_episodes = args.max_training_episodes
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    earlystopping_monitor = args.earlystopping_monitor
    earlystopping_patience = args.earlystopping_patience
    earlystopping_mode = args.earlystopping_mode
    best_model_file = "best_model.pt"  # args.best_model_file

    # -------- Data --------
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=Model.collate_fn, batch_size=batch_size
    )
    val_dataloader = DataLoader(
        val_dataset, collate_fn=Model.collate_fn, batch_size=batch_size
    )

    last_best_acc = 0.0

    if os.path.exists(best_model_file):
        os.remove(best_model_file)

    with trange(max_training_episodes, desc=f"Episode", ncols=160) as pbar:
        for ep in pbar:
            num_training_steps = (
                len(train_dataloader) / batch_size
            ) * num_training_epochs
            model = build_model(args, num_training_steps)
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

            evaluator = Evaluator(
                model,
                val_dataloader,
                task_labels=args.task_labels,
                best_acc=last_best_acc,
                min_best=args.min_best,
                threshold=args.extract_threshold,
            )
            callbacks = [evaluator]

            if earlystopping_patience >= 0:
                early_stopping = EarlyStopping(
                    monitor=earlystopping_monitor,
                    patience=earlystopping_patience,
                    verbose=0,
                    mode=earlystopping_mode,
                )
                callbacks.append(early_stopping)

            model.fit(
                train_dataloader,
                epochs=num_training_epochs,
                steps_per_epoch=None,
                callbacks=[evaluator, early_stopping],
            )

            torch.cuda.empty_cache()
            model = None
            if evaluator.best_acc > last_best_acc:
                last_best_acc = evaluator.best_acc
            else:
                print(
                    f"Episode best acc: {evaluator.best_acc:.5f} is not larger than last_best_acc: {last_best_acc:.5f}. Done."
                )
                break

            pbar.set_postfix({"best_acc": f"{last_best_acc:.5f}"})
            pbar.update()


def run_evaluating(args, val_dataset):

    val_dataloader = DataLoader(
        val_dataset, collate_fn=Model.collate_fn, batch_size=args.batch_size
    )

    model = build_model(args)

    best_model_file = "best_model.pt"  # args.best_model_file
    model.load_weights(best_model_file)

    eval_result = evaluate(
        model, val_dataloader, args.task_labels, threshold=args.extract_threshold
    )
    print(f"eval_result: {eval_result}")


def run_predicting(args, test_data, best_model_file, tokenizer):

    final_results = []
    X0, Y0, Z0 = 0, 0, 0

    model = build_model(args)
    model.load_weights(best_model_file)

    for d in tqdm(test_data, desc="predict"):
        idx, text_a, text_b, true_tag = d.idx, d.text_a, d.text_b, d.task_tag

        pred_label = predict_text(
            args, model, text_a, text_b, tokenizer, threshold=args.extract_threshold
        )

        if true_tag:
            right = False
            if pred_label == true_tag.label:
                X0 += 1
                right = True
            Y0 += 1
            Z0 += 1
            if not right:
                print(f"----------------------------------------")
                print(d)
                print("")
                print("pred_label:", pred_label)
                print("")

        final_results.append(pred_label)

    if Z0 > 0 and Y0 > 0:
        f1, p, r = 2 * X0 / (Y0 + Z0), X0 / Y0, X0 / Z0
        print(f"P: {p:.5f}, R: {r:.5f}, F1: {f1:.5f}")

    return final_results
