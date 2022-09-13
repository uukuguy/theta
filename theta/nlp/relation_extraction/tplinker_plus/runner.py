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

from ..bert4torch.utils import seed_everything, EarlyStopping, sequence_padding
from .dataset import encode_text, encode_sentences
from .modeling import build_model, Evaluator, evaluate, collate_fn
from .modeling import predict_text
from .utils import check_tags
from .utils import split_sentences as default_split_sentences

script_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -------- run_training() --------


def run_training(
    args,
    entity_labels,
    relation_labels,
    train_dataset,
    val_dataset,
    bert_model_path,
):
    seed_everything(args.seed)

    entities_label2id = {label: i
                         for i, label in enumerate(entity_labels)}
    entities_id2label = {i: label
                         for i, label in enumerate(entity_labels)}
    relations_label2id = {label: i
                          for i, label in enumerate(relation_labels)}
    relations_id2label = {i: label
                          for i, label in enumerate(relation_labels)}

    num_training_epochs = args.num_training_epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    earlystopping_monitor = args.earlystopping_monitor
    earlystopping_patience = args.earlystopping_patience
    earlystopping_mode = args.earlystopping_mode
    best_model_file = "best_model.pt"  # args.best_model_file

    # -------- Data --------
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=batch_size)

    last_best_f1 = 0.0
    max_episodes = 100

    if os.path.exists(best_model_file):
        os.remove(best_model_file)

    with trange(max_episodes, desc=f"Episode", ncols=160) as pbar:
        for ep in pbar:
            num_training_steps = (len(train_dataloader) / batch_size) * num_training_epochs
            model = build_model(
                args, entity_labels, relation_labels, bert_model_path, learning_rate, num_training_steps
            )
            if os.path.exists(best_model_file):
                model.load_weights(best_model_file)
                print(f"Load model weights from best_model_file {os.path.realpath(best_model_file)}")
            else:
                last_model_file = "last_model.pt"
                if os.path.exists(last_model_file):
                    model.load_weights(last_model_file)
                    print(f"Load model weights from last_model_file {os.path.realpath(last_model_file)}")

            early_stopping = EarlyStopping(
                monitor=earlystopping_monitor, patience=earlystopping_patience, verbose=0, mode=earlystopping_mode
            )
            evaluator = Evaluator(model, val_dataloader, entities_id2label, relations_id2label, best_f1=last_best_f1)
            model.fit(
                train_dataloader,
                epochs=num_training_epochs,
                steps_per_epoch=None,
                callbacks=[evaluator, early_stopping]
            )

            torch.cuda.empty_cache()
            model = None
            if evaluator.best_f1 > last_best_f1:
                last_best_f1 = evaluator.best_f1
            else:
                print(
                    f"Episode best f1: {evaluator.best_f1:.5f} is not larger than last_best_f1: {last_best_f1:.5f}. Done."
                )
                break

            pbar.set_postfix({'best_f1': f"{last_best_f1:.5f}"})
            pbar.update(1)


def run_evaluating(args, entity_labels, relation_labels, val_dataset, bert_model_path):

    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=args.batch_size)

    model = build_model(args, entity_labels, relation_labels, bert_model_path)

    best_model_file = "best_model.pt"  # args.best_model_file
    model.load_weights(best_model_file)

    entities_label2id = {label: i
                         for i, label in enumerate(entity_labels)}
    entities_id2label = {i: label
                         for i, label in enumerate(entity_labels)}
    relations_label2id = {label: i
                          for i, label in enumerate(relation_labels)}
    relations_id2label = {i: label
                          for i, label in enumerate(relation_labels)}

    eval_result = evaluate(model, val_dataloader, entities_id2label, relations_id2label, threshold=0)
    print(f"eval_result: {eval_result}")


def run_predicting(args, entity_labels, relation_labels, test_data, best_model_file, bert_model_path, tokenizer):

    entities_label2id = {label: i
                         for i, label in enumerate(entity_labels)}
    entities_id2label = {i: label
                         for i, label in enumerate(entity_labels)}
    relations_label2id = {label: i
                          for i, label in enumerate(relation_labels)}
    relations_id2label = {i: label
                          for i, label in enumerate(relation_labels)}

    final_results = []
    X0, Y0, Z0 = 0, 0, 0

    model = build_model(args, entity_labels, relation_labels, bert_model_path)
    model.load_weights(best_model_file)

    for d in tqdm(test_data, desc="predict"):
        idx, full_text, true_tags = d[:3]

        token_ids, encoded_labels = encode_text(
            full_text, true_tags, entities_label2id, relations_label2id, args.max_length, tokenizer
        )
        batch_token_ids = [token_ids]
        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)

        full_tags, sent_tags_list = predict_text(
            args, model, full_text, tokenizer, entity_labels, relation_labels, repeat=args.repeat, threshold=0
        )

        full_tags['idx'] = idx

        if true_tags:
            #  print(f"full_text: {full_text}")
            #  print(f"full_tags: {full_tags}")
            #  print(f"true_tags: {true_tags}")
            check_results = check_tags(full_text, full_tags['tags'], true_tags)
            total_result = check_results['total']
            diff_list, (X, Y, Z), (f1, p, r) = total_result
            X0 += X
            Y0 += Y
            Z0 += Z
            if X != Z or Y != Z:
                print(f"----------------------------------------")
                print("idx:", idx, "text:", full_text)
                print("identical:", diff_list['identical'])
                print("added:", diff_list['added'])
                print("removed:", diff_list['removed'])
                print("")

        final_results.append((full_tags, sent_tags_list))

    if Z0 > 0 and Y0 > 0:
        f1, p, r = 2 * X0 / (Y0 + Z0), X0 / Y0, X0 / Z0
        print(f"P: {p:.5f}, R: {r:.5f}, F1: {f1:.5f}")

    return final_results
