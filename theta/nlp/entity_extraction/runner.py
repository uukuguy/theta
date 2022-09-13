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

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------- run_training() --------


def run_training(
    args,
    ner_labels,
    train_dataset,
    val_dataset,
    bert_model_path,
    max_training_episodes=100,
):
    seed_everything(args.seed)

    categories_label2id = {label: i for i, label in enumerate(ner_labels)}
    categories_id2label = dict(
        (value, key) for key, value in categories_label2id.items()
    )

    num_training_epochs = args.num_training_epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    earlystopping_monitor = args.earlystopping_monitor
    earlystopping_patience = args.earlystopping_patience
    earlystopping_mode = args.earlystopping_mode
    best_model_file = "best_model.pt"  # args.best_model_file

    # -------- Data --------
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=batch_size
    )
    val_dataloader = DataLoader(
        val_dataset, collate_fn=collate_fn, batch_size=batch_size
    )

    # -------- Model --------
    #  num_training_steps = (len(train_dataloader) / batch_size) * num_training_epochs
    #  model = build_model(args, ner_labels, bert_model_path, learning_rate, num_training_steps)

    last_best_f1 = 0.0

    if os.path.exists(best_model_file):
        os.remove(best_model_file)

    with trange(max_training_episodes, desc=f"Episode", ncols=160) as pbar:
        for ep in pbar:
            num_training_steps = (
                len(train_dataloader) / batch_size
            ) * num_training_epochs
            model = build_model(
                args, ner_labels, bert_model_path, learning_rate, num_training_steps
            )
            if os.path.exists(best_model_file):
                model.load_weights(best_model_file)
                print(
                    f"Load model weights from best_model_file {os.path.realpath(best_model_file)}"
                )
            else:
                last_model_file = "last_model.pt"
                if os.path.exists(last_model_file):
                    model.load_weights(last_model_file)
                    print(
                        f"Load model weights from last_model_file {os.path.realpath(last_model_file)}"
                    )

            early_stopping = EarlyStopping(
                monitor=earlystopping_monitor,
                patience=earlystopping_patience,
                verbose=0,
                mode=earlystopping_mode,
            )
            evaluator = Evaluator(
                model,
                val_dataloader,
                categories_id2label,
                best_f1=last_best_f1,
                min_best=args.min_best,
            )
            model.fit(
                train_dataloader,
                epochs=num_training_epochs,
                steps_per_epoch=None,
                callbacks=[evaluator, early_stopping],
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

            pbar.set_postfix({"best_f1": f"{last_best_f1:.5f}"})
            pbar.update(1)


def run_evaluating(args, ner_labels, val_dataset, bert_model_path):

    val_dataloader = DataLoader(
        val_dataset, collate_fn=collate_fn, batch_size=args.batch_size
    )

    model = build_model(args, ner_labels, bert_model_path)

    best_model_file = "best_model.pt"  # args.best_model_file
    model.load_weights(best_model_file)

    categories_label2id = {label: i for i, label in enumerate(ner_labels)}
    categories_id2label = dict(
        (value, key) for key, value in categories_label2id.items()
    )
    eval_result = evaluate(model, val_dataloader, categories_id2label, threshold=0)
    print(f"eval_result: {eval_result}")


def run_predicting(
    args, ner_labels, test_data, best_model_file, bert_model_path, tokenizer
):

    categories_label2id = {label: i for i, label in enumerate(ner_labels)}
    categories_id2label = dict(
        (value, key) for key, value in categories_label2id.items()
    )

    final_results = []
    X0, Y0, Z0 = 0, 0, 0

    model = build_model(args, ner_labels, bert_model_path)
    model.load_weights(best_model_file)

    for d in tqdm(test_data, desc="predict"):
        idx, full_text, true_tags = d[:3]

        token_ids, encoded_labels = encode_text(
            full_text, true_tags, categories_label2id, args.max_length, tokenizer
        )
        batch_token_ids = [token_ids]
        batch_token_ids = torch.tensor(
            sequence_padding(batch_token_ids), dtype=torch.long, device=device
        )

        full_tags, sent_tags_list = predict_text(
            args,
            model,
            full_text,
            tokenizer,
            ner_labels,
            repeat=args.repeat,
            threshold=0,
        )

        full_tags["idx"] = idx

        if true_tags:
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
                print("true_tags:", true_tags)
                print("")
                print("identical:", diff_list["identical"])
                print("added:", diff_list["added"])
                print("removed:", diff_list["removed"])
                print("")

        final_results.append((full_tags, sent_tags_list))

    if Z0 > 0 and Y0 > 0:
        f1, p, r = 2 * X0 / (Y0 + Z0), X0 / Y0, X0 / Z0
        print(f"P: {p:.5f}, R: {r:.5f}, F1: {f1:.5f}")

    return final_results


#  def run_predicting_for_dataset(args, ner_labels, test_dataset, best_model_file, bert_model_path):
#
#      categories_label2id = {label: i
#                             for i, label in enumerate(ner_labels)}
#      categories_id2label = dict((value, key) for key, value in categories_label2id.items())
#
#      final_results = []
#      X0, Y0, Z0 = 0, 0, 0
#
#      model = build_model(args, ner_labels, bert_model_path)
#      model.load_weights(best_model_file)
#
#      test_data = test_dataset.data
#      for i in range(len(test_dataset)):
#          idx, full_text, true_tags = test_data[i]
#
#          token_ids, encoded_labels = test_dataset[i]
#          batch_token_ids = [token_ids]
#          batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
#
#          #  full_tags, sent_tags_list = predict_sentences(
#          #      args, model, sentences, tokenizer, repeat=repeat, threshold=0)
#          full_tags, sent_tags_list = predict_batch_tokens(
#              args, model, batch_token_ids, categories_id2label, repeat=args.repeat, threshold=0
#          )
#          full_tags['idx'] = idx
#
#          if true_tags:
#              check_results = check_tags(full_text, full_tags, true_tags)
#              total_result = check_results['total']
#              diff_list, (X, Y, Z), (f1, p, r) = total_result
#              X0 += X
#              Y0 += Y
#              Z0 += Z
#              if X != Z or Y != Z:
#                  print(f"----------------------------------------")
#                  print(f"idx: {idx}, text: {full_text}")
#                  print(f"diff_list: {diff_list}")
#                  print("")
#
#          final_results.append((full_tags, sent_tags_list))
#
#      if Z0 > 0 and Y0 > 0:
#          f1, p, r = 2 * X0 / (Y0 + Z0), X0 / Y0, X0 / Z0
#          print(f"P: {p:.5f}, R: {r:.5f}, F1: {f1:.5f}")
#
#      return final_results
