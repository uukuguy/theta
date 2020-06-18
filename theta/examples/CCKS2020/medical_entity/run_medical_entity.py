#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, json, random, copy
from tqdm import tqdm
from loguru import logger
from pathlib import Path

from theta.utils import load_json_file, split_train_eval_examples
from theta.modeling import LabeledText, load_ner_examples, load_ner_labeled_examples, save_ner_preds, show_ner_datainfo

if os.environ['NER_TYPE'] == 'span':
    from theta.modeling.ner_span import load_model, NerTrainer, get_args
else:
    from theta.modeling.ner import load_model, NerTrainer, get_args

ner_labels = ['疾病和诊断', '影像检查', '实验室检验', '手术', '药物', '解剖部位']


# -------------------- Data --------------------
def clean_text(text):
    if text:
        text = text.strip()
        #  text = re.sub('\t', ' ', text)
    return text


def train_data_generator(train_file):

    lines = load_json_file(train_file)

    for i, x in enumerate(tqdm(lines)):
        guid = str(i)
        text = clean_text(x['originalText'])
        sl = LabeledText(guid, text)

        entities = x['entities']
        for entity in entities:
            start_pos = entity['start_pos']
            end_pos = entity['end_pos'] - 1
            category = entity['label_type']
            sl.add_entity(category, start_pos, end_pos)

        yield str(i), text, None, sl.entities


def load_train_val_examples(args):
    lines = []
    for guid, text, _, entities in train_data_generator(args.train_file):
        sl = LabeledText(guid, text, entities)
        lines.append({'guid': guid, 'text': text, 'entities': entities})

    allow_overlap = args.allow_overlap
    if args.num_augements > 0:
        allow_overlap = False

    train_base_examples = load_ner_labeled_examples(
        lines,
        ner_labels,
        seg_len=args.seg_len,
        seg_backoff=args.seg_backoff,
        num_augements=args.num_augements,
        allow_overlap=allow_overlap)

    train_examples, val_examples = split_train_eval_examples(
        train_base_examples,
        train_rate=args.train_rate,
        fold=args.fold,
        shuffle=True)

    logger.info(f"Loaded {len(train_examples)} train examples, "
                f"{len(val_examples)} val examples.")
    return train_examples, val_examples


def test_data_generator(test_file):

    lines = load_json_file(test_file)
    for i, s in enumerate(tqdm(lines)):
        guid = str(i)
        text_a = clean_text(s['originalText'])

        yield guid, text_a, None, None


def load_test_examples(args):
    test_base_examples = load_ner_examples(test_data_generator,
                                           args.test_file,
                                           seg_len=args.seg_len,
                                           seg_backoff=args.seg_backoff)

    logger.info(f"Loaded {len(test_base_examples)} test examples.")
    return test_base_examples


# -------------------- Model --------------------
class AppTrainer(NerTrainer):
    def __init__(self, args, ner_labels):
        super(AppTrainer, self).__init__(args, ner_labels, build_model=None)

    #  def on_predict_end(self, args, test_dataset):
    #      super(Trainer, self).on_predict_end(args, test_dataset)


def generate_submission(args):
    reviews_file = f"{args.latest_dir}/{args.dataset_name}_reviews_fold{args.fold}.json"
    reviews = json.load(open(reviews_file, 'r'))

    submission_file = f"{args.submissions_dir}/{args.dataset_name}_submission_{args.local_id}.json.txt"
    with open(submission_file, 'w') as wt:
        for guid, json_data in reviews.items():
            output_data = {'originalText': json_data['text'], 'entities': []}
            for json_entity in json_data['entities']:
                output_data['entities'].append({
                    'label_type':
                    json_entity['category'],
                    'overlap':
                    0,
                    'start_pos':
                    json_entity['start'],
                    'end_pos':
                    json_entity['end'] + 1
                })
            output_data['entities'] = sorted(output_data['entities'],
                                             key=lambda x: x['start_pos'])
            output_string = json.dumps(output_data, ensure_ascii=False)
            wt.write(f"{output_string}\n")

    logger.info(f"Saved {len(reviews)} lines in {submission_file}")

    import shutil
    if os.path.exists(args.local_dir):
        shutil.rmtree(args.local_dir)
    shutil.copytree(args.latest_dir, args.local_dir)
    logger.info("Copy tree {args.latest_dir} to {args.loca_dir}")


def main(args):

    if args.do_eda:
        show_ner_datainfo(ner_labels, train_data_generator, args.train_file,
                          test_data_generator, args.test_file)

    elif args.do_submit:
        generate_submission(args)

    else:
        trainer = AppTrainer(args, ner_labels)

        if args.do_train:
            train_examples, val_examples = load_train_val_examples(args)
            trainer.train(args, train_examples, val_examples)

        elif args.do_eval:
            _, eval_examples = load_train_val_examples(args)
            model = load_model(args)
            trainer.evaluate(args, model, eval_examples)

        elif args.do_predict:
            test_examples = load_test_examples(args)
            model = load_model(args)
            trainer.predict(args, model, test_examples)
            save_ner_preds(args, trainer.pred_results, test_examples)


if __name__ == '__main__':

    def add_special_args(parser):
        return parser

    args = get_args([add_special_args])
    main(args)
