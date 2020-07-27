#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, glob, json, datetime
from tqdm import tqdm
from loguru import logger
import rich
from rich import print
from rich.console import Console
console = Console()

from .modeling import NerDataset

all_models = []


def find_models(args):
    global all_models

    output_dir = args.output_dir
    files = glob.glob(os.path.join(output_dir, "*/local_id"))

    for file in files:
        local_id = None
        with open(file, 'r') as rd:
            local_id = rd.readline().strip()
            model_path = os.path.split(file)[0]

            ctime = os.stat(file).st_ctime
            ctime = datetime.datetime.fromtimestamp(ctime).strftime(
                '%Y/%m/%d %H:%M:%S')

            all_models.append((local_id, model_path, ctime))
    all_models = sorted(all_models, key=lambda x: x[2], reverse=True)


def get_model(local_id):
    for model in all_models:
        if model[0] == local_id:
            return model
    return None


def get_model_args_path(model):
    if model is None:
        args_path = os.path.join(f"{args.output_dir}",
                                 "latest/best/training_args.json")
        if not os.path.exists(args_path):
            args_path = os.path.join(f"{args.output_dir}",
                                     "latest/training_args.json")

    else:
        args_path = os.path.join(model[1], "best/training_args.json")
        if not os.path.exists(args_path):
            args_path = os.path.join(model[1], "training_args.json")

    return args_path


def show_model(args):
    logger.info(f"{args.local_id}")
    model = None
    if args.local_id:
        model_id = args.local_id[0]
        model = get_model(model_id)
    #  for model_id in args.local_id:
    #      model = get_model(model_id)

    args_path = get_model_args_path(model)

    ctime = os.stat(args_path).st_ctime
    ctime = datetime.datetime.fromtimestamp(ctime).strftime(
        '%Y/%m/%d %H:%M:%S')

    training_args = json.load(open(args_path))
    logger.warning(f"-------- {training_args['local_id']} --------")
    logger.info(f'{ctime}')

    logger.info(f"dataset_name: {training_args['dataset_name']}")
    logger.info(f"num_train_epochs: {training_args['num_train_epochs']}")
    logger.info(f"learning_rate: {training_args['learning_rate']}")
    logger.info(
        f"train_max_seq_length: {training_args['train_max_seq_length']}")
    logger.info(f"eval_max_seq_length: {training_args['eval_max_seq_length']}")
    logger.info(
        f"per_gpu_train_batch_size: {training_args['per_gpu_train_batch_size']}"
    )
    logger.info(
        f"per_gpu_eval_batch_size: {training_args['per_gpu_eval_batch_size']}")
    logger.info(
        f"per_gpu_predict_batch_size: {training_args['per_gpu_predict_batch_size']}"
    )

    logger.info('-' * 50)
    logger.debug(f"local_dir: {training_args['local_dir']}")
    logger.debug(f"train_file: {training_args['train_file']}")
    logger.debug(f"eval_file: {training_args['eval_file']}")
    logger.debug(f"test_file: {training_args['test_file']}")

    logger.info(f"fold: {training_args['fold']}")
    logger.info(f"num_augements: {training_args['num_augements']}")
    logger.info(f"seg_len: {training_args['seg_len']}")
    logger.info(f"seg_backoff: {training_args['seg_backoff']}")
    logger.info(f"train_rate: {training_args['train_rate']}")
    logger.info(f"train_sample_rate: {training_args['train_sample_rate']}")
    max_train_examples = training_args.get('max_train_examples', 0)
    if max_train_examples > 0:
        logger.warning(f"max_train_examples: {max_train_examples}")
    else:
        logger.debug(f"max_train_examples: {max_train_examples}")
    logger.info(f"random_type: {training_args.get('random_type', None)}")
    confidence = training_args.get('confidence', 0.5)
    logger.info(f"confidence: {confidence}")
    enable_nested_entities = training_args.get('enable_nested_entities', False)
    if enable_nested_entities:
        logger.warning(f"enable_nested_entities: {enable_nested_entities}")
    else:
        logger.info(f"enable_nested_entities: {enable_nested_entities}")

    logger.info('-' * 50)

    enable_fp16 = training_args.get('fp16', None)
    if enable_fp16:
        logger.warning(f"fp16: {enable_fp16}")
    else:
        logger.debug(f"fp16: {enable_fp16}")

    enable_kd = training_args.get('enable_kd', None)
    if enable_kd:
        logger.warning(f"enable_kd: {enable_kd}")
    else:
        logger.debug(f"enable_kd: {enable_kd}")

    enable_sda = training_args.get('enable_sda', None)
    if enable_sda:
        logger.warning(f"enable_sda: {enable_sda}")
        logger.info(f"sda_teachers: {training_args['sda_teachers']}")
        logger.info(f"sda_coeff: {training_args['sda_coeff']:.2f}")
        logger.info(f"sda_decay: {training_args['sda_decay']:.3f}")
    else:
        logger.debug(f"enable_sda: {enable_sda}")

    logger.info('-' * 50)

    ner_type = training_args.get("ner_type", None)
    if ner_type:
        logger.info(f"ner_type: {training_args['ner_type']}")
    logger.info(f"model_type: {training_args['model_type']}")
    logger.info(f"model_path: {training_args['model_path']}")
    logger.info(f"tracking_uri: {training_args['tracking_uri']}")
    logger.info(f"num_labels: {training_args['num_labels']}")
    logger.info(f"seed: {training_args['seed']}")
    logger.info(f"loss_type: {training_args['loss_type']}")
    if training_args['loss_type'] == 'FocalLoss':
        logger.info(f"focalloss_gamma: {training_args['focalloss_gamma']}")
        logger.info(f"focalloss_alpha: {training_args['focalloss_alpha']}")

    if args.detail:
        logger.warning('-' * 50)
        logger.info(
            json.dumps({k: v
                        for k, v in sorted(training_args.items())},
                       ensure_ascii=False,
                       indent=2))


def list_models(args):
    print('-' * 102)
    print("local_id", ' ' * 28, "ctime", ' ' * 15, "model_path")
    print('-' * 102)
    for local_id, model_path, ctime in all_models:
        print(local_id, '    ', ctime, ' ', model_path)


def diff_models(args):
    logger.info(f"{args.local_id}")
    if len(args.local_id) >= 1:

        if len(args.local_id) >= 2:
            src_model_id = args.local_id[0]
            src_model = get_model(src_model_id)
            if src_model is None:
                logger.warning(f"Model {src_model_id} not found.")
                return
            dest_model_id = args.local_id[1]
            dest_model = get_model(dest_model_id)
            if dest_model is None:
                logger.warning(f"Model {dest_model_id} not found.")
                return

            logger.info(f"{[src_model[2], dest_model[2]]}")
        else:
            src_model = None
            dest_model_id = args.local_id[0]
            dest_model = get_model(dest_model_id)
            if dest_model is None:
                logger.warning(f"Model {dest_model_id} not found.")
                return

        #  src_args_path = os.path.join(src_model[1], "best/training_args.json")
        #  if not os.path.exists(src_args_path):
        #      src_args_path = os.path.join(src_model[1], "training_args.json")
        #  dest_args_path = os.path.join(dest_model[1], "best/training_args.json")
        #  if not os.path.exists(dest_args_path):
        #      dest_args_path = os.path.join(dest_model[1], "training_args.json")

        src_args_path = get_model_args_path(src_model)
        dest_args_path = get_model_args_path(dest_model)

        src_args = json.load(open(src_args_path))
        dest_args = json.load(open(dest_args_path))

        for k, v in sorted(src_args.items()):
            if k in dest_args and v == dest_args[k]:
                continue
            logger.debug(f"{k}, {v}")
            logger.debug(f"{k}, {dest_args.get(k, None)}")
            logger.debug('')

        for k, v in sorted(dest_args.items()):
            if k not in src_args:
                logger.debug(f"{k}, {src_args.get(k, None)}")
                logger.debug(f"{k}, {v}")
                logger.debug('')


def new_model(args):
    latest_dir = os.path.join(args.output_dir, "latest")
    import shutil
    if os.path.exists(latest_dir):
        shutil.rmtree(latest_dir)
    os.makedirs(latest_dir)

    import uuid
    local_id = str(uuid.uuid1()).replace('-', '')[:8]
    local_id_file = os.path.join(latest_dir, "local_id")
    with open(local_id_file, 'w') as wt:
        wt.write(f"{local_id}")
    logger.info(f"New model id: {local_id}")


def use_model(args):
    logger.info(f"{args.local_id}")
    if len(args.local_id) >= 1:
        model_id = args.local_id[0]
        model = get_model(model_id)
        if model:
            local_id, model_path, ctime = model

            latest_dir = os.path.join(args.output_dir, "latest")
            import shutil
            if os.path.exists(latest_dir):
                shutil.rmtree(latest_dir)
            shutil.copytree(model_path, latest_dir)
            logger.info(
                f"Use local model({local_id}) {model_path} to {latest_dir}")


def get_dataset_name(args):
    dataset_name = args.dataset_name
    if dataset_name is None:
        dataset_name = os.path.basename(os.path.abspath(os.curdir))
    return dataset_name


def get_dataset_module(dataset_name):
    import importlib
    dataset_module = importlib.import_module(f"{dataset_name}")

    return dataset_module


def export_train_data(args):
    dataset_name = get_dataset_name(args)
    dataset_module = get_dataset_module(dataset_name)

    train_data_generator = None
    if 'train_data_generator' in dataset_module.__dict__:
        train_data_generator = dataset_module.train_data_generator
    ner_labels = None
    if 'ner_labels' in dataset_module.__dict__:
        ner_labels = dataset_module.ner_labels
    ner_connections = None
    if 'ner_connections' in dataset_module.__dict__:
        ner_connections = dataset_module.ner_connections

    #  logger.info(f"ner_labels: {ner_labels}")
    #  logger.info(f"ner_connections: {ner_connections}")
    console.log("[bold cyan]ner_labels:[/bold cyan]", ner_labels)
    console.log("[bold cyan]ner_connections:[/bold cyan]", ner_connections)

    ner_dataset = NerDataset(dataset_name, ner_labels, ner_connections)
    ner_dataset.load(train_data_generator)
    ner_dataset.info()

    export_format = args.format
    if export_format == "json":
        data_file = "tmp.json"
        ner_dataset.save(data_file)
    elif export_format == "brat":
        brat_data_dir = args.brat_data_dir
        if not brat_data_dir:
            brat_data_dir = "brat_data"
        if not os.path.exists(brat_data_dir):
            os.makedirs(brat_data_dir)
        ner_dataset.export_to_brat_files(brat_data_dir, max_pages=50)
    elif export_format == "poplar":
        pass
    else:
        raise Exception(
            f"Bad export format {export_format}. Only available ['json', 'brat', 'poplar']"
        )


def export_test_data(args):
    dataset_name = get_dataset_name(args)
    dataset_module = get_dataset_module(dataset_name)

    test_data_generator = None
    if 'test_data_generator' in dataset_module.__dict__:
        test_data_generator = dataset_module.test_data_generator
    ner_labels = None
    if 'ner_labels' in dataset_module.__dict__:
        ner_labels = dataset_module.ner_labels
    ner_connections = None
    if 'ner_connections' in dataset_module.__dict__:
        ner_connections = dataset_module.ner_connections

    #  logger.info(f"ner_labels: {ner_labels}")
    #  logger.info(f"ner_connections: {ner_connections}")
    console.log("[bold cyan]ner_labels:[/bold cyan]", ner_labels)
    console.log("[bold cyan]ner_connections:[/bold cyan]", ner_connections)

    reviews_file = os.path.join(
        args.output_dir, args.local_id[0],
        f"{dataset_name}_reviews_{args.local_id[0]}.json")

    def reviews_data_generator(reviews_file):
        json_data = json.load(open(reviews_file, 'r'))
        for guid, items in json_data.items():
            text = items['text']
            entities = items['entities']
            tags = []
            for ent in entities:
                tags.append({
                    'category': ent['category'],
                    'start': ent['start'],
                    'mention': ent['mention']
                })
            yield guid, text, None, tags

    ner_dataset = NerDataset(dataset_name, ner_labels, ner_connections)
    ner_dataset.load(reviews_data_generator, reviews_file)
    ner_dataset.info()

    export_format = args.format
    if export_format == "json":
        data_file = "tmp.json"
        ner_dataset.save(data_file)
    elif export_format == "brat":
        brat_data_dir = args.brat_data_dir
        if not brat_data_dir:
            brat_data_dir = "brat_data"
        if not os.path.exists(brat_data_dir):
            os.makedirs(brat_data_dir)
        ner_dataset.export_to_brat_files(brat_data_dir, max_pages=50)
    elif export_format == "poplar":
        pass
    else:
        raise Exception(
            f"Bad export format {export_format}. Only available ['json', 'brat', 'poplar']"
        )


def import_brat_data(args):
    dataset_name = get_dataset_name(args)
    brat_data_dir = args.brat_data_dir
    logger.info(
        f"Loading brat data from {brat_data_dir} into {dataset_name} dataset.")

    ner_dataset = NerDataset(dataset_name)
    ner_dataset.load_from_brat_data(brat_data_dir)
    ner_dataset.save(f"{dataset_name}_from_brat.json")


def import_poplar_data(args):
    pass


def json_to_brat(args):
    dataset_name = args.dataset_name

    dataset_module = get_dataset_module(dataset_name)

    ner_labels = None
    if 'ner_labels' in dataset_module.__dict__:
        ner_labels = dataset_module.ner_labels
    ner_connections = None
    if 'ner_connections' in dataset_module.__dict__:
        ner_connections = dataset_module.ner_connections

    dataset_file = args.dataset_file
    brat_data_dir = args.brat_data_dir

    ner_dataset = NerDataset(dataset_name, ner_labels, ner_connections)
    ner_dataset.load_from_file(dataset_file)
    ner_dataset.info()

    brat_data_dir = args.brat_data_dir
    if not brat_data_dir:
        brat_data_dir = "brat_data"
    if not os.path.exists(brat_data_dir):
        os.makedirs(brat_data_dir)
    ner_dataset.export_to_brat_files(brat_data_dir, max_pages=20)


def brat_to_json(args):
    dataset_file = args.dataset_file
    brat_data_dir = args.brat_data_dir


def main(args):
    find_models(args)

    if args.list:
        list_models(args)
    elif args.diff:
        diff_models(args)
    elif args.show:
        show_model(args)
    elif args.new:
        new_model(args)
    elif args.use:
        use_model(args)
    elif args.export_train_data:
        export_train_data(args)
    elif args.export_test_data:
        export_test_data(args)
    elif args.import_brat_data:
        import_brat_data(args)
    elif args.import_poplar_data:
        import_poplar_data(args)
    elif args.json_to_brat:
        json_to_brat(args)
    elif args.brat_to_json:
        brat_to_json(args)
    else:
        print("Usage: theta [list|diff]")


def get_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--new", action='store_true')
    parser.add_argument("--use", action='store_true')
    parser.add_argument("--diff", action='store_true')
    parser.add_argument("--list", action='store_true')
    parser.add_argument("--show", action='store_true')
    parser.add_argument("--detail", action='store_true')
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--local_id", action='append')

    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--brat_data_dir", default="brat_data")
    parser.add_argument("--format",
                        default='json',
                        choices=['json', 'brat', 'poplar'])
    parser.add_argument("--export_train_data", action='store_true')
    parser.add_argument("--export_test_data", action='store_true')
    parser.add_argument("--import_brat_data", action='store_true')
    parser.add_argument("--import_poplar_data", action='store_true')

    parser.add_argument("--json_to_brat", action='store_true')
    parser.add_argument("--brat_to_json", action='store_true')
    parser.add_argument("--dataset_file", default=None)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    main(args)
