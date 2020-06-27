#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, glob, json, datetime
from tqdm import tqdm
from loguru import logger

all_models = []


def get_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--new", action='store_true')
    parser.add_argument("--use", action='store_true')
    parser.add_argument("--diff", action='store_true')
    parser.add_argument("--list", action='store_true')
    parser.add_argument("--show", action='store_true')
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--local_id", action='append')

    args = parser.parse_args()

    return args


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


def show_model(args):
    logger.info(f"{args.local_id}")
    for model_id in args.local_id:
        model = get_model(model_id)
        args_path = os.path.join(model[1], "best/training_args.json")
        training_args = json.load(open(args_path))
        logger.warning(f"----- {model_id} -----")
        logger.info(f'{model[2]}')
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
        src_args_path = os.path.join(src_model[1], "best/training_args.json")
        dest_args_path = os.path.join(dest_model[1], "best/training_args.json")

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
    import uuid
    local_id = str(uuid.uuid1()).replace('-', '')
    local_id_file = os.path.join(args.output_dir, "latest/local_id")
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
    else:
        print("Usage: theta [list|diff]")


if __name__ == '__main__':
    args = get_args()
    main(args)
