#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, time
from pathlib import Path
from loguru import logger


def add_common_args(parser):

    # --------------- Command arguments ---------------
    parser.add_argument("--do_train",
                        action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict",
                        action="store_true",
                        help="Whether to run predictions on the test set.")
    parser.add_argument("--resume_train",
                        action="store_true",
                        help="Continue to train model.")

    # --------------- Main arguments ---------------

    parser.add_argument(
        "--seed",
        type=int,
        default=8864,
        help="Random seed.",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        #  required=True,
        help=
        "The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        #  required=True,
        help="The output dir.",
    )
    parser.add_argument(
        "--submissions_dir",
        type=str,
        default="./submissions",
        #  required=True,
        help="The submissions dir.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The cache dir.",
    )

    parser.add_argument(
        "--model_type",
        default="bert",
        type=str,
        help="Model type selected in the list: bert, xlnet ",
    )

    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        help="The pretrained model path.",
    )
    # --------------- Data arguments ---------------
    parser.add_argument(
        "--train_max_seq_length",
        default=256,
        type=int,
        help=
        "The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--eval_max_seq_length",
        default=256,
        type=int,
        help=
        "The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument("--per_gpu_train_batch_size",
                        default=8,
                        type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size",
                        default=8,
                        type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--per_gpu_predict_batch_size",
                        default=1,
                        type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay",
                        default=0.0,
                        type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs",
                        default=3,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help=
        "If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )

    parser.add_argument("--warmup_steps",
                        default=0,
                        type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_rate",
                        default=0.1,
                        type=float,
                        help="Linear warmup rate of total steps.")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help=
        "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help=
        "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--save_checkpoints", action="store_true", help="")
    parser.add_argument("--cache_features", action="store_true", help="")

    # ------------------------------
    parser.add_argument("--train_file",
                        default="train.bios",
                        type=str,
                        help="Train file under data dir.")
    parser.add_argument("--eval_file",
                        default="eval.bios",
                        type=str,
                        help="Eval file under data dir.")
    parser.add_argument("--test_file",
                        default="test.bios",
                        type=str,
                        help="Test file under data dir.")
    parser.add_argument("--experiment_name",
                        type=str,
                        default=None,
                        help="The name of experiment.")
    parser.add_argument("--task_name",
                        type=str,
                        default=None,
                        help="The name of task.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        #  required=True,
        help="Dataset name for cached filename.")
    parser.add_argument("--train_rate",
                        default=0.8,
                        type=float,
                        help="train and eval rate.")
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold used.",
    )

    parser.add_argument("--train_sample_rate",
                        default=1.0,
                        type=float,
                        help="train sample rate.")

    # ------------------------------
    # Knowledge distillation
    parser.add_argument("--enable_kd",
                        action='store_true',
                        help="Whether to do knowledge distillation (KD).")
    parser.add_argument("--kd_coeff",
                        type=float,
                        default=1.0,
                        help="KD loss coefficient.")
    parser.add_argument("--kd_decay", type=float, default=0.995, help="The exponential decay of KD.")

    # ------------------------------
    parser.add_argument("--server_ip",
                        type=str,
                        default="",
                        help="For distant debugging.")
    parser.add_argument("--server_port",
                        type=str,
                        default="",
                        help="For distant debugging.")
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Whether to run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.")

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output files.",
    )
    parser.add_argument(
        "--no_eval_on_each_epoch",
        action="store_true",
        help="No evaluate on each epoch.",
    )

    parser.add_argument(
        "--num_labels",
        default=2,
        type=int,
        help="Number of labels in dataset.",
    )
    parser.add_argument("--logging_steps",
                        type=int,
                        default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps",
                        type=int,
                        default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help=
        "Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument(
        '--predict_all_checkpoints',
        action="store_true",
        help=
        "Predict all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda",
                        action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir",
                        action="store_true",
                        help="Overwrite the content of the output directory")
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seg_len", type=int, default=256, help="")
    parser.add_argument("--seg_backoff", type=int, default=64, help="")
    parser.add_argument("--max_span_len", type=int, default=32, help="")
    parser.add_argument("--num_augements", type=int, default=0, help="")
    parser.add_argument(
        "--loss_type",
        type=str,
        default="CrossEntropyLoss",
        help=
        "Loss type: ['CrossEntropyLoss', 'FocalLoss', 'LabelSmoothingCrossEntropy', 'CircleLoss']"
    )
    parser.add_argument("--focalloss_gamma", type=float, default=2.0)
    parser.add_argument("--focalloss_alpha", type=float, default=None)
    parser.add_argument("--allow_overlap", action="store_true", help="")

    return parser


def get_main_args(
    add_modeling_args,
    special_args: list = None,
):
    import argparse
    parser = argparse.ArgumentParser()

    parser = add_common_args(parser)
    parser = add_modeling_args(parser)

    if special_args:
        for sa in special_args:
            parser = sa(parser)

    args = parser.parse_args()

    if args.task_name is None or len(args.task_name) == 0:
        basename = os.path.basename(sys.argv[0])
        taskname = basename[:basename.rfind('.')]
        p0 = taskname.find('_')
        if p0 >= 0:
            taskname = taskname[p0 + 1:]
        args.task_name = taskname

    if args.experiment_name is None or len(args.experiment_name) == 0:
        t = time.localtime()
        args.experiment_name = f"exp-{args.task_name}-" \
            f"{t.tm_year:04d}{t.tm_mon:02d}{t.tm_mday:02d}" \
            f"{t.tm_hour:02d}{t.tm_min:02d}{t.tm_sec:02d}"

    logname = args.task_name
    logger.add(Path(args.output_dir) / f"{logname}.log")

    return args
