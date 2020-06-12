#!/usr/bin/env python
# -*- coding: utf-8 -*-

from loguru import logger
from ..common_args import get_main_args


def add_modeling_args(parser):
    parser.add_argument("--soft_label", action="store_true", help="Soft label for ner span")
    return parser


def get_args(special_args: list = None, ):
    return get_main_args(add_modeling_args, special_args)
