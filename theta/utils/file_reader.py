#!/usr/bin/env python
# -*- coding: utf-8 -*-


def tsv_line_reader(fd, sep='\t', quotechar=None):

    def gen():
        for i in fd:
            yield i.rstrip('\n').split(sep)

    return gen()


def tsv_reader(csv_file, headers=None, sep='\t', quotechar=None):
    """Reads a tab separated value file."""
    with open(csv_file, 'r', encoding='utf8') as fd:
        reader = tsv_line_reader(fd, sep=sep, quotechar=quotechar)
        if headers is None:
            headers = next(reader)
        Example = namedtuple('Example', headers)

        for line in reader:
            example = Example(*line)
            yield example


def csv_reader(fd, quotechar=None):
    return tsv_reader(fd, sep=' ', quotechar=quotechar)


# examples = list(tqdm(csv_reader(cvs_file, delimiter='\t')))

# --------------------------------
import json
from collections import namedtuple


def json_line_generator(json_file):

    def gen():
        with open(json_file, 'r', encoding='utf8') as fd:
            while True:
                line = fd.readline()
                if line:
                    json_data = json.loads(line)
                    yield json_data
                else:
                    break

    return gen()


def dict_to_namedtuple(dict_data, name="Example"):
    keys = dict_data.keys()
    values = list(dict_data.values())
    keys = [k.replace('@', '') for k in keys]
    Example = namedtuple(name, keys)
    for i, c in enumerate(Example._fields):
        v = values[i]
        if isinstance(v, dict):
            sub_example = dict_to_namedtuple(v, name=c.upper())
            values[i] = sub_example
        elif isinstance(v, list):
            for j, sub_v in enumerate(v):
                if isinstance(sub_v, dict):
                    v[j] = dict_to_namedtuple(sub_v, name=c.upper())

    example = Example(*values)

    return example


def jsonl_reader(json_file):
    reader = json_line_generator(json_file)
    for jd in reader:
        assert isinstance(jd, dict)
        example = dict_to_namedtuple(jd)
        yield example


def json_reader(json_file):
    json_data = json.load(open(json_file))
    for jd in json_data:
        assert isinstance(jd, dict)
        example = dict_to_namedtuple(jd)
        yield example


# examples = list(tqdm(json_reader(jsonl_file)))
