#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random, json
from typing import Callable, List, Tuple, Type, Union
from copy import deepcopy

from loguru import logger
from tqdm import tqdm


def is_uniform(entities):
    for i, x in enumerate(entities):
        for x1 in entities[i:]:
            if x != x1:
                return False
    return True


def remove_duplicate_entities(R):
    new_R = []
    for i, r in enumerate(R):
        found_dup = False
        for r1 in R[i + 1:]:
            if r == r1:
                found_dup = True
                break
        if found_dup:
            continue
        new_R.append(r)
    return new_R


def merge_entities(entities_list, key=None, min_dups=2):
    new_X = []
    X = [x for z in entities_list for x in z]
    #  logger.info(f"X: {X}")
    if key:
        X = sorted(X, key=key)

    for i, x in enumerate(X[:1 - min_dups]):
        if is_uniform(X[i:i + min_dups]):
            new_X.append(deepcopy(x))
    new_X = remove_duplicate_entities(new_X)
    return new_X


def merge_ner_tags(ner_tags_list, min_dups=2):
    if len(ner_tags_list) == 0:
        return []
    from copy import deepcopy
    merged_tags = deepcopy(ner_tags_list[0])
    if len(ner_tags_list) > 1:
        for i, tags_list in enumerate(
                tqdm(zip(*ner_tags_list), desc="Merge ner tags")):
            new_tags = merge_entities(
                tags_list,
                key=lambda x: x['start'] * 1000 + len(x['mention']),
                min_dups=min_dups)
            merged_tags[i] = new_tags

    return merged_tags


class BaseSamples:
    def __init__(self,
                 data_file: str = None,
                 data_generator: Callable = None,
                 data_list: List = None,
                 labels_list: List = None):
        self.data_list = data_list if data_list is not None else []
        self.labels_list = labels_list if labels_list is not None else []
        self.data_generator = data_generator
        self.data_file = data_file

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int):
        return self.data_list[idx]

    def _new_instance(self, **kwargs):
        raise NotImplementedError

    def shuffle(self, random_state: int = None):
        if random_state is not None:
            random.seed(random_state)
        random.shuffle(self.data_list)

        return self

    @property
    def label2id(self):
        return {x: i for i, x in enumerate(self.labels_list)}

    @property
    def id2label(self):
        return {i: x for i, x in enumerate(self.labels_list)}

    @property
    def rows(self):
        def _rows():
            for data in self.data_list:
                yield data

        return _rows

    def load_samples(self):
        logger.warning(f"self.data_generator: {self.data_generator}")
        logger.warning(f"self.data_file: {self.data_file}")
        if self.data_generator is not None:
            self.load_samples_from_generator(self.data_generator)
        elif self.data_file is not None:
            self.load_samples_from_file(self.data_file)
        else:
            logger.warning(
                f"Unknow how to load samples. data_generator: {self.data_generator}, data_file: {self.data_file}"
            )

    def load_samples_from_generator(self, data_generator):
        self.data_list = []
        for x in data_generator():
            self.data_list.append(x)

    def load_samples_from_file(self, data_file):
        self.data_list = []
        raise NotImplementedError

    def split(self,
              ratios: Union[float, List, Tuple] = None,
              random_state=None):
        if type(ratios) == float:
            train_rate = ratios
            eval_rate = 1.0 - train_rate
            test_rate = 0.0
        elif type(ratios) == list or type(ratios) == tuple:
            if len(ratios) == 2:
                train_rate, eval_rate = ratios
                test_rate = 1.0 - train_rate - eval_rate
            elif len(ratios) == 3:
                train_rate, eval_rate, test_rate = ratios
        else:
            raise ValueError(f"ratios: {ratios} must be float or list.")

        num_total_samples = self.__len__()
        num_train_samples = int(num_total_samples * train_rate)
        if test_rate > 0.0:
            num_eval_samples = int(num_total_samples * eval_rate)
            num_test_samples = num_total_samples - num_train_samples - num_eval_samples
        else:
            num_eval_samples = num_total_samples - num_train_samples
            num_test_samples = 0

        if random_state is not None:
            from sklearn.model_selection import train_test_split
            train_data_list, eval_data_list = train_test_split(
                self.data_list,
                train_size=train_rate,
                random_state=random_state)
            train_samples = self._new_instance(data_list=train_data_list,
                                               labels_list=self.labels_list)
            eval_samples = self._new_instance(data_list=eval_data_list,
                                              labels_list=self.labels_list)

            return train_samples, eval_samples
        else:
            train_samples = self._new_instance(
                data_list=self.data_list[:num_train_samples],
                labels_list=self.labels_list)
            eval_samples = self._new_instance(
                data_list=self.data_list[num_train_samples:num_train_samples +
                                         num_eval_samples],
                labels_list=self.labels_list)
            if num_test_samples > 0:
                test_samples = self._new_instance(
                    data_list=self.data_list[-num_test_samples:],
                    labels_list=self.labels_list)
                return train_samples, eval_samples, test_samples
            else:
                return train_samples, eval_samples


def check_labels(glue_labels, labels_list):
    unknown_labels = set(labels_list) - set(glue_labels)
    if any(unknown_labels):
        logger.warning(
            f"Labels {unknown_labels} not in glue_labels: {glue_labels}")
    notexist_labels = set(glue_labels) - set(labels_list)
    if any(notexist_labels):
        logger.warning(
            f"Labels {notexist_labels} not in labels_list: {set(labels_list)}")


class GlueSamples(BaseSamples):
    def __init__(self, **kwargs):
        super(GlueSamples, self).__init__(**kwargs)

    def _new_instance(self, **kwargs):
        return GlueSamples(**kwargs)

    #  @classmethod
    #  def from_generator(cls,
    #                     data_generator: Callable,
    #                     glue_labels,
    #                     data_source: str = None):
    #      samples = GlueSamples(data_generator=data_generator,
    #                            labels_list=glue_labels)
    #      for sid, text_a, text_b, label in data_generator(data_source):
    #          samples.data_list.append((sid, text_a, text_b, label))
    #
    #      return samples


class EntitySamples(BaseSamples):
    def __init__(self, **kwargs):
        super(EntitySamples, self).__init__(**kwargs)

    def _new_instance(self, **kwargs):
        return EntitySamples(**kwargs)

    @property
    def label2id(self):
        return {x: i + 1 for i, x in enumerate(self.labels_list)}

    @property
    def id2label(self):
        return {i + 1: x for i, x in enumerate(self.labels_list)}

    @property
    def tags(self):
        all_tags = []
        for data in self.data_list:
            all_tags.append(data['tags'])
        return all_tags

    def load_samples_from_file(self, data_file):
        self.data_file = data_file
        self.data_list = []
        try:
            json_data = json.load(open(data_file, 'r'))
        except:
            json_data = []
            for line in tqdm(open(data_file, 'r').readlines(), desc="jsonl"):
                data = json.loads(line.strip())
                json_data.append(data)

        for data in tqdm(json_data, desc="Checking"):
            assert isinstance(data, dict)
            assert 'idx' in data, f"data: {data}"
            assert 'text' in data, f"data: {data}"
            assert 'tags' in data, f"data: {data}"
            tags = data['tags']
            for tag in tags:
                assert isinstance(tag, dict)
                assert 'category' in tag, f"data: {data}"
                assert 'start' in tag, f"data: {data}"
                assert 'mention' in tag, f"data: {data}"
        self.data_list = json_data

    @classmethod
    def merge_tags(self, samples_list, min_dups=2):
        if len(samples_list) == 0:
            return None
        if len(samples_list) == 1:
            return samples_list[0]
        lens = [len(samples) for samples in samples_list]
        assert int(sum(lens) / len(lens)) == lens[0]

        ner_tags_list = [samples.tags for samples in samples_list]
        merged_tags = merge_ner_tags(ner_tags_list, min_dups=min_dups)

        new_data_list = []
        for data, tags in zip(samples_list[0], merged_tags):
            new_data = deepcopy(data)
            new_data['tags'] = tags
            new_data_list.append(new_data)
        labels_list = deepcopy(samples_list[0].labels_list)

        return EntitySamples(data_list=new_data_list, labels_list=labels_list)


class SPOSamples(BaseSamples):
    def __init__(self, **kwargs):
        super(SPOSamples, self).__init__(**kwargs)

    def _new_instance(self, **kwargs):
        return SPOSamples(**kwargs)

    #  @classmethod
    #  def from_generator(cls,
    #                     data_generator: Callable,
    #                     ner_labels,
    #                     data_source: str = None):
    #      samples = SPOSamples(labels_list=ner_labels)
    #      categories = []
    #      for sid, text, _, json_tags in data_generator(data_source):
    #          """
    #          json_tag: {'category': 'Person', 'start': 0, 'mention': 'name'}
    #          """
    #          samples.data_list.append((sid, text, json_tags))
    #
    #          for tag in json_tags:
    #              c = tag['category']
    #              if c not in categories:
    #                  categories.append(c)
    #      categories = sorted(list(set(categories)))
    #      logger.info(f"Loaded categories: {categories}")
    #
    #      if not samples.labels_list:
    #          samples.labels_list = categories
    #
    #      return samples
