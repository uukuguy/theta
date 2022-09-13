#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json
import numpy as np
import gc
import collections

script_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

class TrainingDataset(object):
    """预训练数据集生成器"""

    def __init__(self, texts, tokenizer, sequence_length=512):
        """参数说明：
        tokenizer必须是bert4keras自带的tokenizer类；
        """
        self.texts = texts
        self.map_features = {}
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.token_pad_id = tokenizer._token_pad_id
        self.token_cls_id = tokenizer._token_start_id
        self.token_sep_id = tokenizer._token_end_id
        self.token_mask_id = tokenizer._token_mask_id
        self.vocab_size = tokenizer._vocab_size

        self.process_texts()

    def padding(self, sequence, padding_value=None):
        """对单个序列进行补0"""
        if padding_value is None:
            padding_value = self.token_pad_id

        sequence = sequence[: self.sequence_length]
        padding_length = self.sequence_length - len(sequence)
        return sequence + [padding_value] * padding_length

    def sentence_process(self, text):
        """单个文本的处理函数，返回处理后的instance"""
        raise NotImplementedError

    def paragraph_process(self, texts, starts, ends, paddings):
        """单个段落（多个文本）的处理函数
        说明：texts是单句组成的list；starts是每个instance的起始id；
              ends是每个instance的终止id；paddings是每个instance的填充id。
        做法：不断塞句子，直到长度最接近sequence_length，然后padding。
        """
        instances, instance = [], [[start] for start in starts]

        for text in texts:
            # 处理单个句子
            sub_instance = self.sentence_process(text)
            sub_instance = [i[: self.sequence_length - 2] for i in sub_instance]
            new_length = len(instance[0]) + len(sub_instance[0])

            # 如果长度即将溢出
            if new_length > self.sequence_length - 1:
                # 插入终止符，并padding
                complete_instance = []
                for item, end, pad in zip(instance, ends, paddings):
                    item.append(end)
                    item = self.padding(item, pad)
                    complete_instance.append(item)
                # 存储结果，并构建新样本
                instances.append(complete_instance)
                instance = [[start] for start in starts]

            # 样本续接
            for item, sub_item in zip(instance, sub_instance):
                item.extend(sub_item)

        # 插入终止符，并padding
        complete_instance = []
        for item, end, pad in zip(instance, ends, paddings):
            item.append(end)
            item = self.padding(item, pad)
            complete_instance.append(item)

        # 存储最后的instance
        instances.append(complete_instance)

        return instances

    def process_texts(self):
        self.map_features = {}
        count = 0
        for texts in self.texts:
            instances = self.paragraph_process(texts)
            for instance in instances:
                input_ids, masked_lm_labels = instance[0], instance[1]
                assert len(input_ids) <= self.sequence_length
                features = collections.OrderedDict()
                features["input_ids"] = input_ids
                features["masked_lm_labels"] = masked_lm_labels
                self.map_features[str(count)] = features
                count += 1

    def serialize(self, instances, db, count):
        """写入到文件"""
        for instance in instances:
            input_ids, masked_lm_labels = instance[0], instance[1]
            assert len(input_ids) <= self.sequence_length
            features = collections.OrderedDict()
            features["input_ids"] = input_ids
            features["masked_lm_labels"] = masked_lm_labels
            db[str(count)] = features
            count += 1
        return count

    def process_to_file(self, corpus, record_name):
        """处理输入语料（corpus）"""
        count = 0

        import shelve
        db = shelve.open(record_name)
        for texts in corpus:
            instances = self.paragraph_process(texts)
            count = self.serialize(instances, db, count)

        db.close()
        del instances
        gc.collect()

        # 记录对应的文件名和样本量
        record_info = {"filename": record_name, "samples_num": count}
        json.dump(record_info, open(record_name + ".json", "w", encoding="utf-8"))

        print("write %s examples into %s" % (count, record_name))


class TrainingDatasetRoBERTa(TrainingDataset):
    """预训练数据集生成器（RoBERTa模式）"""

    def __init__(self, texts, tokenizer, word_segment, mask_rate=0.15, sequence_length=512):
        """参数说明：
        tokenizer必须是bert4torch自带的tokenizer类；
        word_segment是任意分词函数。
        """
        self.word_segment = word_segment
        self.mask_rate = mask_rate
        super(TrainingDatasetRoBERTa, self).__init__(texts, tokenizer, sequence_length)

    def token_process(self, token_id):
        """以80%的几率替换为[MASK]，以10%的几率保持不变，
        以10%的几率替换为一个随机token。
        """
        rand = np.random.random()
        if rand <= 0.8:
            return self.token_mask_id
        elif rand <= 0.9:
            return token_id
        else:
            return np.random.randint(0, self.vocab_size)

    def sentence_process(self, text):
        """单个文本的处理函数
        流程：分词，然后转id，按照mask_rate构建全词mask的序列, 来指定哪些token是否要被mask
        """
        words = self.word_segment(text)
        rands = np.random.random(len(words))

        token_ids, mask_ids = [], []
        for rand, word in zip(rands, words):
            word_tokens = self.tokenizer.tokenize(text=word)[1:-1]
            word_token_ids = self.tokenizer.tokens_to_ids(word_tokens)

            if rand < self.mask_rate:
                word_mask_ids = [self.token_process(i) for i in word_token_ids]
                token_ids.extend(word_mask_ids)
                mask_ids.extend(word_token_ids)
            else:
                token_ids.extend(word_token_ids)
                word_mask_ids = [0] * len(word_tokens)
                mask_ids.extend(word_mask_ids)

        return [token_ids, mask_ids]

    def paragraph_process(self, texts):
        """给原方法补上starts、ends、paddings"""
        starts = [self.token_cls_id, 0]
        ends = [self.token_sep_id, 0]
        paddings = [self.token_pad_id, 0]
        return super(TrainingDatasetRoBERTa, self).paragraph_process(
            texts, starts, ends, paddings
        )