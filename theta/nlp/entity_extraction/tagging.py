#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, field


@dataclass
class TaskLabels:
    """
    实体抽取任务需要的标签信息结构
    """

    # 实体类别列表
    entity_labels: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.entities_label2id = {
            label: i for i, label in enumerate(self.entity_labels)
        }
        self.entities_id2label = {
            i: label for i, label in enumerate(self.entity_labels)
        }


@dataclass
class EntityTag:
    """
    实体标注结构
    """

    c: str = None  # category: 实体类别
    s: int = -1  # start: 文本片断的起始位置
    m: str = None  # mention: 文本片断内容

    def to_json(self):
        return {"category": self.c, "start": self.s, "mention": self.m}

    def from_json(self, json_data):
        self.c, self.s, self.m = (
            json_data["category"],
            json_data["start"],
            json_data["mention"],
        )

        return self


@dataclass
class TaskTag(EntityTag):
    """
    实体抽取任务标注结构
    """

    pass


@dataclass
class TaggedData:
    """
    实体标注样本数据结构
    """

    idx: str = ""
    text: str = ""
    tags: List[TaskTag] = field(default_factory=list)
    others: Any = None  # 应用侧自行定义的附加标注信息
