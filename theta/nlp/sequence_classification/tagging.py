#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, field


@dataclass
class TaskLabels:
    """
    序列分类任务需要的标签信息结构
    """

    # 分类类别列表
    labels: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for i, label in enumerate(self.labels)}


@dataclass
class TaskTag:
    """
    序列分类任务标注结构
    """

    label: Union[str, List[str]] = None


@dataclass
class TaggedData:
    """
    序列分类样本数据结构
    """

    idx: str = None
    text_a: str = None
    text_b: str = None
    label: TaskTag = None,
    others: Any = None  # 应用侧自行定义的附加标注信息
