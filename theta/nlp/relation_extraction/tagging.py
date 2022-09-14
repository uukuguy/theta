#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, field

from ..tagging import EntityTag

@dataclass
class TaskLabels:
    """
    关系抽取任务需要的标签信息结构
    """

    # 主客实体类别列表
    entity_labels: List[str] = field(default_factory=list)
    # 关系类别列表
    relation_labels: List[str] = field(default_factory=list)
    # 关系类别与对应主客实体类别对的对应表
    relations_map: Dict[str, Tuple[str]] = field(default_factory=dict)

    def __post_init__(self):
        self.entities_label2id = {
            label: i for i, label in enumerate(self.entity_labels)
        }
        self.entities_id2label = {
            i: label for i, label in enumerate(self.entity_labels)
        }
        self.relations_label2id = {
            label: i for i, label in enumerate(self.relation_labels)
        }
        self.relations_id2label = {
            i: label for i, label in enumerate(self.relation_labels)
        }

@dataclass
class SubjectTag(EntityTag):
    """
    关系的主实体结构
    """

    pass


@dataclass
class ObjectTag(EntityTag):
    """
    关系的客实体结构
    """

    pass


@dataclass
class TaskTag:
    """
    关系抽取任务标注结构
    """

    s: SubjectTag = None  # subject: 关系的主实体
    p: str = None  # predicate: 关系类别
    o: ObjectTag = None  # object: 关系的客实体

    def to_json(self):
        return {
            "subject": self.s.to_json(),
            "predicate": self.p,
            "object": self.o.to_json(),
        }

    def from_json(self, json_data):
        self.s, self.p, self.o = (
            SubjectTag().from_json(json_data["subject"]),
            json_data["predicate"],
            ObjectTag().from_json(json_data["object"]),
        )

        return self

    @property
    def subject(self):
        return self.s

    @property
    def predicate(self):
        return self.p

    @property
    def object(self):
        return self.o
    
    @subject.setter
    def subject(self, v):
        self.s = v

    @predicate.setter
    def predicate(self, v):
        self.p = v

    @object.setter
    def object(self, v):
        self.o = v


@dataclass
class TaggedData:
    """
    关系标注样本数据结构
    """

    idx: str = ""
    text: str = ""
    tags: List[TaskTag] = field(default_factory=list)
    metadata: Any = None  # 应用侧自行定义的附加标注信息

    def to_json(self):
        return {
            'idx': self.idx,
            'text': self.text,
            'tags': [ tag.to_json() for tag in self.tags],
            'metadata': self.metadata
        }

    def from_json(self, json_data):
        self.idx = json_data['idx']
        self.text = json_data['text']
        self.tags = [ TaskTag().from_json(tag_data) for tag_data in json_data['tags']]
        self.metadata = json_data.get('metadata', None)