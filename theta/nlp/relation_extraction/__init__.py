
from .tagging import (
    TaskLabels,
    TaggedData,
    TaskTag,
    SubjectTag,
    ObjectTag
)

# 缺省采用 gplinker 关系抽取方法
from .gplinker import TaskDataset, Model, Evaluator