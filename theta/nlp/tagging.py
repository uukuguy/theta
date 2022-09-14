from dataclasses import dataclass

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

    @property
    def category(self):
        return self.c

    @property
    def start(self):
        return self.s

    @property
    def mention(self):
        return self.m

    @category.setter
    def category(self, v):
        self.c = v

    @start.setter
    def start(self, v):
        self.s = v

    @mention.setter
    def mention(self, v):
        self.m = v