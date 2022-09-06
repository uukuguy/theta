#!/usr/bin/env python
# -*- coding: utf-8 -*-


# 同时具备字典和属性访问能力
class DictObject(object):
    __setitem__ = object.__setattr__
    __getitem__ = object.__getattribute__

    def __init__(self, default_dict: dict = None):
        if default_dict:
            for k, v in default_dict.items():
                self.__setattr__(k, v)

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def __str__(self):
        return f"{str(self.__dict__)}"
