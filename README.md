# Theta 

Deep learning toolbox for end-to-end text information extraction tasks.

Theta定位是解决实际工程项目中文本信息抽取任务的实用工具箱，端到端实现从原始文本输入到结构化输出全过程。用户工作聚焦于输入数据格式转换，调整关键参数调度theta完成模型训练推理任务及输出格式化数据利用。

Theta应用场景包括国家级重点企业非结构化数据挖掘利用、开放域文本数据结构化抽取、各大在线实体关系抽取类评测赛事等。

Theta性能指标要求达到业内主流头部水准，近期参加了包括CCF2019、CHIP2019、CCKS2020、CCL2020等C字头顶级赛事，目前取得10余次决赛奖项，包括7次前三，2次第一。


## 更新

- 2022.09.06 0.50.0

  nlp.entity_extraction, nlp.relation_extraction

  

## 安装

测试版
```
pip install git+http://github.com/idleuncle/theta.git
```
正式版

```
pip install -U theta
```



## CLUE-CLUENER 细粒度命名实体识别

本数据是在清华大学开源的文本分类数据集THUCTC基础上，选出部分数据进行细粒度命名实体标注，原数据来源于Sina News RSS.

训练集：10748 验证集：1343

标签类别： 数据分为10个标签类别，分别为: 地址（address），书名（book），公司（company），游戏（game），政府（goverment），电影（movie），姓名（name），组织机构（organization），职位（position），景点（scene）

数据下载地址：https://github.com/CLUEbenchmark/CLUENER2020

排行榜地址：https://cluebenchmarks.com/ner.html

完整代码见theta/examples/CLUENER：[cluener.ipynb](theta/examples/CLUENER/cluener.ipynb)

选用bert-base-chinese预训练模型，CLUE测评F1得分77.160。

```
# 训练
make -f Makefile.cluener train

# 推理
make -f Makefile.cluener predict

# 生成提交结果文件
make -f Makefile.cluener submission
```

## CLUE-TNEWS 今日头条中文新闻（短文）分类任务

以下样例是CLUE（[中文任务基准测评](https://cluebenchmarks.com/index.html)）中今日头条中文新闻（短文）分类任务。

数据集来自今日头条的新闻版块，共提取了15个类别的新闻，包括旅游，教育，金融，军事等。

数据量：训练集(53,360)，验证集(10,000)，测试集(10,000)

> 例子：
> {"label": "102", "label_desc": "news_entertainment", "sentence": "江疏影甜甜圈自拍，迷之角度竟这么好看，美吸引一切事物"}
> 每一条数据有三个属性，从前往后分别是 分类ID，分类名称，新闻字符串（仅含标题）。

选用bert-base-chinese预训练模型，CLUE测评F1得分56.100。

完整代码见theta/examples/TNEWS：[tnews.ipynb](theta/examples/TNEWS/tnews.ipynb)

[TNEWS数据集下载](https://storage.googleapis.com/cluebenchmark/tasks/tnews_public.zip)

### 导入基础库

```
import json
from tqdm import tqdm
from loguru import logger
import numpy as np

from theta.modeling import load_glue_examples
from theta.modeling.glue import GlueTrainer, load_model, get_args
from theta.utils import load_json_file
```


