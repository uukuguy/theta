# Theta 

Deep learning toolbox for end-to-end text classification and entity extraction tasks.

Theta是基于深度学习的文本挖掘基础能力工具箱，提供文本分类、文本抽取、文本匹配、阅读理解、知识问答、知识图谱等端到端开箱即用工具集。

## 安装

测试版
```
pip install git+http://122.112.206.124:3000/idleuncle/theta@0.20.0
```
正式版

```
pip install theta==0.20.0
```
## 文本分类任务

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

### 自定义数据生成器

多数情况下，此节是唯一需要自行修改的部分。

根据实际需要处理数据的格式，自定义样本标签集和训练数据、验证数据、测试数据生成器。

样本标签集glue_labels定义分类标签列表，列表项是实际需要输出的标签字符串。

数据生成器需要遵守的基本规范是生成器逐行返回(guid, text_a, text_b, label)四元组。


```
# 样本标签集
labels_file = './data/labels.json'
labels_data = load_json_file(labels_file)
glue_labels = [x['label_desc'] for x in labels_data]
logger.info(f"glue_labels: {len(glue_labels)} {glue_labels}")


def clean_text(text):
    text = text.strip().replace('\n', '')
    text = text.replace('\t', ' ')
    return text


# 训练数据生成器
def train_data_generator(train_file):
    train_data = load_json_file(train_file)
    for i, json_data in enumerate(tqdm(train_data, desc="train")):
        guid = str(i)
        text = json_data['sentence']
        text = clean_text(text)
        label = json_data['label_desc']

        yield guid, text, None, label


# 验证数据生成器
def eval_data_generator(eval_file):
    eval_data = load_json_file(eval_file)
    for i, json_data in enumerate(tqdm(eval_data, desc="eval")):
        guid = str(i)
        text = json_data['sentence']
        text = clean_text(text)
        label = json_data['label_desc']

        yield guid, text, None, label


# 测试数据生成器
def test_data_generator(test_file):
    test_data = load_json_file(test_file)
    total_examples = len(test_data)
    for i, json_data in enumerate(tqdm(test_data, desc="test")):
        guid = str(json_data['id'])
        text = json_data['sentence']
        text = clean_text(text)

        yield guid, text, None, None
```

### 载入数据集

以下代码不需要修改，原样使用即可。

```
# 载入训练数据集
def load_train_examples(train_file):
    train_examples = load_glue_examples(train_data_generator, train_file)
    logger.info(f"Loaded {len(train_examples)} train examples.")

    return train_examples


# 载入验证数据集
def load_eval_examples(eval_file):
    eval_examples = load_glue_examples(eval_data_generator, eval_file)
    logger.info(f"Loaded {len(eval_examples)} eval examples.")

    return eval_examples


# 载入测试数据集
def load_test_examples(test_file):
    test_examples = load_glue_examples(test_data_generator, test_file)
    logger.info(f"Loaded {len(test_examples)} test examples.")

    return test_examples
```

### 自定义模型

Theta提供缺省模型，多数情况下不需要自定义模型。关于自定义模型的详细情况，在进阶文档中说明。

### 自定义训练器

当使用缺省模型时，训练器也是不需要定义的，直接使用AppTrainer=GlueTrainer即可。

通常自定义训练器的目的是通过重载Trainer，获取训练及推理过程的实时数据。

```
class AppTrainer(GlueTrainer):
    def __init__(self, args, glue_labels):
        # 使用自定义模型时，传入build_model参数。
        super(AppTrainer, self).__init__(args, glue_labels, build_model=None)
```

### 主函数

主函数是固定套路，通常不需要修改。 
可以在add_special_args函数中定义自行需要的命令行参数，并在main函数中处理，具体例子见以下do_eda参数。

```
def main(args):

    if args.do_eda:
        from theta.modeling import show_glue_datainfo
        show_glue_datainfo(glue_labels, train_data_generator, args.train_file,
                           test_data_generator, args.test_file)
    else:
        trainer = AppTrainer(args, glue_labels)

        # --------------- Train ---------------
        if args.do_train:
            train_examples = load_train_examples(args.train_file)
            eval_examples = load_eval_examples(args.eval_file)
            trainer.train(args, train_examples, eval_examples)

        # --------------- Evaluate ---------------
        elif args.do_eval:
            eval_examples = load_eval_examples(args.eval_file)
            model = load_model(args)
            trainer.evaluate(args, model, eval_examples)

        # --------------- Predict ---------------
        elif args.do_predict:
            test_examples = load_test_examples(args)
            model = load_model(args)
            trainer.predict(args, model, test_examples)

            save_predict_results(args, trainer.pred_results,
                                 f"./{args.dataset_name}_predict.json",
                                 test_examples)


if __name__ == '__main__':

    def add_special_args(parser):
        parser.add_argument("--do_eda", action="store_true")
        return parser

    from theta.modeling.glue import get_args
    args = get_args([add_special_args])
    main(args)

```
### 启动训练

```
	python run_tnews.py \
		--do_train \
		--model_type bert \
		--model_path /opt/share/pretrained/pytorch/bert-base-chinese 
		--data_dir ./data \
		--output_dir ./output \
		--dataset_name tnews \
		--train_file ./data/train.json
		--learning_rate 2e-5 \
		--train_max_seq_length 160 \
		--per_gpu_train_batch_size 64 \
		--per_gpu_eval_batch_size 64 \
		--num_train_epochs 10 
```

### 启动验证

```
	python run_tnews.py \
		--do_eval \
		--model_type bert \
		--model_path ./output/best \
		--data_dir ./data \
		--output_dir ./output \
		--dataset_name tnews \
		--eval_file ./data/dev.json
		--eval_max_seq_length 160 \
		--per_gpu_eval_batch_size 64 
```

### 启动推理
```
	python run_tnews.py \
		--do_predict \
		--model_type bert \
		--model_path ./output/best \
		--data_dir ./data \
		--output_dir ./output \
		--dataset_name tnews \
		--test_file ./data/test.json
		--eval_max_seq_length 160 \
		--per_gpu_predict_batch_size 64 
```
