# Theta



##任务目标

brat中正确显示kgcs实体标注文件。

##安装



```bash
pip install theta
```

在.bashrc中加入以下命令：

```bash
alias tt='python -m theta'
```

重新登录shell。



## 命令行支持



tt --list

列出当前目录下的所有已构建模型。



tt --new

开始构建一个新模型。



tt --show --local_id <model_id>

显示指定模型的参数设置。



tt --diff --local_id <model1> --local_id <mode2>

比较两个模型之间参数设置差异。



tt --export_train_data --dataset <dataset_name> --format [json|brat|poplar] 

导入当前模型训练数据集数据，格式支持json, brat, poplar。

dataset参数未指定时，使用当前目录名为数据集名。当前目录下必须有<dataset_name>.py，其中定义了train_data_generator, test_data_generator函数。



tt --export_test_data --dataset <dataset_name> --format [json|brat|poplar]

导入当前模型测试数据集数据，格式支持json, brat, poplar。





## 数据探索

跟踪查看数据集的变化细节，掌握数据集在流水线各环节的表现形态。

### NerDataset





