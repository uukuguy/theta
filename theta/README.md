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

所有工作围绕着数据集对象(NerDataset)展开，

```python
dataset_name = "ner_task"
ner_labels = ["人物", "机构", "地点"]
ner_connections = []
```

#### Q&A

1. ***如何构建数据集对象？***

   * 标准格式文件载入方式构建NerDataset数据集

   ```python
   from theta.modeling import NerDataset
   ner_dataset = NerDataset(dataset_name, \
                            ner_labels=ner_labels, \
                            ner_connections=ner_connections)
   ner_dataset.load_from_file(data_filename)
   ```

   

   * 数据生成器方式构建NerDataset数据集

   ```python
   from theta.modeling import NerDataset, TaggedText, EntityTag
   ner_dataset = NerDataset(dataset_name, \
                            ner_labels=ner_labels, \
                           ner_connections=ner_connections)
   # 数据生成器要求逐行返回(guid, text, None, tags)，其中tags列表项格式为{'category':c, 'start': s, 'mention':m}。
   for guid, text, _, tags in data_generator:
     tagged_text = TaggedText(guid, text)
     for tag in tags:
       c = tag['category']
       s = tag['start']
       m = tag['mention']
       tagged_text.add_tag(EntityTag(category=c, start=s, mention=m))
     ner_dataset.append(tagged_text)
   ```

   

   * 编程方式构建NerDataset数据集

   ```python
   from theta.modeling import NerDataset, TaggedText, EntityTag
   ner_dataset = NerDataset(dataset_name, ner_labels=ner_labels, \
                            ner_connections=ner_connections)
   
   tagged_text = TaggedText(guid, text)
   tagged_text.add_tag(EntityTag(category='机构', start="0", mention="国务院"))
   
   ner_dataset.append(tagged_text)
   ```

   
   

2. ***如何遍历NerDataset数据集?***

   * 直接载入标准格式数据文件，完成遍历处理。


   ```python
   from theta.modeling import ner_data_generator
   for guid, text, _, tags in ner_data_generator( \
   																data_filename, \ 		
   																dataset_name=dataset_name, \
   																ner_labels=ner_labels, \
   																ner_connections=ner_connections):
   		logger.info(f"guid: {guid}, text: {text}, tags: {tags}")
   
   ```

   * 遍历已构造NerDataset数据集

   ```python
   for tagged_text in tqdm(ner_dataset):
     guid = tagged_text.guid
     text = tagged_text.text
     tags = [x.to_dict() for x in tagged_text.tags]
   	logger.info(f"guid: {guid}, text: {text}, tags: {tags}")
   ```

   

3. ***如何持久化NerDataset数据集？***

   ```
   # 保存为NerDataset标准格式
   ner_dataset.save(ner_data_file)
   
   # 导出为brat标注格式
   ner_dataset.export_to_brat(brat_data_dir, max_pages=10)
   
   # 导出为poplar标注格式
   ner_dataset.export_to_poplar(poplar_file, max_pages=100, start_page=0)
   ```

   

4. ***如何导入其它标注格式文件？***


   ```
   # 导入brat标注式格式
   ner_dataset.import_from_brat(brat_data_dir)
   
   # 导入poplar标注格式
   ner_dataset.import_from_poplar(poplar_file)
   ```

5. ***NerDataset数据集的典型操作***

   * 比较两个数据集
     返回自身仅有、比较对象仅有、两者相同共三个NerDataset对象。


   ```
   from theta.modeling import NerDataset
   self_only_dataset, other_only_dataset, same_dataset  = ner_dataset.diff(another_ner_dataset)
   ```

   * 合并多个数据集
     按最少重复次数保留合并多个数据集

   ```
   from theta.modeling import ner_merge_datasets
   merged_ner_dataset = ner_merge_datasets(ner_dataset_list, min_dups=2)
   ```

   * 两个数据集标注重叠区域
     返回每条数据的标注重叠区域，[[(c, s, m), (c, s, m) ...], [(c, s, m), (c, s, m), ...], ...]

   ```
   overlap_list = ner_dataset.overlap(another_ner_dataset)
   ```

   

6. NerDataset数据集统计信息

   ```
   ner_dataset.info()
   ```

   

7. NerDataset数据集标注可视化
   方法一：导出brat, poplar等标注格式，在相应标注工具中查看
   方法二：

8. 











