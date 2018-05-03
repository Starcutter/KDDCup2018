# KDDCup2018

|folders|contents|
|---|---|
|sandbox|尝试进行的对数据的观察学习以及实验等，主要是 jupyter notebooks。|
|data|训练数据以及提交样例|
|networks|神经网络初步模型|

### 自动预测，生成CSV:

```
python sandbox/genCSV.py <random_state>
```

`<random_state>`为可选参数，整形，随机种子。不填则默认为0.

将在`results/`文件夹中生成名称为`submit_<%m%d>_<random_state>.csv`的提交文件

### 自动提交当天的所有文件

```
python sandbox/submitAll.py
```

在`result/`下按照文件名中的日期搜索文件，并提交。