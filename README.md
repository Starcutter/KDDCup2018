# KDDCup2018

|folders|contents|
|---|---|
|sandbox|尝试进行的对数据的观察学习以及实验等|
|data|训练数据|
|utils|通用函数|
|models|各种模型|
|results|提交结果|

### 自动预测，生成CSV:

```
python utils/gen_csv.py <random_state>
```

`<random_state>`为可选参数，整形，随机种子。不填则默认为0.

将在`results/`文件夹中生成名称为`submit_<%m%d>_<random_state>.csv`的提交文件

### 自动提交当天的所有文件

```
python utils/all_submit.py
```
