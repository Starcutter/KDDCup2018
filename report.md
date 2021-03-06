# KDD CUP of Fresh Air

## 计 55 RainForest 组

##### 左浩佳 徐韧喆 陈齐斌



### 1 代码组织结构

- 代码提交、组织结构与运行方法见 Github repo [KDDCup2018](https://github.com/Starcutter/KDDCup2018)。
- 代码结构如下：
  - 根目录
    - `networks`文件夹，神经网络尝试，详情见后文solution 2.
    - `sandbox`文件夹
      - `demo.py`，fbprophet尝试，详情见后文solution 1.
      - `rf.py`，随机森林尝试，详情见后文solution 3.
      - `genCSV.py`，生成提交文件。
    - `utils`文件夹
      - `data.py`，数据预处理
      - `dataset.py`，将处理好的数据整合成数据集格式。
      - `eval.py`，自己实现的SMAPE评估函数。



### 2 小组成员分工与工作量

| 任务      |     小组成员 |
| :-------: | :--------:|
| Data Preprocessing | 徐韧喆、左浩佳 |
| Solution 1 (fbprophet) | 左浩佳 |
| Solution 2 (neural networks) | 陈齐斌 |
| Solution 3 (Random forest regression) | 徐韧喆 |
| Hyperparameter tuning | 左浩佳、陈齐斌 |
| result submission | 左浩佳 |

三人分工均匀，工作量大致相当，在实验过程中我们尝试了三种不同的模型，每人分别实现了一种。



### 3 数据处理和模型构建中所做的尝试

#### 3.1 数据预处理

##### 3.1.1 处理流程

```flow
st=>start: 原始数据
op=>operation: 将日期时间信息转为索引；
整合获取最新数据接口
op0=>operation: 数据补全、归一化
op1=>operation: 一个站点历史 N 天的PM2.5、PM10、O3数据；
历史 N 天的网格气象数据；
未来 M 天的网格预测气象数据
e=>end: 预测一个站点未来两天的 PM2.5、PM10、O3

st->op->op0->op1->e
```

#### 3.1.2 预处理方法

- 在统一数据调用的接口后（如：获得某站点连续某几天的数据），为了尝试各种预处理，以及它们的组合的效果，我们使用了 wrapper 的方式实现预处理。
- 在规定如下的 dataset 索引方式之后（即用 dset[i] 获取单个数据点，len(dset) 得到数据集大小），可以用 wrapper 的方式，对数据集进行包装，从而尝试各种预处理、特征提取方法的使用先后顺序、最佳组合。
- 例如如下的 `EndDateWrapper` 可以截取数据集使用的日期区间；`PastTDaysWrapper` 可以将一个用一天的数据作为 feature 的数据集转变为用 N 天数据作为 feature 的数据集；`FillNaNsWrapper`会用平均值+噪声填充数据集中的NaN值，而不改变对之后算法的接口。
- 例：`dataset = FillNaNsWrapper(PastTDaysWrapper(orig_dataset, t))`

```python
class Dataset(object):

    def addData(self):
    ...

    def __init__(self):
    ...

    def _normalize(self):
    ...

    def __len__(self):
    ...

    def __getitem__(self, idx):
    ...
```

```python
class PastTDaysWrapper(DatasetWrapper):

    def __len__(self):
        return len(self.dataset) + 1 - config.PRED_FUTURE_T_DAYS -
               config.USE_PAST_T_DAYS

    def __getitem__(self, idx):
        aq, meo = self.dataset[idx: idx + config.USE_PAST_T_DAYS]
        y, meo_pred = self.dataset[idx + config.USE_PAST_T_DAYS:
                                   idx + config.USE_PAST_T_DAYS +
                                   config.PRED_FUTURE_T_DAYS]

        aq, meo = flatten_first_2_dimensions(
            aq), flatten_first_2_dimensions(meo)
        meo_pred, y = flatten_first_2_dimensions(
            meo_pred), flatten_first_2_dimensions(y)
        return AqMeoPredY(aq, meo, meo_pred, y)

class EndDateWrapper(DatasetWrapper):
...

class FillNaNsWrapper(DatasetWrapper):
...
```

#### 3.2 Solution 1 (fbprophet)

- 时间序列预测工具 [fbprophet](https://facebook.github.io/prophet/)
	- 以天为单位的历史周期数据，可提供季节性调整
	- 分段拟合，change point 频率需要调参
	- 误差增长快；易过拟合；难以同时考虑天气


#### 3.3 Solution 2 (neural networks)

- Baseline (0.7~0.8)
	- Simple RNN 本站点的 air quality
	- Multi-task loss （没有用预测出来的结果当作输入）
	- NaNs to 0
- Multi-head network (0.6)
	- BiLSTM
	- Convolutional LSTM (arXiv:1506.04214) 处理时序的（过去和 caiyun 预报）气象站 grid 信息
	- Concat features
	- 缺少合理的预处理
		- Station invariant (365 * 35) 可以保证数据足够，但无法区分 grid 信息对不同位置的站点的影响
		- 应该以站点为中心把 grid 数据输入给 ConvLSTM


#### 3.4 Solution 3 (Random forest regression)

- 原理
  - 随机森林由许多的决策树组成，因为这些决策树的形成采用了随机的方法，所以叫做随机森林。随机森林中的决策树之间是没有关联的，当测试数据进入随机森林时，其实就是让每一颗决策树进行分类看看这个样本应该属于哪一类，最后取所有决策树中分类结果最多的那类为最终的结果（每棵树的权重要考虑进来）。详情可查阅课程ppt6.2.
- 实现
  - 使用sklearn.ensemble.RandomForestRegressor回归器进行计算，用一个站点历史七天的PM2.5、PM10、O3数据和未来两天的网格预测气象数据的平均值(7 * 24 * 3 + 2 * 24 * 5)预测未来两天的PM2.5、PM10、O3数据(2 * 24 * 3)。

#### 3.5 Hyperparameter tuning 0.5 --> 0.4

- 超参数
  - past_n_days: 使用过去几天的空气与气象数据
  - future_n_days: 使用将来几天的气象预报数据
  - n_estimators: forest 中 tree 的数量
  - max_features: 自动选取 feature 的最大数量
  - min_samples_split, min_samples_leaf, bootstrap …
- sklearn.model_selection.GridSearchCV

<img align="center" width="60%" src="http://otukr87eg.bkt.clouddn.com/f68a5c55d0ee16b4f8c55e7c4f4a0836.jpg"></img>



### 4 使用的工具

- python
  - fbprophet库：用于时间序列预测
  - pytorch：用于实现神经网络
  - sklearn中RandomForestRegressor：使用随机森林进行拟合



### 5 最终采用的方案

采用方案三随机森林的办法。



### 6 结果 (SMAPE)

去掉结果最差的5天之后，在官方榜单上，我们的SMAPE结果是0.4646，排名66位。