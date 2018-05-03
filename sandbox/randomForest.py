import torch
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from utils.dataset import StationInvariantKddDataset
from utils.eval import SMAPE


dataset = torch.load('data/dataset_bj.pt')

x = np.vstack([np.concatenate([dataset[idx].aq.flatten()])
               for idx in range(len(dataset))])
y = np.vstack([np.concatenate([dataset[idx].y.flatten()])
               for idx in range(len(dataset))])

# x.shape: (485 days * 35 stations, 48 hours * 3 index)
# y.shape: (485 days * 35 stations, 24 hours * 3 index)
print(x.shape, y.shape)

train_x, test_x, train_y, test_y = train_test_split(
    x, y, test_size=0.2, random_state=0)

# Keep nan in test_y to avoid calculaing smape on missing values
train_x = np.nan_to_num(train_x)
train_y_with_nan = train_y
train_y = np.nan_to_num(train_y)
test_x = np.nan_to_num(test_x)
print('Size of train, validation: {}, {}'.format(train_x.shape, test_x.shape))

model = RandomForestRegressor()

param_grid = {"n_estimators": [10, 200],
    "max_features": ['auto', 30, 80],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 3, 10],
    "bootstrap": [True, False]
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid.fit(train_x, train_y)
print(grid.best_score_)
print(grid.best_params_)


def eval(model, x, y):
    y_hat = model.predict(x)
    y = y.reshape(y.shape[0], 24, -1)
    y_hat = y_hat.reshape(y_hat.shape[0], 24, -1)
    return SMAPE(y_hat * dataset.aq_std + dataset.aq_mean,
                 y * dataset.aq_std + dataset.aq_mean)


print('SMAPE on train set:', eval(grid, train_x, train_y_with_nan))
print('SMAPE on test set:', eval(grid, test_x, test_y))
