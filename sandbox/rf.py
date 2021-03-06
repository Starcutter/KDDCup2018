import torch
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from utils.dataset import StationInvariantKddDataset
from utils.eval import SMAPE
import pandas as pd


def getRandomForestModel(date, city="bj", random_state=0, use_pred=True):
    dataset = torch.load(f'data/dataset_{city}.pt')
    dataset.T = 7

    if use_pred:
        x = np.vstack([np.concatenate([dataset[idx].aq.flatten(),
                                    np.mean(dataset[idx].meo_pred, axis=(2, 3)).flatten()])
                    for idx in range(len(dataset))])
    else:
        x = np.vstack([np.concatenate([dataset[idx].aq.flatten()])
                    for idx in range(len(dataset))])
    y = np.vstack([np.concatenate([dataset[idx].y.flatten()])
                for idx in range(len(dataset))])

    # x.shape: (485 days * 35 stations, T * 24 hours * 3 index)
    # y.shape: (485 days * 35 stations, 24 hours * 3 index)
    print("len = ", len(dataset))
    x = x[: len(dataset.stations) * ((date - pd.Timestamp("2017-01-01")).days - dataset.T_future - dataset.T + 1)]
    y = y[: len(dataset.stations) * ((date - pd.Timestamp("2017-01-01")).days - dataset.T_future - dataset.T + 1)]
    print(x.shape, y.shape)

    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.2, random_state=random_state)

    # Keep nan in test_y to avoid calculaing smape on missing values
    train_x = np.nan_to_num(train_x)
    train_y_with_nan = train_y
    train_y = np.nan_to_num(train_y)
    test_x = np.nan_to_num(test_x)
    print('Size of train, validation: {}, {}'.format(train_x.shape, test_x.shape))

    model = RandomForestRegressor(n_jobs=-1)

    if city == "bj":
        param_grid = {
            "n_estimators": [40],
            "max_features": [96],
            "min_samples_split": [4],
            "min_samples_leaf": [2],
            "bootstrap": [False],
        }
    else:
        param_grid = {
            "n_estimators": [40],
            "max_features": [96],
            "min_samples_split": [4],
            "min_samples_leaf": [2],
            "bootstrap": [False],
        }

    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    grid.fit(train_x, train_y)
    print(grid.best_score_)
    print(grid.best_params_)


    def eval(model, x, y):
        y_hat = model.predict(x)
        y = y.reshape(y.shape[0], 24 * 2, -1)
        y_hat = y_hat.reshape(y_hat.shape[0], 24 * 2, -1)
        return SMAPE(y_hat * dataset.aq_std + dataset.aq_mean,
                    y * dataset.aq_std + dataset.aq_mean)


    print('SMAPE on train set:', eval(grid, train_x, train_y_with_nan))
    print('SMAPE on test set:', eval(grid, test_x, test_y))

    return grid

if __name__ == '__main__':
    # model_bj = getRandomForestModel("bj", 0)
    model_ld = getRandomForestModel("ld", 0)