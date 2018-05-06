import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import config
from utils.dataset import Dataset, EndDateWrapper, StationInvariantPastTDaysWrapper
from utils.eval import SMAPE
from utils.transform import date_to_idx, flatten_first_2_dimensions


class RFModel(object):

    def __init__(self, city, date):
        dataset = pickle.load(open(f'data/dataset_{city}.pkl', 'rb'))
        self.ori_dataset = dataset
        self.aq_mean = dataset.aq_mean
        self.aq_std = dataset.aq_std
        dataset = EndDateWrapper(dataset, date)
        dataset = StationInvariantPastTDaysWrapper(dataset)

        if config.USE_MEO_PRED:
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
        print(x.shape, y.shape)

        train_x, test_x, train_y, test_y = train_test_split(
            x, y, test_size=0.2, random_state=config.SEED)

        # Keep nan in test_y to avoid calculaing smape on missing values
        train_x = np.nan_to_num(train_x)
        train_y_with_nan = train_y
        train_y = np.nan_to_num(train_y)
        test_x = np.nan_to_num(test_x)
        print('Size of train, validation: {}, {}'.format(
            train_x.shape, test_x.shape))

        model = RandomForestRegressor(n_jobs=-1)

        param_grid = {
            "n_estimators": [10],
            "max_features": [96],
            "min_samples_split": [4],
            "min_samples_leaf": [2],
            "bootstrap": [False],
        }

        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
        grid.fit(train_x, train_y)
        print(grid.best_score_)
        print(grid.best_params_)
        self.model = grid
        print('SMAPE on train set:', self.eval(train_x, train_y_with_nan))
        print('SMAPE on test set:', self.eval(test_x, test_y))

    def predict(self, x):
        return self.model.predict(x)

    def predict_on_date(self, date):
        idx = date_to_idx(date)

        aq, _ = self.ori_dataset[idx - config.USE_PAST_T_DAYS:idx]
        aq = flatten_first_2_dimensions(aq)
        meo_pred = flatten_first_2_dimensions(
            self.ori_dataset.meo[idx:idx + config.PRED_FUTURE_T_DAYS])
        aq = np.transpose(aq, (1, 0, 2))
        aq = aq.reshape((aq.shape[0], -1))
        meo_pred = np.mean(meo_pred, axis=(2, 3)).flatten()
        meo_pred = np.repeat(meo_pred[np.newaxis, :], repeats=aq.shape[0], axis=0)
        x = np.concatenate([aq, meo_pred], axis=1)
        x = np.nan_to_num(x)
        y_hat = self.predict(x)
        y_hat = y_hat.reshape(y_hat.shape[0], 24, -1)
        y_hat = y_hat * self.aq_std + self.aq_mean
        return y_hat

    def eval(self, x, y):
        y_hat = self.predict(x)
        y = y.reshape(y.shape[0], 24, -1)
        y_hat = y_hat.reshape(y_hat.shape[0], 24, -1)
        return SMAPE(y_hat * self.aq_std + self.aq_mean,
                     y * self.aq_std + self.aq_mean)


if __name__ == '__main__':
    for city in ['bj', 'ld']:
        config.CITY = city
        model = RFModel(city, config.TRAIN_BEFORE_DATE)
        with open(config.MODEL_SAVED_PATH.format(city), 'wb') as f:
            pickle.dump(model, f)
