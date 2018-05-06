import pickle
import numpy as np
import pandas as pd
import config
from models.rf import RFModel
from utils.dataset import Dataset
from transform import date_to_idx, flatten_first_2_dimensions


def getDataFrame(date):
    model = pickle.load(open(config.MODEL_SAVED_PATH.format(config.CITY), 'rb'))
    n_stations, n_features, metrics = {
        'bj': (35, 3, ['PM2.5', 'PM10', 'O3']),
        'ld': (13, 2, ['PM2.5', 'PM10']),
    }[config.CITY]

    print(model.predict_on_date)
    y_hat = model.predict_on_date(date)
    y_hat = y_hat.reshape((n_stations * 24, n_features))
    y_hat[y_hat < 0] = 0

    # TODO fix stations
    df = pd.DataFrame(y_hat, columns=metrics)
    df.insert(0, 'test_id', [station + '#' + str(i)
                             for station in stations for i in range(48)])

    return df


if __name__ == '__main__':

    config.CITY = 'bj'
    bj_df = getDataFrame(config.PRED_ON_DATE)
    config.CITY = 'ld'
    ld_df = getDataFrame(config.PRED_ON_DATE)

    df = pd.concat([bj_df, ld_df])
    submission = df.reindex(columns=['test_id', 'PM2.5', 'PM10', 'O3'])
    submission = submission.reset_index(drop=True)

    filename = 'results/submit_' + \
        date.strftime('%m%d%H%M') + '_' + str(random_state) + '.csv'
    submission.to_csv(filename, index=None)
