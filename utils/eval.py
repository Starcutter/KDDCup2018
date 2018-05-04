import numpy as np
import torch
import pandas as pd
from utils.dataset import StationInvariantKddDataset


def SMAPE(actual, predicted):
    dividend = np.abs(np.array(actual) - np.array(predicted))
    denominator = np.array(actual) + np.array(predicted)
    smape = np.divide(dividend, denominator, out=np.zeros_like(
        dividend), where=denominator != 0, casting='unsafe')
    nan_mask = np.isnan(smape)
    return 2 * np.mean(smape[~nan_mask])


def eval(start_date, pred_file):
    dataset_bj = torch.load('data/dataset_bj.pt')
    bj_stations = dataset_bj.stations
    dataset_ld = torch.load('data/dataset_ld.pt')
    ld_stations = dataset_ld.stations

    date_to_eval = pd.Timestamp(start_date)

    truth_bj = np.vstack([dataset_bj.get_data(idx, date_to_eval - pd.Timedelta(1, unit='h')).y
        for idx in range(len(bj_stations))])

    truth_bj = truth_bj.reshape(len(bj_stations), -1, 3)
    truth_bj = truth_bj * dataset_bj.aq_std + dataset_bj.aq_mean
    truth_bj = truth_bj.reshape(-1, 3)

    truth_ld = np.vstack([dataset_ld.get_data(idx, date_to_eval - pd.Timedelta(1, unit='h')).y
        for idx in range(len(ld_stations))])

    truth_ld = truth_ld.reshape(len(ld_stations), -1, 2)
    truth_ld = truth_ld * dataset_ld.aq_std + dataset_ld.aq_mean
    truth_ld = truth_ld.reshape(-1, 2)
    truth_ld = np.hstack([truth_ld, np.zeros((len(truth_ld), 1))])

    truth_matrix = np.vstack([truth_bj, truth_ld])

    submit_df = pd.read_csv(pred_file).fillna(0.0)
    submit_matrix = submit_df.drop(columns=['test_id'], axis=1).as_matrix()

    pd.DataFrame(truth_matrix).to_csv('truth_matrix_new.csv')

    nan_lines = list(set(np.where(np.isnan(truth_matrix))[0]))
    mask = np.array([True] * truth_matrix.shape[0])
    mask[nan_lines] = False

    truth_matrix_filtered = truth_matrix[mask, :]
    submit_matrix_filtered = submit_matrix[mask, :]

    return SMAPE(truth_matrix_filtered, submit_matrix_filtered)


if __name__ == '__main__':
    print(eval('2018-05-04', 'results/submit_0504_0.csv'))
