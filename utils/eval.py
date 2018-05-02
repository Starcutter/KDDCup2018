import numpy as np
import os
os.chdir("../")
from data import *

def SMAPE(actual, predicted):
    dividend = np.abs(np.array(actual) - np.array(predicted))
    denominator = np.array(actual) + np.array(predicted)
    return 2 * np.mean(np.divide(dividend, denominator, out=np.zeros_like(dividend), where=denominator != 0, casting='unsafe'))

def eval(start_date, pred_file):
    bj_stations = [
        'aotizhongxin_aq',
        'badaling_aq',
        'beibuxinqu_aq',
        'daxing_aq',
        'dingling_aq',
        'donggaocun_aq',
        'dongsi_aq',
        'dongsihuan_aq',
        'fangshan_aq',
        'fengtaihuayuan_aq',
        'guanyuan_aq',
        'gucheng_aq',
        'huairou_aq',
        'liulihe_aq',
        'mentougou_aq',
        'miyun_aq',
        'miyunshuiku_aq',
        'nansanhuan_aq',
        'nongzhanguan_aq',
        'pingchang_aq',
        'pinggu_aq',
        'qianmen_aq',
        'shunyi_aq',
        'tiantan_aq',
        'tongzhou_aq',
        'wanliu_aq',
        'wanshouxigong_aq',
        'xizhimenbei_aq',
        'yanqin_aq',
        'yizhuang_aq',
        'yongdingmennei_aq',
        'yongledian_aq',
        'yufa_aq',
        'yungang_aq',
        'zhiwuyuan_aq'
    ]
    ld_stations = [
        'CD1',
        'GN0',
        'LW2',
        'GR4',
        'MY7',
        'KF1',
        'GR9',
        'BL0',
        'CD9',
        'TH4',
        'HV1',
        'ST5',
        'GN3'
    ]
    date_to_eval = pd.Timestamp(start_date)
    bj_df = airQualityData('bj', date_to_eval, date_to_eval + pd.Timedelta(47, unit='h'))
    ld_df = airQualityData('ld', date_to_eval, date_to_eval + pd.Timedelta(47, unit='h'))

    truth_df = pd.DataFrame()
    for station in bj_stations:
        station_df = bj_df[bj_df['station_id'] == station].drop_duplicates(['time'])
        dt = date_to_eval
        for i in range(48):
            if len(station_df[station_df['time'] == dt]) == 0:
                t = pd.DataFrame([station, dt, np.NAN, np.NAN, np.NAN]).T
                t.columns = station_df.columns
                station_df = pd.concat([station_df, t], ignore_index=True)
            dt += pd.Timedelta(1, unit='h')
        station_df.sort_values(by='time', inplace=True)
        truth_df = pd.concat([truth_df, station_df])

    for station in ld_stations:
        station_df = ld_df[ld_df['station_id'] == station].drop_duplicates(['time'])
        station_df['O3'] = 0.0
        dt = date_to_eval
        for i in range(48):
            if len(station_df[station_df['time'] == dt]) == 0:
                t = pd.DataFrame([station, dt, np.NAN, np.NAN, 0.0]).T
                t.columns = station_df.columns
                station_df = pd.concat([station_df, t], ignore_index=True)
            dt += pd.Timedelta(1, unit='h')
        station_df.sort_values(by='time', inplace=True)
        truth_df = pd.concat([truth_df, station_df])

    truth_df = truth_df.reset_index(drop=True)

    submit_df = pd.read_csv(pred_file).fillna(0.0)
    submit_matrix = submit_df.drop(columns=['test_id'], axis=1).as_matrix()

    truth_matrix = truth_df.drop(columns=['station_id', 'time'], axis=1).as_matrix().astype(np.float64)

    nan_lines = list(set(np.where(np.isnan(truth_matrix))[0]))
    mask = np.array([True] * truth_matrix.shape[0])
    mask[nan_lines] = False

    truth_matrix_filtered = truth_matrix[mask, :]
    submit_matrix_filtered = submit_matrix[mask, :]

    return SMAPE(truth_matrix_filtered, submit_matrix_filtered)


if __name__ == '__main__':
    print(eval('2018-05-01', 'results/submit_0501.csv'))



