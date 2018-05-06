import os
import pandas as pd
import numpy as np
import datetime
import config
import pickle
from collections import namedtuple
from utils.transform import date_to_idx, flatten_first_2_dimensions


AqMeo = namedtuple('AqMeo', ['aq', 'meo'])
AqMeoPredY = namedtuple('AqMeoPredY', ['aq', 'meo', 'meo_pred', 'y'])


class Dataset(object):

    def addMissingData(self, df, start, end, stations):
        df = df.drop_duplicates(["time", "station_id"])
        df.set_index(['time', 'station_id'], inplace=True)
        timeIdx = pd.date_range(start, end, freq="h").tolist()
        idx = pd.MultiIndex.from_product(
            [timeIdx, stations], names=['time', 'station_id'])
        df = df.reindex(idx)
        df.reset_index(level=[1, 0], inplace=True)
        return df

    def addData(self):
        print("adding missing data")
        self.aq = self.addMissingData(
            self.aq, self.start, self.end, self.stations)
        self.meo = self.addMissingData(
            self.meo, self.start, self.end + pd.Timedelta(2, unit="d"), self.grids)

    def __init__(self):
        print("loading data in %s" % config.CITY)
        self.w, self.h, self.aq_channels, self.meo_channels = {
            'bj': (31, 21, 3, 5),
            'ld': (41, 21, 2, 5)
        }[config.CITY]
        self.start = pd.Timestamp("2017-01-01 00:00:00")
        self.end = pd.Timestamp(
            datetime.datetime.utcnow().strftime("%Y-%m-%d %H:00:00"))
        if config.CITY == "bj":
            self.aq = airQualityData("bj", self.start, self.end)
            self.meo = meteorologyGridData(
                "bj", self.start, self.end + pd.Timedelta(2, unit="d"))
            self.stations = [
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
            self.grids = []
            for i in range(0, 651):
                tmp = str(i)
                while len(tmp) < 3:
                    tmp = "0" + tmp
                self.grids.append("beijing_grid_" + tmp)
        else:
            self.aq = airQualityData("ld", self.start, self.end)
            # self.aq["O3"] = float("nan")
            self.meo = meteorologyGridData("ld", self.start, self.end)
            self.stations = [
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
            self.grids = []
            for i in range(0, 861):
                tmp = str(i)
                while len(tmp) < 3:
                    tmp = "0" + tmp
                self.grids.append("london_grid_" + tmp)
        self.stations = sorted(self.stations)
        self.aq = self.aq[self.aq["station_id"].isin(self.stations)]
        self.addData()

        self.aq = self.aq.sort_values(by=["time", "station_id"])
        self.meo = self.meo.sort_values(by=["time", "station_id"])
        self.aq = self.aq.reset_index(drop=True)
        self.meo = self.meo.reset_index(drop=True)
        self.aq = self.aq.drop(columns=["time", "station_id"])
        self.meo = self.meo.drop(columns=["time", "station_id"])
        self.aq = self.aq.values
        self.meo = self.meo.values
        self._normalize()

        # Will discard the last day if incomplete (before 7:00AM China time)
        aq_datapoints_per_day = len(self.stations) * 24
        meo_datapoints_per_day = self.w * self.h * 24
        self.use_days = self.aq.size // aq_datapoints_per_day // self.aq_channels
        meo_days = self.meo.size // meo_datapoints_per_day // self.meo_channels
        self.aq = self.aq[:self.use_days * aq_datapoints_per_day]
        self.meo = self.meo[:meo_days * meo_datapoints_per_day]
        self.aq = self.aq.reshape(self.use_days, 24, len(
            self.stations), self.aq_channels).astype(np.float32)
        self.meo = self.meo.reshape(
            meo_days, 24, self.w, self.h, self.meo_channels)
        self.meo = np.transpose(self.meo, (0, 1, 4, 2, 3)).astype(np.float32)
        print("load successfully!")

    def _normalize(self):
        self.aq_mean = np.nanmean(self.aq, axis=0)
        self.aq_std = np.nanstd(self.aq, axis=0)
        self.aq -= self.aq_mean
        self.aq /= self.aq_std
        self.meo_mean = np.nanmean(self.meo, axis=0)
        self.meo_std = np.nanstd(self.meo, axis=0)
        self.meo -= self.meo_mean
        self.meo /= self.meo_std

    def __len__(self):
        return self.use_days

    def __getitem__(self, idx):
        return AqMeo(self.aq[idx], self.meo[idx])


class DatasetWrapper(object):

    def __init__(self, dataset):
        self.dataset = dataset
        self.stations = self.dataset.stations

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class EndDateWrapper(DatasetWrapper):
    '''Note: not including the end date. Should set to TOMORROW to get all.'''

    def __init__(self, dataset, date):
        super().__init__(dataset)
        self.end_idx = date_to_idx(date)

    def __len__(self):
        return min(self.end_idx, len(self.dataset))

    def __getitem__(self, idx):
        return self.dataset[idx]


class PastTDaysWrapper(DatasetWrapper):

    def __len__(self):
        return len(self.dataset) + 1 - config.PRED_FUTURE_T_DAYS - config.USE_PAST_T_DAYS

    def __getitem__(self, idx):
        aq, meo = self.dataset[idx: idx + config.USE_PAST_T_DAYS]
        y, meo_pred = self.dataset[idx + config.USE_PAST_T_DAYS:
                                   idx + config.USE_PAST_T_DAYS + config.PRED_FUTURE_T_DAYS]

        aq, meo = flatten_first_2_dimensions(
            aq), flatten_first_2_dimensions(meo)
        meo_pred, y = flatten_first_2_dimensions(
            meo_pred), flatten_first_2_dimensions(y)
        return AqMeoPredY(aq, meo, meo_pred, y)


class StationInvariantWrapper(DatasetWrapper):

    def __len__(self):
        return len(self.dataset.stations) * len(self.dataset)

    def __getitem__(self, idx):
        aq, meo = self.dataset[idx // len(self.dataset.stations)]
        return AqMeo(aq[:, idx % len(self.dataset.stations)], meo[:, idx % len(self.dataset.stations)])


class StationInvariantPastTDaysWrapper(PastTDaysWrapper):

    def __len__(self):
        return len(self.dataset.stations) * super().__len__()

    def __getitem__(self, idx):
        aq, meo, meo_pred, y = super().__getitem__(idx // len(self.dataset.stations))
        return AqMeoPredY(aq[:, idx % len(self.dataset.stations)], meo, meo_pred, y[:, idx % len(self.dataset.stations)])


class Subset(object):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def random_split(dataset, lengths):
    lengths = [length * len(dataset) // sum(lengths) for length in lengths]
    lengths[-1] += len(dataset) - sum(lengths)
    indices = torch.randperm(len(dataset))
    return [Subset(dataset, indices[sum(lengths[:i]):sum(lengths[:i + 1])])
            for i in range(len(lengths))]


if __name__ == "__main__":
    for city in ['bj', 'ld']:
        save_path = f'data/dataset_{city}.pkl'
        if os.path.exists(save_path):
            continue
        from utils.data import *
        config.CITY = city
        print(config.CITY)
        dataset = Dataset()
        with open(save_path, 'wb') as f:
            pickle.dump(dataset, f)

    print("Two datasets have been saved successfully!")
