
# coding: utf-8

# In[1]:


import os

import pandas as pd
import numpy as np
import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from collections import namedtuple


# In[4]:


KddData = namedtuple('KddData', ['aq', 'meo', 'meo_pred', 'y'])


class KddDataset(Dataset):

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

    def __init__(self, city="bj", T=2, T_future=2):
        if city == "bj":
            self.w = 31
            self.h = 21
        else:
            self.w = 41
            self.h = 21
        print("loading data in %s" % city)
        self.city = city
        self.T = T
        self.T_future = T_future
        self.start = pd.Timestamp("2017-01-01 00:00:00")
        self.end = pd.Timestamp(datetime.datetime.utcnow().strftime("%Y-%m-%d %H:00:00"))
        if city == "bj":
            self.aq = airQualityData("bj", self.start, self.end)
            self.meo = meteorologyGridData("bj", self.start, self.end + pd.Timedelta(2, unit="d"))
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
        return (self.end - self.start).days + 1 - self.T_future - self.T + 1

    def __getitem__(self, idx):
        aq = self.aq[idx * len(self.stations) * 24: (idx + self.T) * len(
            self.stations) * 24].reshape(self.T * 24, len(self.stations), -1)
        meo = self.meo[idx * len(self.grids) * 24: (idx + self.T) * len(
            self.grids) * 24].reshape(self.T * 24, self.w, self.h, -1)
        meo_pred = self.meo[(idx + self.T) * len(self.grids) * 24: (idx + self.T + self.T_future)
                            * len(self.grids) * 24].reshape(self.T_future * 24, self.w, self.h, -1)
        y = self.aq[(idx + self.T) * len(self.stations) * 24: (idx + self.T + self.T_future)
                    * len(self.stations) * 24].reshape(self.T_future * 24, len(self.stations), -1)
        # aq = np.reshape(aq, (self.T * 24, -1)).astype(np.float32)
        meo = np.transpose(meo, (0, 3, 1, 2)).astype(np.float32)
        meo_pred = np.transpose(meo_pred, (0, 3, 1, 2)).astype(np.float32)
        # y = np.reshape(y, (self.T_future * 24, -1)).astype(np.float32)
        return KddData(aq, meo, meo_pred, y)

    def getLatestData(self):
        idx = (self.end + pd.Timedelta(2, unit="d") - self.start).days * 24 \
            + (self.end - self.start).seconds // 3600 - (self.T + self.T_future) * 24 + 1
        aq = self.aq[idx * len(self.stations): (idx + self.T * 24) * len(
            self.stations)].reshape(self.T * 24, len(self.stations), -1)
        meo = self.meo[idx * len(self.grids): (idx + self.T * 24) * len(
            self.grids)].reshape(self.T * 24, self.w, self.h, -1)
        meo_pred = self.meo[(idx + self.T * 24) * len(self.grids): (idx + (self.T + self.T_future) * 24)
                            * len(self.grids)].reshape(self.T_future * 24, self.w, self.h, -1)
        meo = np.transpose(meo, (0, 3, 1, 2)).astype(np.float32)
        meo_pred = np.transpose(meo_pred, (0, 3, 1, 2)).astype(np.float32)
        return KddData(aq, meo, meo_pred, None)

class StationInvariantKddDataset(KddDataset):
    def __init__(self, city):
        super().__init__(city)

    def __len__(self):
        return len(self.stations) * super().__len__()

    def __getitem__(self, idx):
        aq, meo, meo_pred, y = super().__getitem__(idx // len(self.stations))
        return KddData(aq[:, idx % len(self.stations)], meo, meo_pred, y[:, idx % len(self.stations)])

    def getLatestData(self, idx):
        aq, meo, meo_pred, y = super().getLatestData()
        return KddData(aq[:, idx % len(self.stations)], meo, meo_pred, None)

class Subset(torch.utils.data.Dataset):
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
    save_path_bj = f'data/dataset_bj.pt'
    save_path_ld = f'data/dataset_ld.pt'
    if os.path.exists(save_path_bj) and os.path.exists(save_path_ld):
        pass
    else:
        from utils.data import *
        if not os.path.exists(save_path_bj):
            dataset_bj = StationInvariantKddDataset("bj")
            torch.save(dataset_bj, open(save_path_bj, 'wb'))
        if not os.path.exists(save_path_ld):
            dataset_ld = StationInvariantKddDataset("ld")
            torch.save(dataset_ld, open(save_path_ld, 'wb'))
    print("Two datasets have been saved successfully!")
