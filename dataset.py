
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from collections import namedtuple


# In[7]:


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
            self.meo, self.start, self.end, self.grids)

    def __init__(self, city="bj", T=2, T_future=1):
        print("loading data in %s" % city)
        self.city = city
        self.T = T
        self.T_future = T_future
        self.start = pd.Timestamp("2017-01-01 00:00:00")
        self.end = pd.Timestamp(
            datetime.datetime.utcnow().date()) - pd.Timedelta(1, unit="h")
        if city == "bj":
            self.aq = airQualityData("bj", self.start, self.end)
            self.meo = meteorologyGridData("bj", self.start, self.end)
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
            self.aq["O3"] = float("nan")
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
        print("load successfully!")

    def __len__(self):
        return (self.end - self.start).days + 1 - 2

    def __getitem__(self, idx):
        aq = self.aq.values[idx * len(self.stations) * 24: (idx + self.T)
                            * len(self.stations) * 24].reshape(-1)
        meo = self.meo.values[idx * len(self.grids) *
                              24: (idx + self.T) * len(self.grids) * 24].reshape(-1)
        meo_pred = self.meo.values[(idx + self.T) * len(self.grids) * 24:
                                   (idx + self.T + self.T_future) * len(self.grids) * 24].reshape(-1)
        y = self.aq.values[(idx + self.T) * len(self.stations) * 24:                                    (
            idx + self.T + self.T_future) * len(self.stations) * 24].reshape(-1)
        return KddData(aq, meo, meo_pred, y)


if __name__ == "__main__":
    city = 'bj'
    save_path = f'data/dataset_{city}.pt'
    if os.path.exists(save_path):
        dataset = torch.load(open(save_path, 'rb'))
    else:
        from data import *
        dataset = KddDataset("bj")
        torch.save(dataset, open(save_path, 'wb'))
    r = dataset[0]
    aq, meo, meo_pred, y = r
    print(aq.shape, meo.shape, meo_pred.shape, y.shape)

