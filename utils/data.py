
# coding: utf-8

# In[1]:


import os
import pandas as pd
import datetime


# In[2]:


if not os.path.exists("./data/api"):
    os.mkdir("./data/api")
    os.mkdir("./data/api/bj")
    os.mkdir("./data/api/bj/aq")
    os.mkdir("./data/api/bj/meo")
    os.mkdir("./data/api/ld")
    os.mkdir("./data/api/ld/aq")
    os.mkdir("./data/api/ld/meo")


# # Beijing AQ

# In[3]:


df1 = pd.read_csv('data/beijing_201802_201803_aq.csv')
df2 = pd.read_csv('data/beijing_17_18_aq.csv')
stations = list(df1.groupby(by="stationId").size().index)
l = []
for station in stations:
    new = df1[df1.stationId == station]
    old = df2[df2.stationId == station]
    l.extend([old, new])
beijing_aq = pd.concat(l)
beijing_aq["utc_time"] = pd.to_datetime(beijing_aq["utc_time"])
beijing_aq = beijing_aq.rename(columns={"stationId": "station_id"})


# # London AQ

# In[4]:


df1 = pd.read_csv(
    'data/London_historical_aqi_forecast_stations_20180331.csv', index_col=0)
df2 = pd.read_csv('data/London_historical_aqi_other_stations_20180331.csv')
df2 = df2.drop(["Unnamed: 5", "Unnamed: 6"], axis=1)
df2 = df2.dropna(how="all")
df2 = df2[: -1]
df2 = df2.rename(columns={"Station_ID": "station_id"})
london_aq = pd.concat([df1, df2])
london_aq = london_aq.rename(columns={"MeasurementDateGMT": "utc_time"})
london_aq["utc_time"] = pd.to_datetime(london_aq["utc_time"])


# # Beijing grid meteorology

# In[5]:


df = pd.read_csv("data/Beijing_historical_meo_grid.csv")


# In[6]:


beijing_meo = df.rename(columns={
    "stationName": "station_id",
    "utc_time": "time",
    "wind_speed/kph": "wind_speed",
})
beijing_meo["time"] = pd.to_datetime(beijing_meo["time"])


# # London grid meteorology

# In[7]:


df = pd.read_csv("data/London_historical_meo_grid.csv")


# In[8]:


london_meo = df.rename(columns={
    "stationName": "station_id",
    "utc_time": "time",
    "wind_speed/kph": "wind_speed"
})
london_meo["time"] = pd.to_datetime(london_meo["time"])


# # APIs

# In[9]:


import requests
from io import StringIO


def buildDataFrame(text):
    text = StringIO(text)
    df = pd.read_csv(text)
    return df


# In[26]:


def airQualityData(city="bj", start=pd.Timestamp("2018-04-01 00:00:00"),
                        end=pd.Timestamp("2018-04-01 23:00:00")):
    now = pd.Timestamp(datetime.datetime.utcnow().date())
    i = pd.Timestamp("2018-04-01")
    while i < now:
        filename = "./data/api/%s/aq/%d-%d.csv" % (city, i.month, i.day)
        if not os.path.exists(filename):
            e = i + pd.Timedelta(1, unit="d")
            url = 'https://biendata.com/competition/airquality/%s/%d-%d-%d-%d/%d-%d-%d-%d/2k0d1d8' % (
                city, i.year, i.month, i.day, i.hour, e.year, e.month, e.day, e.hour)
            response = requests.get(url)
            df = buildDataFrame(response.text)
            df.to_csv(filename, index=None)
        i += pd.Timedelta(1, unit="d")
    if city == "bj":
        target = ["station_id", "time", "PM2.5", "PM10", "O3"]
    else:
        target = ["station_id", "time", "PM2.5", "PM10"]
    dic = {
        "PM25_Concentration": "PM2.5",
        "PM10_Concentration": "PM10",
        "O3_Concentration": "O3",
        "PM10 (ug/m3)": "PM10",
        "PM2.5 (ug/m3)": "PM2.5",
        "utc_time": "time",
    }
    mid = pd.Timestamp("2018-04-01 00:00:00")
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    if end >= mid:
        i = pd.Timestamp("2018-04-01")
        while i < now:
            tmp = pd.read_csv("./data/api/%s/aq/%d-%d.csv" %
                              (city, i.month, i.day))
            tmp["time"] = pd.to_datetime(tmp["time"])
            df1 = pd.concat([df1, tmp])
            i += pd.Timedelta(1, unit="d")
        if end >= now:
            url = 'https://biendata.com/competition/airquality/%s/%d-%d-%d-%d/%d-%d-%d-%d/2k0d1d8' % (
                city, now.year, now.month, now.day, now.hour, end.year, end.month, end.day, end.hour)
            response = requests.get(url)
            tmp = buildDataFrame(response.text)
            tmp["time"] = pd.to_datetime(tmp["time"])
            df1 = pd.concat([df1, tmp])
        df1 = df1.rename(columns=dic)
        df1 = df1[(df1.time >= start) & (df1.time <= end)]
        try:
            df1 = df1[target]
        except:
            pass
        try:
            df1["time"] = pd.to_datetime(df1["time"])
        except:
            pass
    if start < mid:
        if city == "bj":
            df2 = beijing_aq[(start <= beijing_aq["utc_time"])
                             & (beijing_aq["utc_time"] <= end)]
        else:
            df2 = london_aq[(start <= london_aq["utc_time"])
                            & (london_aq["utc_time"] <= end)]
        df2 = df2.rename(columns=dic)
        try:
            df2 = df2[target]
        except:
            pass
    df = pd.concat([df2, df1])
    df = df.reset_index(drop=True)
    if df.empty:
        df = pd.DataFrame()
    return df


# In[41]:


def meteorologyGridData(city="bj", start=pd.Timestamp("2018-04-01 00:00:00"),
                        end=pd.Timestamp("2018-04-01 23:00:00")):
    now = pd.Timestamp(datetime.datetime.utcnow().date())
    i = pd.Timestamp("2018-04-01")
    while i < now:
        filename = "./data/api/%s/meo/%d-%d.csv" % (city, i.month, i.day)
        if not os.path.exists(filename):
            e = i + pd.Timedelta(1, unit="d")
            url = 'https://biendata.com/competition/meteorology/%s_grid/%d-%d-%d-%d/%d-%d-%d-%d/2k0d1d8' % (
                city, i.year, i.month, i.day, i.hour, e.year, e.month, e.day, e.hour)
            response = requests.get(url)
            df = buildDataFrame(response.text)
            df.to_csv(filename, index=None)
        i += pd.Timedelta(1, unit="d")
    target = ["station_id", "time", "temperature", "pressure",
              "humidity", "wind_direction", "wind_speed"]
    mid = pd.Timestamp("2018-04-01 00:00:00")
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    df3 = pd.DataFrame()
    if end >= mid:
        i = pd.Timestamp("2018-04-01")
        while i < now:
            tmp = pd.read_csv("./data/api/%s/meo/%d-%d.csv" %
                              (city, i.month, i.day))
            tmp["time"] = pd.to_datetime(tmp["time"])
            df1 = pd.concat([df1, tmp])
            i += pd.Timedelta(1, unit="d")
        if end >= now:
            url = 'https://biendata.com/competition/meteorology/%s_grid/%d-%d-%d-%d/%d-%d-%d-%d/2k0d1d8' % (
                city, now.year, now.month, now.day, now.hour, end.year, end.month, end.day, end.hour)
            response = requests.get(url)
            tmp = buildDataFrame(response.text)
            tmp["time"] = pd.to_datetime(tmp["time"])
            df1 = pd.concat([df1, tmp])
        df1 = df1[(df1.time >= start) & (df1.time <= end)]
        try:
            df1 = df1[target]
        except:
            pass
        try:
            df1["time"] = pd.to_datetime(df1["time"])
        except:
            pass
    if start < mid:
        if city == "bj":
            df2 = beijing_meo[(start <= beijing_meo["time"])
                              & (beijing_meo["time"] <= end)]
        else:
            df2 = london_meo[(start <= london_meo["time"])
                             & (london_meo["time"] <= end)]
        try:
            df2 = df2[target]
        except:
            pass
    df = pd.concat([df2, df1])
    now = pd.Timestamp(datetime.datetime.utcnow().strftime("%Y-%m-%d %H:00:00"))
    if now < end:
        s = now - pd.Timedelta(1, unit="h")
        url = "http://kdd.caiyunapp.com/competition/forecast/%s/%d-%d-%d-%d/2k0d1d8" % (
            city, s.year, s.month, s.day, s.hour)
        response = requests.get(url)
        df3 = buildDataFrame(response.text)
        df3 = df3.rename(columns={"forecast_time": "time"})
        try:
            df3["time"] = pd.to_datetime(df3["time"])
            df3 = df3[df3["time"] > now]
            df3 = df3[target]
        except:
            df3 = pd.DataFrame()
    df = pd.concat([df, df3])
    df = df.reset_index(drop=True)
    if df.empty:
        df = pd.DataFrame()
    return df
