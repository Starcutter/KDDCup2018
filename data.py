
# coding: utf-8

# In[2]:


import pandas as pd


# # Beijing AQ

# In[3]:


df1 = pd.read_csv('data/beijing_201802_201803_aq.csv')
df2 = pd.read_csv('data/beijing_17_18_aq.csv')
stations = list(df1.groupby(by="stationId").size().index)
l = []
for station in stations:
    new = df1[df1.stationId==station]
    old = df2[df2.stationId==station]
    l.extend([old, new])
beijing_aq = pd.concat(l)
beijing_aq["utc_time"] = pd.to_datetime(beijing_aq["utc_time"])
beijing_aq = beijing_aq.rename(columns={"stationId": "station_id"})


# # London AQ

# In[48]:


df1 = pd.read_csv('data/London_historical_aqi_forecast_stations_20180331.csv', index_col=0)
df2 = pd.read_csv('data/London_historical_aqi_other_stations_20180331.csv')
df2 = df2.drop(["Unnamed: 5", "Unnamed: 6"], axis=1)
df2 = df2.dropna(how="all")
df2 = df2[: -1]
df2 = df2.rename(columns={"Station_ID": "station_id"})
london_aq = pd.concat([df1, df2])
london_aq = london_aq.rename(columns={"MeasurementDateGMT": "utc_time"})
london_aq["utc_time"] = pd.to_datetime(london_aq["utc_time"])


# # Beijing grid meteorology

# In[68]:


df = pd.read_csv("data/Beijing_historical_meo_grid.csv")


# In[71]:


beijing_meo = df.rename(columns={
    "stationName": "station_id",
    "utc_time": "time",
    "wind_speed/kph": "wind_speed",
})
beijing_meo["time"] = pd.to_datetime(beijing_meo["time"])


# # London grid meteorology

# In[63]:


df = pd.read_csv("data/London_historical_meo_grid.csv")


# In[66]:


london_meo = df.rename(columns={
    "stationName": "station_id",
    "utc_time": "time",
    "wind_speed/kph": "wind_speed"
})
london_meo["time"] = pd.to_datetime(london_meo["time"])


# # APIs

# In[103]:


import requests


def buildDataFrame(text):
    if len(text) < 10:
        return pd.DataFrame()
    tmp = text.split("\n")
    if (tmp[-1] == ""):
        tmp = tmp[: -1]
    for i in range(0, len(tmp)):
        if tmp[i][-1] == "\r":
            tmp[i] = tmp[i][: -1]
        tmp[i] = tmp[i].split(",")
    df = pd.DataFrame(tmp[1: ], columns=tmp[0])
    return df


def airQualityData(city="bj", year=2018, month=5, day=1, hour=0):
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
    if year == 2018 and month >= 4:
        url = 'https://biendata.com/competition/airquality/%s/%d-%d-%d-%d/%d-%d-%d-%d/2k0d1d8' % (city, year, month, day, hour, year, month, day, hour)
        response = requests.get(url)
        df = buildDataFrame(response.text)
        try:
            df["time"] = pd.to_datetime(df["time"])
        except:
            pass
    else:
        time = pd.Timestamp("%d-%d-%d %d:00:00" % (year, month, day, hour))
        if city == "bj":
            df = beijing_aq[beijing_aq["utc_time"] == time]
        else:
            df = london_aq[london_aq["utc_time"] == time]
        df = df.reset_index()
    
    df = df.rename(columns=dic)
    try:
        df = df[target]
    except:
        pass
    if df.empty:
        df = pd.DataFrame()
    return df


def meteorologyGridData(city="bj", year=2018, month=5, day=1, hour=0):
    target = ["station_id", "time", "temperature", "pressure",
              "humidity", "wind_direction", "wind_speed"]
    url = 'https://biendata.com/competition/meteorology/%s_grid/%d-%d-%d-%d/%d-%d-%d-%d/2k0d1d8' % (city, year, month, day, hour, year, month, day, hour)
    if year == 2018 and month >= 4:
        response = requests.get(url)
        df = buildDataFrame(response.text)
        try:
            df["time"] = pd.to_datetime(df["time"])
        except:
            pass
    else:
        time = pd.Timestamp("%d-%d-%d %d:00:00" % (year, month, day, hour))
        if city == "bj":
            df = beijing_meo[beijing_meo["time"] == time]
        else:
            df = london_meo[london_meo["time"] == time]
    try:
        df = df[target]
    except:
        pass
    df = df.reset_index(drop=True)
    if df.empty:
        df = pd.DataFrame()
    return df


def observedMeteorology(city="bj", year=2018, month=5, day=1, hour=0):
    if city != "bj":
        print("No observed meteorology data for London!!!!")
        return
    url = 'https://biendata.com/competition/meteorology/%s/%d-%d-%d-%d/%d-%d-%d-%d/2k0d1d8' % (city, year, month, day, hour, year, month, day, hour)
    response = requests.get(url)
    df = buildDataFrame(response.text)
    if df.empty:
        df = pd.DataFrame()
    return df

# airQualityData("ld", 2017, 4, 26, 2)
meteorologyGridData("ld", 2018, 4, 26, 2)

