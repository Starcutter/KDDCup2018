import requests
import datetime
import pandas as pd


def buildDataFrame(text):
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
    url = 'https://biendata.com/competition/airquality/%s/%d-%d-%d-%d/%d-%d-%d-%d/2k0d1d8' % (city, year, month, day, hour, year, month, day, hour)
    response = requests.get(url)
    return buildDataFrame(response.text)


def meteorologyGridData(city="bj", year=2018, month=5, day=1, hour=0):
    url = 'https://biendata.com/competition/meteorology/%s_grid/%d-%d-%d-%d/%d-%d-%d-%d/2k0d1d8' % (city, year, month, day, hour, year, month, day, hour)
    response = requests.get(url)
    return buildDataFrame(response.text)


def observedMeteorology(city="bj", year=2018, month=5, day=1, hour=0):
    if city != "bj":
        print("No observed meteorology data for London!!!!")
        return
    url = 'https://biendata.com/competition/meteorology/%s/%d-%d-%d-%d/%d-%d-%d-%d/2k0d1d8' % (city, year, month, day, hour, year, month, day, hour)
    response = requests.get(url)
    return buildDataFrame(response.text)
