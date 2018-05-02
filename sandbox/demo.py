
# coding: utf-8

import os
import pandas as pd
import numpy as np
from fbprophet import Prophet
from utils.data import *
import time
from datetime import datetime


df = pd.read_csv('data/beijing_201802_201803_aq.csv')
bj_stations = list(df.groupby(by='stationId').size().index)

df = pd.read_csv('results/sample_submission.csv')
ld_stations = list(set([x.split('#')[0] for x in list(df['test_id'])]) - set(bj_stations))


def predict(city='bj', metrics=['PM2.5'], station_names=['aotizhongxin_aq'],
            date_to_forc=pd.Timestamp('2018-05-02 00:00:00'), changepoint_prior_scale=10):
    df = airQualityData(city, date_to_forc - pd.Timedelta(7, unit='d'),
                        date_to_forc - pd.Timedelta(1, unit='h'))
    submit_df = pd.DataFrame()
    for station in station_names:
        df_station = pd.DataFrame()
        df_station['test_id'] = [station + '#' + str(i) for i in range(0, 48)]
        for metric in metrics:
            try:
                metric_df = df[df['station_id'] == station][['time', metric]]
                metric_df.columns = ['ds', 'y']
                metric_df = metric_df.reset_index(drop=True)
                metric_df['y'] = np.log(metric_df['y'])

                #metric_df['cap'] = 500 if metric == 'PM2.5' else 1000
                #metric_df['floor'] = 10
                m = Prophet(#growth='logistic', # default: linear
                            yearly_seasonality=False, weekly_seasonality=False, daily_seasonality='auto',
                            #seasonality_prior_scale=0.1, # no use if all seasonality set to False
                            changepoint_prior_scale=changepoint_prior_scale)
                m.fit(metric_df)
                future = m.make_future_dataframe(periods=72, freq='H', include_history=True)
                #future['cap'] = 500 if metric == 'PM2.5' else 1000
                #future['floor'] = 10
                forecast = m.predict(future)
                forecast = forecast[(forecast['ds'] >= date_to_forc)
                                    & (forecast['ds'] < date_to_forc + pd.Timedelta(2, unit='d'))]
                df_station[metric] = list(np.exp(forecast['yhat']))
            except:
                df_station[metric] = list(np.random.random((48)) * 50)
        submit_df = pd.concat([submit_df, df_station])
    return submit_df


today = pd.Timestamp(datetime.utcfromtimestamp(time.time()))
today = today.replace(hour=0, minute=0, second=0, microsecond=0, nanosecond=0)
tomorrow = today + pd.Timedelta(1, unit='d')

bj_pred = predict('bj', ['PM2.5', 'PM10', 'O3'], bj_stations, tomorrow)
ld_pred = predict('ld', ['PM2.5', 'PM10'], ld_stations, tomorrow)

df = pd.concat([bj_pred, ld_pred])
submit = df.reindex(columns=['test_id', 'PM2.5', 'PM10', 'O3'])
submit.to_csv('results/submit_'+tomorrow.strftime('%m%d')+'.csv', index=None)

