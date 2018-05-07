'''
Configuration of dataset and algorithms.
'''
import pandas as pd
from datetime import datetime, timedelta

# Constants
START_DATE = pd.Timestamp("2017-01-01 00:00:00")
TODAY = pd.Timestamp(datetime.utcnow().strftime("%Y-%m-%d 00:00:00"))
YESTERDAY = TODAY + timedelta(days=-1)
TOMORROW = TODAY + timedelta(days=1)
MODEL_SAVED_PATH = 'results/model_{}.pkl'

# Algorithms
SEED = 0
# Will use all data before this day to get the model, including (train, valid)
TRAIN_BEFORE_DATE = pd.Timestamp('2018-05-04')
# Will evaluate and get the csv on this day
PRED_ON_DATE = pd.Timestamp('2018-05-04')
# Shall set both above to TOMORROW before a submission (after 7:00AM China time)
# And set it to TODAY to see the score we got a day before (after 7:00AM China time)

# Data
CITY = 'bj'
USE_PAST_T_DAYS = 7
PRED_FUTURE_T_DAYS = 2
USE_MEO_PRED = True
