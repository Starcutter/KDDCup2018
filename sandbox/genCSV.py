
# coding: utf-8


# In[3]:


from sandbox.rf import *
from utils.dataset import StationInvariantKddDataset
import pandas as pd
import datetime
import sys


# In[28]:


def getDataFrame(city, random_state=0, use_pred=True):
    dataset = torch.load(f'data/dataset_{city}.pt')
    print(len(dataset))
    dataset.T = 7
    stations = dataset.stations
    
    x = np.vstack(
        [np.concatenate(
            [dataset.get_latest_data(idx).aq.flatten(),
             np.mean(dataset.get_latest_data(idx).meo_pred, axis=(2, 3)).flatten()]
        )]
        for idx in range(len(stations)))
    print(x.shape)
    x = np.nan_to_num(x)
    
    model = getRandomForestModel(city, random_state, use_pred)
    y_hat = model.predict(x)
    y_hat = y_hat.reshape(y_hat.shape[0], 24 * 2, -1)
    y_hat = y_hat * dataset.aq_std + dataset.aq_mean

    numStations = len(stations)
    numFeatures = 3 if city == 'bj' else 2
    metrics = ['PM2.5', 'PM10']
    if city == 'bj':
        metrics.append('O3')
    
    y_hat = y_hat.reshape((numStations * 48, numFeatures))
    y_hat[y_hat < 0] = 0
    
    df = pd.DataFrame(y_hat, columns=metrics)
    df.insert(0, 'test_id', [station + '#' + str(i) for station in stations for i in range(48)])
    
    return df


# In[31]:


date_to_forc = pd.Timestamp(datetime.datetime.utcnow().date()) + pd.Timedelta(1, unit='d')
m_d_str = date_to_forc.strftime('%m%d')


# In[34]:


def genCSV(random_state=0):
    bj_df = getDataFrame('bj', random_state)
    ld_df = getDataFrame('ld', random_state)

    df = pd.concat([bj_df, ld_df])
    submission = df.reindex(columns=['test_id', 'PM2.5', 'PM10', 'O3'])
    submission = submission.reset_index(drop=True)

    filename = 'results/submit_'+m_d_str+'_'+str(random_state)+'.csv'
    submission.to_csv(filename, index=None)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        genCSV(int(sys.argv[1]))
    else:
        genCSV(0)

