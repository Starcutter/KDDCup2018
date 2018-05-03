from utils.api_submit import submit
import pandas as pd

date_to_forc = pd.Timestamp(datetime.datetime.utcnow().date()) + pd.Timedelta(1, unit='d')
m_d_str = date_to_forc.strftime('%m%d')

files = os.listdir('results/')
for file in files:
    if m_d_str in file:
        submit('results/' + file)