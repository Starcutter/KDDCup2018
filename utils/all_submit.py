from utils.api_submit import submit
import pandas as pd
import datetime
import os

date_to_forc = pd.Timestamp(datetime.datetime.utcnow().date())
m_d_str = date_to_forc.strftime('%m%d')

filenames = [filename for filename in os.listdir('results/') if m_d_str in filename]
print(filenames)
input('Press enter to confirm to submit:')

for filename in filenames:
    submit('results/' + filename)
