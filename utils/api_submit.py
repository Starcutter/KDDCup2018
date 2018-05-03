
# coding: utf-8

import requests

def submit(filename):
    files = {'files': open(filename, 'rb')}

    data = {
        # user_id is your username which can be found on the top-right corner on our website when you logged in.
        "user_id": "meta_z",
        # your team_token.
        "team_token": "65355b59c393aec8e71eaa3b1cdb70a5692e8a61816a37cde79a2bed49bba644",
        "description": filename.split('/')[1],  # no more than 40 chars.
        "filename": filename,  # your filename
    }

    url = 'https://biendata.com/competition/kdd_2018_submit/'

    response = requests.post(url, files=files, data=data)

    print(response.text)

if __name__ == '__main__':
    submit('results/submit_0503.csv')
