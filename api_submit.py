
# coding: utf-8

import requests

files={'files': open('submit.csv','rb')}

data = {
    "user_id": "meta_z",   #user_id is your username which can be found on the top-right corner on our website when you logged in.
    "team_token": "65355b59c393aec8e71eaa3b1cdb70a5692e8a61816a37cde79a2bed49bba644", #your team_token.
    "description": 'submission 2018-05-01',  #no more than 40 chars.
    "filename": "submit.csv", #your filename
}

url = 'https://biendata.com/competition/kdd_2018_submit/'

response = requests.post(url, files=files, data=data)

print(response.text)


