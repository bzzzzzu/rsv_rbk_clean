import os
import requests
import pandas as pd
import time

category_to_link = {
    '5409f11ce063da9c8b588a12': 'rbcfreenews',
    '5409f11ce063da9c8b588a13': 'economics',
    '5409f11ce063da9c8b588a18': 'rbcfreenews',
    '540d5eafcbb20f2524fc0509': 'rbcfreenews',
    '540d5ecacbb20f2524fc050a': 'rbcfreenews',
    '552e430f9a79475dd957f8b3': 'money',
    '5e54e2089a7947f63a801742': 'auto',
    '5e54e22a9a7947f560081ea2': 'realty',
    '5433e5decbb20f277b20eca9': 'rbcfreenews',
    '61f9569a9a794794245a82ab': 'politics',
}

if not os.path.exists('articles/'):
    os.makedirs('articles/')

names = ['train_dataset_train.csv', 'test_dataset_test.csv']

for name in names:
    data = pd.read_csv(name)
    data['publish_date'] = pd.to_datetime(data['publish_date'])

    save_path = str.split(name, '_')[0]
    if not os.path.exists(f'articles/{save_path}/'):
        os.makedirs(f'articles/{save_path}/')

    for i in range(0, data.shape[0]):
        category = category_to_link[data.iloc[i]['category']]
        day = data.iloc[i]['publish_date'].day
        if day < 10:
            day = '0' + str(day)
        month = data.iloc[i]['publish_date'].month
        if month < 10:
            month = '0' + str(month)
        year = data.iloc[i]['publish_date'].year
        code = data.iloc[i]['document_id'][0:24]

        if category == 'rbcfreenews':
            url_str = f'https://www.rbc.ru/{category}/{code}'
        else:
            url_str = f'https://www.rbc.ru/{category}/{day}/{month}/{year}/{code}'

        if not os.path.exists(f'articles/{save_path}/{code}.html'):
            r = requests.get(url_str)
            print(f'i: {i}, response code: {r.status_code}, code: {code}')

            with open(f'articles/{save_path}/{code}.html', 'w', encoding='utf-8') as f:
                f.write(r.text)

            if r.status_code == 200:
                time.sleep(1)
            else:
                print(f'non-200 response for {code}')
                print(f'category: {category}, day: {day}, month: {month}, year: {year}, code: {code}')
                print(f'url_str: {url_str}')
                print(f'title: {data.iloc[i]["title"]}')
                time.sleep(60)
        else:
            print(f'{code} already exists, skipping')
