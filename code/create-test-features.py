# create test features
import os
os.chdir('C:/Users/yan/Documents/record-linkage/data')

import pandas as pd
import recordlinkage as rl
from recordlinkage.preprocessing import clean

# load data
amazon = pd.read_csv('Amazon_test.csv')
google = pd.read_csv('Google_test.csv')



# clean data
amazon['title'] = clean(amazon['title'])
amazon['description'].update(clean(amazon['description']))
amazon['manufacturer'].update(clean(amazon['manufacturer']))
amazon.columns = ['index','idAmazon','name','description','manufacturer','price']
amazon.set_index('idAmazon')

google['name'].update(clean(google['name']))
google['description'].update(clean(google['description']))
google['manufacturer'].update(clean(google['manufacturer']))
google['price'].update(clean(google['price'],replace_by_none='[^ \\-\\_0-9.]+'))
google['price'] = pd.to_numeric(google['price'])
google.columns = ['index','idGoogle','name','description','manufacturer','price']
google.set_index('idGoogle')

indexer = rl.FullIndex()

candidates = indexer.index(amazon,google)

compare = rl.Compare()
compare.string('name','name',method='cosine',label='name')
compare.string('description','description',method='qgram',label='description',)
compare.string('manufacturer','manufacturer',method='cosine',label='manufacturer')
compare.numeric('price','price',label='price')

features = compare.compute(candidates,amazon,google)

predictions = features[features.sum(axis=1)>0.54]
predictions = predictions[predictions.name>0.33]

predictions.to_csv('features_test.csv')