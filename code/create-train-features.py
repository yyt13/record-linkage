# clean data and run a full comparison

#import os
#os.chdir('C:/Users/yan/Documents/record-linkage/data')

import pandas as pd
import recordlinkage as rl
from recordlinkage.preprocessing import clean

# load data
amazon = pd.read_csv('Amazon_train.csv',index_col=0)
google = pd.read_csv('Google_train.csv',index_col=0)

# clean data
amazon['title'] = clean(amazon['title'])
amazon['description'].update(clean(amazon['description']))
amazon['manufacturer'].update(clean(amazon['manufacturer']))
amazon.columns = ['idAmazon','name','description','manufacturer','price']
amazon.set_index('idAmazon')

google['name'].update(clean(google['name']))
google['description'].update(clean(google['description']))
google['manufacturer'].update(clean(google['manufacturer']))
google['price'].update(clean(google['price'],replace_by_none='[^ \\-\\_0-9.]+'))
google['price'] = pd.to_numeric(google['price'])
google.columns = ['idGoogle','name','description','manufacturer','price']
google.set_index('idGoogle')


# run a full comparison
indexer = rl.FullIndex()
candidates = indexer.index(amazon,google)

compare = rl.Compare()
compare.string('name','name',method='cosine',label='name')
compare.string('description','description',method='qgram',label='description',)
compare.string('manufacturer','manufacturer',method='cosine',label='manufacturer')
compare.numeric('price','price',label='price')

features = compare.compute(candidates,amazon,google)

# save features

features.to_csv('features.csv')
