# clean data and create a test set

import os
os.chdir('C:/Users/yan/Documents/record-linkage/data')

# load data
import pandas as pd
amazon = pd.read_csv('Amazon_test.csv')
google = pd.read_csv('Google_test.csv')

from recordlinkage.preprocessing import clean

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

import recordlinkage as rl
indexer = rl.FullIndex()
candidates = pd.DataFrame(indexer.index(amazon,google).to_flat_index().to_list(),
                          columns=['index_amazon','index_google'])
ag_test = pd.merge(candidates,amazon,how='left',
                   left_on='index_amazon',right_on='index')
ag_test = pd.merge(ag_test,google,how='left',
                   left_on='index_google',right_on='index')
ag_test = ag_test.set_index(['index_amazon','index_google'])
ag_test = ag_test.drop(['index_x','index_y'],axis=1)
# load true matches
match = pd.read_csv('AG_perfect_matching_test.csv',index_col=0)
match['label']=1

ag_test = pd.merge(ag_test,match,how='left',
                   left_on=['idAmazon','idGoogle'],right_on=['idAmazon','idGoogleBase'])
ag_test= ag_test.drop('idGoogleBase',axis=1)
ag_test['label']= ag_test['label'].fillna(0)

ag_test.to_csv('ag_test.csv')