# create training set and run classical machine learning methods 

import os
import pandas as pd
import recordlinkage as rl
from recordlinkage.preprocessing import clean

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

# set working directory
os.chdir('C:/Users/yan/Documents/record-linkage/data')

# load data
amazon = pd.read_csv('Amazon_train.csv')
google = pd.read_csv('Google_train.csv')

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

# load features

features = pd.read_csv('features.csv',index_col=[0,1])

# load true matches
ag_matching = pd.read_csv('AG_perfect_matching_train.csv',index_col=0)
df = pd.merge(amazon,ag_matching,how='left')
df = pd.merge(df,google,how='left',left_on='idGoogleBase',right_on='idGoogle')
df = df[['index_x','index_y']].dropna()
matches = df.set_index(['index_x','index_y']).index

# threshold-based methods
predictions = features[features.sum(axis=1)>0.54]
predictions = predictions[predictions.name>0.33]

confusion_matrix = rl.confusion_matrix(matches, predictions, len(features))

# print metrics
print("Reduction Ratio",rl.reduction_ratio(len(predictions),amazon,google))
print("Precision:", rl.precision(confusion_matrix))
print("Recall:", rl.recall(confusion_matrix))
print("F-Measure:", rl.fscore(confusion_matrix))

# data fusion
ag = pd.merge(predictions.reset_index()[['level_0','level_1']],amazon,how='left',left_on='level_0',right_on='index')
ag = pd.merge(ag,google,how='left',left_on='level_1',right_on='index')
ag = ag.set_index(['level_0','level_1'])
df['label']=1
ag = pd.merge(ag,df,'left',left_on=['index_x','index_y'],right_on=['index_x','index_y'])
ag = ag.drop(['index_x','index_y'],axis=1)
ag['label']= ag['label'].fillna(0)
ag.to_csv('ag.csv')


# split training and validation set
train, test = train_test_split(predictions, test_size=0.2,random_state=1)

# Get the true pairs for each set
train_matches_index = train.index & matches
test_matches_index = test.index & matches

# K-means Classifier
kmeans = rl.KMeansClassifier()
result_kmeans = kmeans.learn(train)
predict = kmeans.predict(test)
confusion_matrix = rl.confusion_matrix(test_matches_index, predict, len(test))
print("Precision:", rl.precision(confusion_matrix))
print("Recall:", rl.recall(confusion_matrix))
print("F-Measure:", rl.fscore(confusion_matrix))
#Precision: 0.05838
#Recall: 0.23232
#F-Measure: 0.09331

# Logistic Regression
classifier = rl.LogisticRegressionClassifier()
classifier.learn(train, train_matches_index)
predict = classifier.predict(test)
confusion_matrix = rl.confusion_matrix(test_matches_index, predict, len(test))
print("Precision:", rl.precision(confusion_matrix))
print("Recall:", rl.recall(confusion_matrix))
print("F-Measure:", rl.fscore(confusion_matrix))
#Precision: 0.53191
#Recall: 0.25252
#F-Measure: 0.34247

y_score = classifier.prob(test)
test_copy = pd.merge(test,df,how='left',left_index=True,right_on=['index_x','index_y'])
test_copy['label']=test_copy['label'].fillna(0)
y_test = test_copy['label']

fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

# plot ROC graph

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Logistic Regression')
plt.legend(loc="lower right")
plt.show()

# run logistic regression classifer on final test set

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

# load true
ag_matching = pd.read_csv('AG_perfect_matching_test.csv',index_col=0)
df = pd.merge(amazon,ag_matching,how='left')
df = pd.merge(df,google,how='left',left_on='idGoogleBase',right_on='idGoogle')
df = df[['index_x','index_y']].dropna()
df['label'] = 1 
matches = df.set_index(['index_x','index_y']).index
test_matches_index = test.index & matches

test_final = pd.read_csv('features_test.csv',index_col=[0,1])
predict = classifier.predict(test_final)
confusion_matrix = rl.confusion_matrix(test_matches_index, predict, len(test_final))

# Print Metrics
print("Precision:", rl.precision(confusion_matrix))
print("Recall:", rl.recall(confusion_matrix))
print("F-Measure:", rl.fscore(confusion_matrix))

y_score = classifier.prob(test_final)
test_copy = pd.merge(test_final,df,how='left',left_index=True,right_on=['index_x','index_y'])
test_copy['label'] = test_copy['label'].fillna(0)
y_test = test_copy['label']

fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

# plot ROC graph

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Logistic Regression')
plt.legend(loc="lower right")
plt.show()