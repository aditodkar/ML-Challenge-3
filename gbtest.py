from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import datetime
from sklearn.grid_search import GridSearchCV
#from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

print 'fetching data'
for i in pd.read_csv('train.csv', chunksize = 2000000):
	train = i
	break

##############################################################
print 'dealing with datetime'
train['datetime'] = pd.to_datetime(train['datetime'])
train['day']=train['datetime'].dt.day
train['week_of_year']=train['datetime'].dt.week
train['dayofweek']=train['datetime'].dt.dayofweek
train['time']=train['datetime'].dt.time

train['datetime']=train.datetime.apply(str)

Night = datetime.datetime.strptime('00:00:00', '%H:%M:%S').time()
Morning = datetime.datetime.strptime('06:00:00', '%H:%M:%S').time()
afternoon = datetime.datetime.strptime('12:00:00', '%H:%M:%S').time()
Evening = datetime.datetime.strptime('18:00:00', '%H:%M:%S').time() 

train.loc[(train.time >= Morning) & (train.time < afternoon), ['datetime']] = 'M' 
train.loc[(train.time >= afternoon) & (train.time < Evening), ['datetime']] = 'A' 
train.loc[(train.time >= Evening) & (train.time <= datetime.datetime.strptime('23:59:59', '%H:%M:%S').time()), ['datetime']] = 'E' 
train.loc[(train.time >= Night) & (train.time < Morning), ['datetime']] = 'N' 

lst = [train.time, train.ID]
del lst

###################################################################3
print 'fixing browserid firefox'
train.loc[train.browserid == 'Mozilla', ['browserid']] = 'Firefox' 
train.loc[train.browserid == 'Mozilla Firefox', ['browserid']] = 'Firefox' 
print 'fixing browserid IE'
train.loc[train.browserid == 'Internet Explorer', ['browserid']] = 'IE'
train.loc[train.browserid == 'InternetExplorer', ['browserid']] = 'IE'
print 'fixing browserid Chrome'
train.loc[train.browserid == 'Google Chrome', ['browserid']] = 'Chrome'

###########################################################################
#siteid           99921
#browserid        50162
#devid           149817

print 'dealing with null values'
train['devid']=train['devid'].fillna('Mobile')

def nullBrowserFill(data):
    value=data.mode()[0]
    iex=data.index
    train.browserid[iex]=train.browserid[iex].fillna(value)
    
def nullSiteFill(data):
    value=data.mode()[0]
    iex=data.index
    train.siteid[iex]=train.siteid[iex].fillna(value)
    
pd.pivot_table(train, index=['countrycode', 'devid'], values='browserid', aggfunc=nullBrowserFill)
pd.pivot_table(train, index=['countrycode' ,'datetime', 'devid'], values='siteid', aggfunc=nullSiteFill)

############################################################################
print 'creating independent_var and dependent_var'
independent_var = ['datetime', 'siteid', 'offerid', 'category', 'merchant', 'countrycode', 'browserid', 'devid', 'day', 'week_of_year' ,'dayofweek']
dependent_var = 'click'

categorical_variables=['datetime', 'countrycode', 'browserid' , 'devid']
le=LabelEncoder()
for var in categorical_variables:
	train[var]=le.fit_transform(train[var])


train['dayofweek']=train['dayofweek'].astype(np.int16)
train['week_of_year']=train['week_of_year'].astype(np.int32)
train['day']=train['day'].astype(np.uint8)
train['click']=train['click'].astype(np.uint8)
train['browserid']=train['browserid'].astype(np.uint8)
train['datetime']=train['datetime'].astype(np.uint8)
train['siteid']=train['siteid'].astype(np.int32)
print train.info()

X_train, X_test, y_train, y_test = train_test_split(train[independent_var], train[dependent_var],test_size=.2, random_state=42)

del train

print '/ngridsearch cv'
param_test1={'n_estimators':range(300,601,150)}
model=GridSearchCV(estimator=GradientBoostingClassifier(subsample=0.8, min_samples_split=500, min_samples_leaf=50, max_depth=8, max_features='sqrt', random_state=10) , param_grid=param_test1, scoring='roc_auc', n_jobs=2, iid=False,cv=2,verbose=True)
model.fit(X_train, y_train)
print model.best_params_
print model.best_score_

predictions = model.predict(X_test)
print classification_report(y_test, predictions)


###################################################################################################

from numpy import *
import random

def thresholdOut():
	sample_mean = mean([model.predict(x) for x in X_train])
	holdout_mean = mean([model.predict(x) for x in y_train])
	sigma = 1.0 / sqrt(len(X_train))
	threshold = 3.0 * sigma
	if (abs(sample_mean - holdout_mean) < random.normal(threshold, sigma)):
		return sample_mean
	else:
		print 'overfit'
		return holdout_mean * random.normal(0, sigma)

print thresholdOut()
