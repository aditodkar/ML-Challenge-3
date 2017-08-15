# impoting libraries
import pandas as pd
import datetime
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

print 'fetching data'
for i in pd.read_csv('train.csv', chunksize = 1500000):
	train = i
	break

##############################################################
print 'dealing with datetime'
train['datetime'] = pd.to_datetime(train['datetime'])
train['day']=train['datetime'].dt.day
train['week_of_year']=train['datetime'].dt.week
train['dayofweek']=train['datetime'].dt.dayofweek
train['time']=train['datetime'].dt.time

# converting datetime to string type
train['datetime']=train.datetime.apply(str)
Night = datetime.datetime.strptime('00:00:00', '%H:%M:%S').time()
Morning = datetime.datetime.strptime('06:00:00', '%H:%M:%S').time()
afternoon = datetime.datetime.strptime('12:00:00', '%H:%M:%S').time()
Evening = datetime.datetime.strptime('18:00:00', '%H:%M:%S').time() 

# creating series to generate state of the day
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

# setting null value of devid to mobile as the mobile is most widely used device for browsing
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

################################################################################3
# performing label encoding to string type
categorical_variables=['datetime', 'countrycode', 'browserid' , 'devid']
le=LabelEncoder()
for var in categorical_variables:
	train[var]=le.fit_transform(train[var])

print train.info()

################################################################################
# fitting the model
model=DecisionTreeClassifier(max_features =10, max_depth=7, criterion='entropy', splitter='best')
model.fit(train[independent_var],train[dependent_var])

del train

#################################################################################################################
# Applying the same convertions to the test dataset
##################################################################################################################

print 'fetching data'
test = pd.read_csv('test.csv')

##############################################################
print 'dealing with datetime'
test['datetime'] = pd.to_datetime(test['datetime'])
test['day']=test['datetime'].dt.day
test['week_of_year']=test['datetime'].dt.week
test['dayofweek']=test['datetime'].dt.dayofweek
test['time']=test['datetime'].dt.time

test['datetime']=test.datetime.apply(str)

Night = datetime.datetime.strptime('00:00:00', '%H:%M:%S').time()
Morning = datetime.datetime.strptime('06:00:00', '%H:%M:%S').time()
afternoon = datetime.datetime.strptime('12:00:00', '%H:%M:%S').time()
Evening = datetime.datetime.strptime('18:00:00', '%H:%M:%S').time() 

test.loc[(test.time >= Morning) & (test.time < afternoon), ['datetime']] = 'M' 
test.loc[(test.time >= afternoon) & (test.time < Evening), ['datetime']] = 'A' 
test.loc[(test.time >= Evening) & (test.time <= datetime.datetime.strptime('23:59:59', '%H:%M:%S').time()), ['datetime']] = 'E' 
test.loc[(test.time >= Night) & (test.time < Morning), ['datetime']] = 'N' 


target=test['ID'].tolist()

lst = [test.time, test.ID]
del lst


###################################################################3
print 'fixing browserid firefox'
test.loc[test.browserid == 'Mozilla', ['browserid']] = 'Firefox' 
test.loc[test.browserid == 'Mozilla Firefox', ['browserid']] = 'Firefox' 
print 'fixing browserid IE'
test.loc[test.browserid == 'Internet Explorer', ['browserid']] = 'IE'
test.loc[test.browserid == 'InternetExplorer', ['browserid']] = 'IE'
print 'fixing browserid Chrome'
test.loc[test.browserid == 'Google Chrome', ['browserid']] = 'Chrome'

###########################################################################
#siteid           99921
#browserid        50162
#devid           149817

print 'dealing with null values'
test['devid']=test['devid'].fillna('Mobile')

def nullBrowserFill(data):
    value=data.mode()[0]
    iex=data.index
    test.browserid[iex]=test.browserid[iex].fillna(value)
    
def nullSiteFill(data):
    value=data.mode()[0]
    iex=data.index
    test.siteid[iex]=test.siteid[iex].fillna(value)
    
pd.pivot_table(test, index=['countrycode', 'devid'], values='browserid', aggfunc=nullBrowserFill)
pd.pivot_table(test, index=['countrycode' ,'datetime', 'devid'], values='siteid', aggfunc=nullSiteFill)

############################################################################
print 'creating independent_var and dependent_var'
independent_var = ['datetime', 'siteid', 'offerid', 'category', 'merchant', 'countrycode', 'browserid', 'devid', 'day', 'week_of_year' ,'dayofweek']

categorical_variables=['datetime', 'countrycode', 'browserid' , 'devid']
le=LabelEncoder()
for var in categorical_variables:
	test[var]=le.fit_transform(test[var])

print test.info()


#################################################################################################################
##################################################################################################################

output=model.predict_proba(test[independent_var])

lst=[test]
del lst

import csv
writer=csv.writer(open(r'submissionchange1.csv','wb'))
writer.writerow(['ID', 'click'])

for i in range(len(target)):
	writer.writerow([target[i], output[i][1]])	
