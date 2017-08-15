import pandas as pd

train1=pd.read_csv('submissionchange1.csv')

train2=pd.read_csv('GB2submissionchange.csv')

import csv
writer=csv.writer(open(r'DT_GB_Merge_Submission.csv','wb'))
writer.writerow(['ID', 'click'])

target=train1['ID'].tolist()
train1 = train1['click'].tolist()
train2 = train2['click'].tolist()
for i in range(len(target)):
	writer.writerow([target[i], (train1[i] + train2[i])/2.0])	


