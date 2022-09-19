# -*- coding: utf-8 -*-
"""HeartStrokeNewTry.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RTxs4BbRN8J6AKIBJgI_GVe0dwLisyoZ
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('/content/healthcare-dataset-stroke-data.csv')

data.head()

data.info()

data['stroke'].value_counts()

data.isna().sum()

data.describe()

data.columns

"""PreProcessing Data"""

from sklearn import preprocessing

lb_enc = preprocessing.LabelEncoder()
print(data['gender'].unique())
data['gender'] = lb_enc.fit_transform(data['gender'])
data['gender'].unique()

print(data['work_type'].unique())
data['work_type'] = lb_enc.fit_transform(data['work_type'])
data['work_type'].unique()

print(data['ever_married'].unique())
data['ever_married'] = lb_enc.fit_transform(data['ever_married'])
data['ever_married'].unique()

data['smoking_status'] = data['smoking_status'].replace(['Unknown'],'never smoked')

print(data['smoking_status'].unique())
data['smoking_status'] = lb_enc.fit_transform(data['smoking_status'])
data['smoking_status'].unique()

print(data['Residence_type'].unique())
data['Residence_type'] = lb_enc.fit_transform(data['Residence_type'])
data['Residence_type'].unique()

data.shape

pd.DataFrame.corr(data)

from scipy.stats import norm
plt.plot(data['bmi'], norm.pdf(data['bmi'], data['bmi'].mean(), data['bmi'].std()),'o')

plt.plot(data['avg_glucose_level'], norm.pdf(data['avg_glucose_level'], data['avg_glucose_level'].mean(), data['avg_glucose_level'].std()),'o')

for i in data.columns:     #df.columns[w:] if you have w column of line description 
    data[i] = data[i].fillna(data[i].median() )
print(data.isnull().any())

data.isna().sum()#data = data.fillna(value=0)

plt.boxplot(data['bmi'])
plt.show()

plt.boxplot(data['avg_glucose_level'])
plt.show()

"""Outlier Treatment"""

plt.boxplot(data['avg_glucose_level'])
plt.show()

print(data['avg_glucose_level'].quantile(0.7))
print(data['avg_glucose_level'].quantile(0.9))
data['avg_glucose_level'] = np.where(data['avg_glucose_level']>data['avg_glucose_level'].quantile(0.8),data['avg_glucose_level'].quantile(0.8),data['avg_glucose_level'])

print(data['bmi'].quantile(0.8))
print(data['bmi'].quantile(0.9))
data['bmi'] = np.where(data['bmi']>data['bmi'].quantile(0.9),data['bmi'].quantile(0.80),data['bmi'])

print(data['bmi'].quantile(0.25))
data['bmi'] = np.where(data['bmi']<data['bmi'].quantile(0.25),data['bmi'].quantile(0.05),data['bmi'])

plt.boxplot(data['bmi'])
plt.show()

data['bmi'].describe()

corr = pd.DataFrame.corr(data)

sns.heatmap(corr,linewidth=1)

corr

data.describe()

list = ['id','ever_married','work_type', 'Residence_type','stroke']

data['age'] = data['age'].astype('int64')

data.info()



X = data.drop(list,axis=1)
y = data['stroke']

print(X.shape)
print(y.shape)

from imblearn.over_sampling import RandomOverSampler
os = RandomOverSampler()

X, y = os.fit_resample(X,y)

print(X.shape)
print(y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=50)

print(X_test)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)

import pickle
pickle.dump(scaler, open('datascaler.pkl','wb'))

print(X_train)
print(X_test)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=100)
lr.fit(X_train,y_train)

import pickle
pickle.dump(lr, open('linearRegTry.pkl','wb'))

loaded_lr = pickle.load(open('/content/linearRegTry.pkl','rb'))
result = loaded_lr.score(X_test, y_test)

print(result)

y_pred = loaded_lr.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

from sklearn.metrics import f1_score

f1_score(y_test, y_pred, labels=None, pos_label=1, average='macro')

from sklearn.svm import SVC

SupportVectorClassModel = SVC() 
SupportVectorClassModel.fit(X_train,y_train)

y_pred_SVC = SupportVectorClassModel.predict(X_test)

accuracySVC = accuracy_score(y_test,y_pred_SVC)*100

accuracySVC

f1_score(y_test, y_pred_SVC, average='macro')

f1_score(y_test, y_pred_SVC,)

f1_score(y_test, y_pred_SVC, average='micro')

trueCount=0
falseCount=0
for i in y_pred_SVC:
  if(i==1):
    trueCount=trueCount+1
  else:
    falseCount=falseCount+1
print(trueCount,"  ",falseCount)

from sklearn.ensemble import RandomForestClassifier
randfor = RandomForestClassifier(n_estimators = 100)
randfor.fit(X_train,y_train)
y_pred_Randfor = randfor.predict(X_test)

for i in X_test:
  print(i)

for i in y_pred_Randfor:
  print(i)

randfor.predict([[ 0.15539343,  0.73462113, -0.47495564,  0.82924542, -0.68761399,  0.01476729,1.72776642]])

accuracyRandfor = (accuracy_score(y_test,y_pred_Randfor))*100

accuracyRandfor

pickle.dump(randfor, open('randfor96.pkl','wb'))

from sklearn.metrics import f1_score
f1_score(y_test, y_pred_Randfor, average='macro')

f1_score(y_test, y_pred_Randfor, average='weighted')

f1_score(y_test, y_pred_Randfor, average='micro')

f1_score(y_test, y_pred_Randfor)

