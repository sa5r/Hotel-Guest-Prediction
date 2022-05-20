#!/usr/bin/env python
# coding: utf-8

# ## Predicting Hotel Guest Cancellation
# ### Sakher Alqa. - UGA.EDU

# In[5]:


import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt


# In[6]:


df = pd.read_csv('hotel_bookings.csv')
df


# In[32]:


#more clarifications on values
for col in df.columns:
    print(f'feature: {col}')
    df_g = df.groupby(col)
    s = str(df_g.size()).split('\n')
    #printing first 5 lines
    for x in range(0,4):
        print(s[x])
    print('---')


# In[8]:


#dropping unnecessary features
df = df.drop(columns=[
'lead_time',
'arrival_date_year',
'arrival_date_month',
'arrival_date_week_number',
'arrival_date_day_of_month',
'adr',
'reservation_status_date',
])
df


# In[9]:


#y is imbalanced, class is reservation_status
#Canceled     43017, Check-Out    75166, No-Show       1207
#undersampling
Canceled_df = df[ df['reservation_status'] == 'Canceled' ]
Canceled_df = Canceled_df.sample(n = 1200)
Check_Out_df = df[ df['reservation_status'] == 'Check-Out' ]
Check_Out_df = Check_Out_df.sample(n = 1200)
df = df.drop( df[ df['reservation_status'] == 'Canceled' ].index )
df = df.drop( df[ df['reservation_status'] == 'Check-Out'].index )
df = df.append(Canceled_df)
df = df.append(Check_Out_df)


# In[10]:


#y is imbalanced, class is reservation_status
#Canceled     43017, Check-Out    75166, No-Show       1207
#copy No-Show 3 times , Oversampling
#df = df.append([ df[ df['reservation_status'] == 'No-Show' ] ] * 40 )
df_g = df.groupby('reservation_status')
print(df_g.size())


# In[11]:


df_g.size().plot.bar()


# In[12]:


#extract y
y = df['reservation_status']
df = df.drop(columns=['reservation_status'])


# In[13]:


#encoding categorical features
enc = preprocessing.OrdinalEncoder()
X = enc.fit_transform(df,y)
enc = preprocessing.LabelEncoder()
y = enc.fit_transform(y)
y


# In[14]:


#Impute
imputer = KNNImputer()
#imputer = SimpleImputer()
X = imputer.fit_transform(X)


# In[15]:


#normalize
X = preprocessing.normalize(X)
X


# In[16]:


#extracting testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[17]:


#training
pipe = Pipeline(
    steps=[("svm", svm.SVC())]
)
param_grid = {
    "svm__C": [1.0],
}
search = GridSearchCV(pipe, param_grid, cv=10, verbose=3)
fit = search.fit(X_train, y_train)


# In[27]:


print(f'Accuracy {fit.score(X_test,y_test)}')

