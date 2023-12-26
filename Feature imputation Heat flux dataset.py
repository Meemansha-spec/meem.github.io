#!/usr/bin/env python
# coding: utf-8

# In[190]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[191]:


train_data = pd.read_csv('Feauture imputation heat flux dataset/data.csv')
train_data.head()


# In[192]:


sample_sub = pd.read_csv('Feauture imputation heat flux dataset/sample_submission.csv')
sample_sub


# In[193]:


train_data.isna().sum()


# In[194]:


sns.boxplot(train_data)


# In[195]:


train_data.columns


# In[196]:


sns.boxplot(train_data['x_e_out [-]'])


# In[197]:


sns.distplot(train_data['x_e_out [-]'])


# ## Fill in the missing values using median

# In[198]:


train_data = train_data.fillna(train_data.median())


# In[199]:


train_data.isna().sum()


# In[200]:


train_data.head()


# In[201]:


train_data['author'].value_counts()


# In[202]:


train_data['author'] =  train_data['author'].fillna(train_data['author'].mode()[0])


# In[203]:


train_data['geometry'] = train_data['geometry'].fillna(train_data['geometry'].mode()[0])


# In[204]:


train_data.isna().sum()


# In[205]:


train_data['geometry'].value_counts


# In[206]:


from sklearn import preprocessing 
  
# label_encoder object knows  
# how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels in column 'species'. 
train_data['geometry']= label_encoder.fit_transform(train_data['geometry']) 
  
train_data['geometry'].unique() 


# In[207]:


train_data.columns


# ## Split the data into training and testing

# In[208]:


X = train_data.drop(columns = ['id','author','x_e_out [-]'])
y = train_data['x_e_out [-]']


# In[209]:


X


# In[210]:


from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y , test_size = 0.4 , random_state = 42)
X_train.shape , X_test.shape , y_train.shape , y_test.shape


# In[211]:


sample_sub.shape


# ## Use Random Forest Regressor  model

# In[236]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

regr = RandomForestRegressor()

# Defining the Hyperparameter Grid
param_grid = {'n_estimators': [100, 200, 300, 400, 500],
              'max_features': ['auto', 'sqrt', 'log2'],
              'max_depth': [10, 20, 30, 40, 50, None],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4],
              'bootstrap': [True, False]}

# Creating Randomized Search CV Object
random_search = RandomizedSearchCV(estimator=regr, param_distributions=param_grid, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)


# In[237]:


random_search.fit(X_train, y_train)

# Printing the Best Parameters
print(random_search.best_params_)


# In[213]:


regr.score(X_test,y_test)


# In[214]:


y_pred = regr.predict(X_test)


# In[215]:


y_pred 


# In[216]:


y_pred = pd.DataFrame({'x_e_out [-]': y_pred})


# In[218]:


y_pred


# In[223]:


sample_sub


# ## Find rmse value

# In[230]:


from sklearn.metrics import mean_squared_error

mean_squared_error(y_test, y_pred, squared=False)


# In[231]:


y_pred


# In[232]:


Kaggle_sub = pd.concat([sample_sub['id'], y_pred[:10415]],axis = 1)


# In[233]:


Kaggle_sub.head()


# In[234]:


Kaggle_sub.to_csv('kaggle_submission.csv', index=False)


# In[ ]:




