#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import (StandardScaler, PolynomialFeatures)
from sklearn import metrics


# In[2]:


data_y = pd.read_csv('9_y.csv', names=['Y'])
data_x = pd.read_csv('9_x.csv', names=['X1', 'X2', 'X3', 'X4', 'X5'])
data_x


# In[3]:


data_y.describe(), data_x.describe()


# In[4]:


data_xy = data_y.join(data_x)
data_xy


# In[5]:


data_train, data_test = np.split(data_xy, [int(.8*len(data_xy))])
data_train.shape, data_test.shape


# In[6]:


data_train


# In[18]:


def get_linreg(x, y):
    x, y = np.array(x).reshape(-1, 1), np.array(y)
    regressor = LinearRegression()
    regressor.fit(x, y)
    print(' r2 =', regressor.score(x, y))
    print(' Средняя абсолютная ошибка =', metrics.mean_absolute_error(y, regressor.predict(x)))
    print(' MSE =', metrics.mean_squared_error(y, regressor.predict(x)))
    print(' Root Mean Squared =', np.sqrt(metrics.mean_squared_error(y, regressor.predict(x))))
    
    plt.figure(figsize = (8, 5))
    plt.scatter(x, y, alpha=0.7)
    plt.plot(x, regressor.predict(x), 'r')
    plt.show()
    
    return regressor.score(x, y)


# In[19]:


for i in range(1, 6):
    print('\n', f'Парная регрессионная модель Y и X{i}', '\n')
    get_linreg(data_train[f'X{i}'], data_train['Y'])


# In[9]:


# Множественная регрессия


# In[10]:


X, y = np.array(data_x), np.array(data_y)
print(X.shape, y.shape)

regressor = LinearRegression()
regressor.fit(X, y)

print('score =', regressor.score(X, y))
print('intercept_ =', regressor.intercept_[0])

pd.DataFrame(data = regressor.coef_[0], index = data_x.columns, columns=['regressor.coef_'])


# In[11]:


# Дополнительные задания


# In[12]:


def get_polynreg(x, y, deg):
    X, y = np.array(x).reshape(-1, 1), np.array(y)
    poly = PolynomialFeatures(degree = deg)
    
    X_poly = poly.fit_transform(X)
    scaler = StandardScaler()
    X_scaler = scaler.fit_transform(X_poly)
    
    regressor = LinearRegression()
    regressor.fit(X_scaler, y)
    
    print(' score =', regressor.score(X_scaler, y))
    
    return(regressor.score(X_scaler, y))  


# In[13]:


for i in range(1, 6):
    print('\n', f'Полиномиальная регрессия Y и X{i}')
    for j in [2, 3, 10]:
        get_polynreg(data_train[f'X{i}'], data_train['Y'], j)


# In[14]:


res1 = []
res2 = []

for i in range(1, 6):
    res1.append(get_linreg(data_train[f'X{i}'], data_train['Y']))
    
for i in range(1, 6):
    for j in [2, 3, 10]:
        res2.append(get_polynreg(data_train[f'X{i}'], data_train['Y'], j))


# In[15]:


table = pd.DataFrame(columns=['X1', 'X2', 'X3', 'X4', 'X5'])

for i in range(1, 6):
    for k in range(0, 15, 3):
        table[f'X{i}'] = res2[k:k+3]
        
table.loc[3] = res1
table['kind of r2'] = ['r2 polym reg degree = 2', 'r2 polym reg degree = 3', 'r2 polym reg degree = 10', 'r2 lin reg']

table.set_index('kind of r2',inplace=True)
table


# In[26]:


datanew_y = pd.read_csv('1_y.csv', names=['Y'])[:len(data_x)]
datanew_y.shape


# In[31]:


# Новые Y1, Y2, Y3
get_ipython().run_line_magic('time', '')

for i in range(1, 4):
    datanew_y = pd.read_csv(f'{i}_y.csv', names=['Y'])[:len(data_x)]
    
    for i in range(1, 6):
        print('\n', f'Парная регрессионная модель Y_new и X{i}', '\n')
        get_linreg(data_x[f'X{i}'], datanew_y['Y'])

