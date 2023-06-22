#!/usr/bin/env python
# coding: utf-8

# In[6]:


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
from sklearn.linear_model import LogisticRegression
import  numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.datasets import fetch_olivetti_faces
from datetime import datetime
import time
from matplotlib.patches import Rectangle


# In[7]:


faces = fetch_olivetti_faces()
print(faces.DESCR)


# In[8]:


# 40 человек, 10 эмоций
print('Столбцы', list(faces))
print('Кол-во и размеры изображений', faces.images.shape)
print('Кол-во и размеры хар-ки', faces.data.shape)
print('Кол-во и размеры класса', faces.target.shape)


# In[9]:


images = faces.images # 2D
features = faces.data
targets = faces.target


# In[29]:


for i in range(3):
    features[i]
    plt.figure(figsize = (3, 3))
    plt.imshow(images[i])
    plt.show()


# In[31]:


for i in range(3):
    print(features[i])


# In[11]:


def get_metrics(y_test_targets, y_pred_targets):
    print('accuracy_score =', metrics.accuracy_score(y_test_targets, y_pred_targets))
    print('precision_score =', metrics.precision_score(y_test_targets, y_pred_targets, average=None, zero_division=1))
    print('recall_score =', metrics.recall_score(y_test_targets, y_pred_targets, average=None, zero_division=1))


# In[12]:


x_train_features, x_test_features, y_train_targets, y_test_targets = train_test_split(
        features, targets, test_size = 0.2, random_state = 253235, stratify=targets)


# In[13]:


classifier = SVC(kernel="linear")
classifier.fit(x_train_features, y_train_targets)
y_pred_targets = classifier.predict(x_test_features)


# In[14]:


get_metrics(y_test_targets, y_pred_targets)


# In[15]:


# Кривые обучения


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None, n_jobs=None,
        train_sizes=np.linspace(0.1, 1.0, 5),):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))
    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")
    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",)
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",)
    axes[0].plot(train_sizes, train_scores_mean, "o-",
                 color="r", label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, "o-",
                 color="g", label="Cross-validation score")
    axes[0].legend(loc="best")
    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")
    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")
    return plt


# In[16]:


methods = [
           SVC(kernel="linear")
           ]

for i in methods:
    print(i)
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    title = f'{i}'
    plot_learning_curve(i, title, x_train_features, y_train_targets, axes=axes[:, 0])
    plt.show()


# In[17]:


methods = [
           LogisticRegression(solver = 'liblinear'), 
           SVC(kernel = 'linear'), 
           SVC(kernel = 'rbf', gamma = 0.0001), 
           KNeighborsClassifier(n_neighbors = 3), 
           MLPClassifier(solver = 'lbfgs', alpha=0.0001, random_state=1, max_iter=40000)
           ]

x_train_features, x_test_features, y_train_targets, y_test_targets = train_test_split(
        features, targets, test_size = 0.2, random_state = True, stratify=targets)

for i in methods:
    print(i)
    clf = i
    clf.fit(x_train_features, y_train_targets)
    y_pred_targets = clf.predict(x_test_features)
    get_metrics(y_test_targets, y_pred_targets)
    print('\n')


# In[18]:


for i in methods:
    print(i)
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    title = f'{i}'
    plot_learning_curve(i, title, x_train_features, y_train_targets, axes=axes[:, 0])
    plt.show()


# In[19]:


for i in methods:
    print(i)
    start_time = datetime.now()
    clf = i
    clf.fit(x_train_features, y_train_targets)
    y_pred_targets = clf.predict(x_test_features)
    print(datetime.now() - start_time)
    print('\n')


# In[20]:


x_train_features, x_test_features, y_train_targets, y_test_targets = train_test_split(
        features, targets, test_size = 0.2, random_state = True, stratify=targets)

classifier = SVC(kernel="linear")
classifier.fit(x_train_features, y_train_targets)
y_pred_targets = classifier.predict(x_test_features)


# In[21]:


# 20% * 400 samples = 80 max

for i in range (79):
    if y_test_targets[i] != y_pred_targets[i]:
        print('wrong img index =', i)


# In[22]:


y_test_targets


# In[23]:


y_pred_targets


# In[24]:


test_img = []
pred_img = []

for i in y_test_targets:
    test_img.append(images[i])
    
for i in y_pred_targets:
    pred_img.append(images[i])


# In[25]:


def get_img(draw_img, y_test_targets, y_pred_targets, len_img = 15):
    fig = plt.figure(figsize=(10,10))
    
    for i in range(len_img):
        img = fig.add_subplot(5, 6, i+1)
        plt.title(f'{y_test_targets[i]} {y_pred_targets[i]}')
        
        if y_test_targets[i] != y_pred_targets[i]:
            plt.gca().add_patch(Rectangle((0,0), 63, 63, edgecolor='red', facecolor='none', lw=7))
            
        else:
            plt.gca().add_patch(Rectangle((0,0), 63, 63, edgecolor='green', facecolor='none', lw=7))
            
        img.imshow(draw_img[i])


# In[26]:


get_img(test_img, y_test_targets, y_pred_targets)
get_img(pred_img, y_test_targets, y_pred_targets)
plt.show()

