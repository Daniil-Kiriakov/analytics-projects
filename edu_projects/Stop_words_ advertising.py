#!/usr/bin/env python
# coding: utf-8

# In[157]:


import pandas as pd
import numpy as np
import os
import nltk
from nltk import word_tokenize
import re
from tqdm import tqdm

os.chdir('C:\\Users\\danii\\OneDrive\\Desktop\\Работа')


# In[158]:


df = pd.read_csv('data.csv', sep = ';')
df


# In[159]:


df = df.drop_duplicates(keep = 'first')
df.reset_index(drop=True, inplace=True)
df


# In[160]:


stopw = pd.read_csv('stop words.csv')
stopw
stop_words = list(stopw['Stop words'])
len(set(stop_words))


# In[161]:


len((stop_words))


# In[164]:


get_ipython().run_cell_magic('time', '', "index = []\n\nfor i in range(len(df)):\n    \n    token = word_tokenize(df['Keyword'][i])\n\n    for u in range(len(token)):\n        if token[u] in stop_words:\n            index += [i]")


# In[165]:


index = list(set(index))


# In[166]:


for i in index:
    df = df.drop([i], axis = 0)
df


# In[167]:


df.reset_index(drop=True, inplace=True)


# In[168]:


df['Avg. monthly searches'].isna().sum()
df['Avg. monthly searches'].fillna(0)


# In[169]:


keys = list(df['Keyword'])
pat = r'(заказ)|(созд)|(разраб)'

well_df = pd.DataFrame(columns = ['Keyword', 'Avg. monthly searches', 
                                  'Top of page bid (low range)', 'Top of page bid (high range)'])
bad_df = pd.DataFrame(columns = ['Keyword', 'Avg. monthly searches', 
                                  'Top of page bid (low range)', 'Top of page bid (high range)'])


for i in range(len(keys)):
    well_key = re.findall(pat, keys[i])
    if len(well_key) > 0:
        well_df.loc[i] = list(df.loc[i])
    else:
        bad_df.loc[i] = list(df.loc[i])


# In[170]:


well_df.reset_index(drop=True, inplace=True)
well_df


# In[171]:


bad_df.reset_index(drop=True, inplace=True)
print(bad_df.shape)
bad_df.head()


# In[172]:


well_df.info()


# In[173]:


well_df_del_prodv = well_df

pat_ = r'(продвиже)'

for i in range(len(well_df_del_prodv)):
    prodv = re.findall(pat_, well_df_del_prodv['Keyword'][i])
    if len(prodv) > 0:
        well_df_del_prodv = well_df_del_prodv.drop([i], axis = 0)
        
well_df_del_prodv.reset_index(drop=True, inplace=True)  

for i in range(len(well_df_del_prodv)):
    if float(well_df_del_prodv['Avg. monthly searches'][i]) < float(30):
        well_df_del_prodv = well_df_del_prodv.drop([i], axis = 0)
        
well_df_del_prodv.reset_index(drop=True, inplace=True)
well_df_del_prodv # без слова продвижение и больше 30 показов


# In[174]:


df_split = well_df_del_prodv

pat = r'(заказ)|(созд)|(разраб)'
pat_z = r'(заказ)'
pat_c = r'(созд)'
pat_p = r'(разраб)'

df_zakaz = pd.DataFrame(columns = ['Keyword', 'Avg. monthly searches', 
                                  'Top of page bid (low range)', 'Top of page bid (high range)'])
df_cozdan = pd.DataFrame(columns = ['Keyword', 'Avg. monthly searches', 
                                  'Top of page bid (low range)', 'Top of page bid (high range)'])
df_pazrab = pd.DataFrame(columns = ['Keyword', 'Avg. monthly searches', 
                                  'Top of page bid (low range)', 'Top of page bid (high range)'])


for i in range(len(df_split)):
    word = re.findall(pat_z, df_split['Keyword'][i])
    
    if len(word) > 0:
        df_zakaz.loc[i] = list(df_split.loc[i])
        df_split = df_split.drop([i], axis = 0)
        

df_split.reset_index(drop=True, inplace=True)
df_zakaz.reset_index(drop=True, inplace=True)

for i in range(len(df_split)):
    
    word1 = re.findall(pat_c, df_split['Keyword'][i])
    if len(word1) > 0:
        df_cozdan.loc[i] = list(df_split.loc[i])
        df_split = df_split.drop([i], axis = 0)
        
df_split.reset_index(drop=True, inplace=True)
df_cozdan.reset_index(drop=True, inplace=True)

for i in range(len(df_split)):  
    
    word2 = re.findall(pat_p, df_split['Keyword'][i])
    if len(word2) > 0:
        df_pazrab.loc[i] = list(df_split.loc[i])
        df_split = df_split.drop([i], axis = 0)
        
df_split.reset_index(drop=True, inplace=True)
df_pazrab.reset_index(drop=True, inplace=True)


# In[175]:


df_zakaz


# In[176]:


df_cozdan


# In[177]:


print(df_pazrab.shape)
df_pazrab.head()


# In[178]:


#df_zakaz.to_excel('df_zakaz.xlsx')
#df_cozdan.to_excel('df_cozdan.xlsx')
#df_pazrab.to_excel('df_pazrab.xlsx')


# In[69]:


#df.to_excel('ready_keywords.xlsx')


# In[70]:


#well_df.to_excel('well_df.xlsx')


# In[71]:


#bad_df.to_excel('bad_df.xlsx')

