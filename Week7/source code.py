#!/usr/bin/env python
# coding: utf-8

# # <b>Data  Science Project
# 

# # <b> Hate  Speech Detection

# <b> Problem Statement: </b>The term hate speech is understood as any type of verbal, written or behavioural communication that attacks or uses derogatory or discriminatory language against a person or group based on what they are, in other words, based on their religion, ethnicity, nationality, race, colour, ancestry, sex or another identity factor. In this problem, We will take you through a hate speech detection model with Machine Learning and Python.
# 
# Hate Speech Detection is generally a task of sentiment classification. So for training, a model that can classify hate speech from a certain piece of text can be achieved by training it on a data that is generally used to classify sentiments. So for the task of hate speech detection model, We will use the Twitter tweets to identify tweets containing  Hate speech.

# In[3]:


## Import Libraries


# In[4]:


import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import re
import nltk
import string
from nltk.text import Text

plt.style.use('fivethirtyeight')
warnings.filterwarnings("ignore")


# In[5]:


## Import Dataset


# In[6]:


df = pd.read_csv("C://Users//Niharika//Downloads//Twitter Hate Speech (1).csv", usecols = ['label', 'tweet'])
df.tail()


# In[ ]:




