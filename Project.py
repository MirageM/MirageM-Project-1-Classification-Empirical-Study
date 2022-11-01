#!/usr/bin/env python
# coding: utf-8

# # Group 27
# # Student : Mirage Mohammad
# # Student Number: 300080185

# In[2]:


#Importing required packages.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[17]:


#Loading dataset
dataframe = pd.read_csv('/Users/mirage/Desktop/Assignment/diabetic_data.csv')


# **1. Understand the classification task for your dataset**\
# a. Is it a binary/multi-class classification? \
# b. What is the goal? Is this for a particular application?

# Answer:
# a. This is a multi-class classification because one binary model needs to be trained for each class,
# where this is a one vs all model; Race: Caucasian, Aferican
# b. The goal of this classification is have an understanding of real customer data a long with linking their race, gender, age status
# with their income and their family life

# **2. Analyze your dataset** \
# a. Characterize the dataset in terms of number of training examples, number of \
# features, missing data, etc. 

# In[ ]:





# **3. Brainstorm about the attributes (Feature engineering)** \
# a. Think about the features that could be useful for this task, are they all present in the
# dataset? Anything missing? Any attribute provided that doesn’t seem useful to you?

# In[ ]:





# **4. Encode the features** \
# a. As you will use models that need discrete or continuous attributes, think about data
# encoding and transformation.

# In[ ]:





# **5. Prepare your data for the experiment, using cross-validation**

# In[ ]:





# **6. Train at least these 3 models using some default parameters. You should use ALL the models** \
# below:
# a. Naïve Bayes \
# b. Logistic Regression \
# c. Multi-Layer Perceptron 

# In[ ]:





# **7. Test your 3 models using cross-validation (provided the split in step 5)**

# In[ ]:





# **8. Perform an evaluation with precision/recall measures**

# In[ ]:





# **9. For each type of model, modify some parameters, and perform a train/test/evaluate again.**
# Do this for two times.

# In[ ]:





# **10. Analyze the obtained results** \
# a. Compare quantitatively (with the precision/recall measures) your 9 results. The 9
# results come from 3 models, each with default parameters from step 6 + 2 variations
# from step 9. \
# b. Show some examples of results that are good and not good (false positives and false
# negatives), try to understand why and discuss.

# In[ ]:





# In[18]:


dataframe.head()


# In[9]:


# Naive Bayes Agorithm Test, fit represents training
from sklearn.naive_bayes import MultinomialNB

# Training the model
clf = MultinomialNB().fit(train_counts, train_tags)   

