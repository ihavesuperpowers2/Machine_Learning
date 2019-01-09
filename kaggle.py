# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 01:33:41 2019

@author: Admin
"""

import numpy as np
import seaborn as sns
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt

# Importing the dataset
train = pd.read_csv('train_titanic.csv')
test = pd.read_csv('test.csv')

train.head()
display(train.head())
display(train.isnull().sum())
display(test.isnull().sum())
#baarchart for categorical features

def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))
bar_chart('Sex')

bar_chart('Pclass')
train_test_data = [train, test] # combining train and test dataset

for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    
train.drop('Name', axis= 1, inplace= True)
test.drop('Name', axis= 1, inplace= True)

title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    
sex_mapping= {"male": 0, "female": 1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)
    
train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace= True)
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace= True)



facet = sns.FacetGrid(train, hue="Survived",aspect= 4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
 
plt.show()

#Binning
for dataset in train_test_data:
    dataset.loc[ dataset["Age"] <= 16, "Age"] = 0,
    dataset.loc[ (dataset["Age"] > 16) & (dataset["Age"] <= 26), "Age"] = 1
    
    dataset.loc[ (dataset["Age"] > 26) & (dataset["Age"] <= 36), "Age"] = 2
    dataset.loc[ (dataset["Age"] > 36) & (dataset["Age"] <= 62), "Age"] = 3
    dataset.loc[ dataset["Age"] > 62, "Age"] = 4
    
Pclass1 = train[train['Pclass'] == 1]["Embarked"].value_counts()
Pclass2 = train[train['Pclass'] == 2]["Embarked"].value_counts()
Pclass3 = train[train['Pclass'] == 3]["Embarked"].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass2])
df.index = ['1st class', '2nd class', '3rd class']
df.plot(kind= 'bar', stacked= True, figsize= (10, 5))

for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna("S")
    

train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace= True)
test["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace= True)

facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
 
plt.show()

facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.xlim(0, 20)
plt.show()


    

