# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 03:38:46 2019

@author: Admin
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.metrics import accuracy_score

df_train = pd.read_csv('train_titanic.csv')
df_test = pd.read_csv('test.csv')


sns.countplot(x= 'Survived', data = df_train)

df_test['Survived'] = 0
df_test[['PassengerId', 'Survived']].to_csv('no_survivors.csv', index=False)

#count plot. x= Different Attribute values. y= No of those values
sns.countplot(x= 'Pclass', data= df_train)
sns.countplot(x= 'Sex', data= df_train)
#different countplot. Comparing two attributes.
sns.factorplot(x= 'Survived', col= 'Sex', kind= 'count', data= df_train);

print((df_train[df_train.Sex == 'female'].Survived.sum()/df_train[df_train.Sex== 'female'].Survived.count()))
print((df_train[df_train.Sex == 'male'].Survived.sum()/df_train[df_train.Sex== 'male'].Survived.count()))
df_test['Survived'] = (df_test.Sex == 'female')
df_test['Survived'] = df_test.Survived.apply(lambda x: int (x))

df_test[['PassengerId', 'Survived']].to_csv('women_survive.csv', index=False)