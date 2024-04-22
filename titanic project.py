# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:10:48 2024

@author: DELL
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

##Reading the data and cleaning
train_titanic = pd.read_csv(r"C:\Users\DELL\AppData\Local\Temp\Rar$DIa0.395\train.csv")
train_titanic.head()
train_titanic.info()
print(train_titanic.columns)
print(train_titanic.isnull())
print(train_titanic.isnull().sum())
train_titanic = train_titanic.drop(columns="Cabin", axis=1)
train_titanic['Age'].fillna(train_titanic['Age'].mean(), inplace = True)
train_titanic['Embarked'].fillna(train_titanic['Embarked'].mode()[0],inplace = True)

##EDA
sns.pairplot(train_titanic)
sns.heatmap(train_titanic.corr(),annot=True,cmap="prism")
sns.stripplot(x='Embarked',y='Age',data=train_titanic,hue='Survived')
sns.swarmplot(x='Embarked',y='Age',data=train_titanic,hue='Age')
sns.barplot(x='Sex',y='Survived',data=train_titanic,palette ='plasma')
sns.countplot(x ='Sex', data = train_titanic)  
sns.boxplot(data=train_titanic,x='Survived',y='Age',hue='Sex')
sns.violinplot(data=train_titanic,x='Survived',y='Age',hue='Sex',split=True)
plt.scatter(x='Pclass', y='SibSp',data=train_titanic,c = 'red')

## Logistic regression
train_titanic.replace({'Sex':{'male':0,'female':1},'Embarked':{'C':0,'Q':1,'S':2}},inplace=True)
print(train_titanic.head())
X = train_titanic.drop(columns=['PassengerId', 'Survived','Name','Ticket'])
print(X)
Y = train_titanic['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train,Y_train)
from sklearn.metrics import accuracy_score
X_train_pred = logmodel.predict(X_train)
train_accuracy = accuracy_score(Y_train,X_train_pred)
print("Accuracy score of training data :",train_accuracy)
X_test_pred = logmodel.predict(X_test)
test_accuracy = accuracy_score(Y_test,X_test_pred)
print("Accuracy score of testing data :",test_accuracy)


## TESTING THE MODEL
test_titanic = pd.read_csv(r"C:\Users\DELL\AppData\Local\Temp\a4cde0e2-c04f-43b2-8acc-c3035bfe11ec_titanic.zip.1ec\test.csv") 
test_titanic = test_titanic.drop(columns="Cabin", axis=1)
test_titanic['Age'].fillna(test_titanic['Age'].mean(), inplace = True)
test_titanic['Embarked'].fillna(test_titanic['Embarked'].mode()[0],inplace = True)
test_titanic.replace({'Sex':{'male':0,'female':1},'Embarked':{'C':0,'Q':1,'S':2}},inplace=True)
test_X=test_titanic.drop(columns=['PassengerId','Name','Ticket'])
test_X=test_X.dropna()
print(test_X.isna().sum())
print(test_X.columns)
print(test_X)
actual_pred = logmodel.predict(test_X)
print(actual_pred)


##
# from sklearn.svm import SVC
# train_titanic.replace({'Sex':{'male':0,'female':1},'Embarked':{'C':0,'Q':1,'S':2}},inplace=True)
# print(train_titanic.head())
# X = train_titanic.drop(columns=['PassengerId', 'Survived','Name','Ticket'])
# Y = train_titanic['Survived']
# from sklearn.model_selection import train_test_split
# X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=101)
# classifier = SVC()
# classifier.fit(X_train, Y_train)
# from sklearn.metrics import classification_report, confusion_matrix
# Y_pred=model.predict(X_test)
# confusion_matrix(Y_test, Y_pred)








