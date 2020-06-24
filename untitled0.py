# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 17:14:35 2020

@author: Admin
"""

import pandas as pd
from flask import Flask, render_template, url_for, request
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

df=pd.read_csv('spam.csv',encoding='latin-1')
df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)
#features and labels
df['label']=df['class'].map({'ham':0,'spam':1})
X=df['message']
y=df['label']

#extract feature with countvectorizer
cv=CountVectorizer()
X=cv.fit_transform(X)
pickle.dump(cv, open('transform.pkl','wb'))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=42)

#naive bayes classifier
clf=MultinomialNB()
clf.fit(X_train, y_train)
clf.score(X_test ,y_test)
filename='nlp_model.pkl'
pickle.dump(clf, open(filename,'wb'))
