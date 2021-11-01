#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 04:34:49 2021

@author: chihongiong
"""

import numpy as np
from collections import Counter
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

Xtrain_ini = np.load('Xtrain_Classification_Part1.npy')
Ytrain_ini = np.load('Ytrain_Classification_Part1.npy')
Xtest = np.load('Xtest_Classification_Part1.npy')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
Xtrain, X_valid, Ytrain, Y_valid = train_test_split(Xtrain_ini, Ytrain_ini, test_size = 0.3, random_state = 0)

#Logistic Regression
lr = LogisticRegression()
lr.fit(Xtrain, Ytrain)
y_pred_lr = lr.predict(X_valid)
lr_score = lr.score(X_valid, Y_valid)
print('score for LogisticRegression: ', lr_score)
print('Accuracy: ', metrics.accuracy_score(Y_valid, y_pred_lr))
confusion_matrix_lr = confusion_matrix(Y_valid, y_pred_lr)
print('LogisticRegression confusion matrix: ')
print(confusion_matrix_lr)
print(classification_report(Y_valid, y_pred_lr))

#KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(Xtrain, Ytrain)
y_pred_knn = knn_model.predict(X_valid)
knn_score = knn_model.score(X_valid, Y_valid)
print('score for KNeighborsRegressor: ', knn_score)
print('Accuracy: ', metrics.accuracy_score(Y_valid, y_pred_knn))
confusion_matrix_knn = confusion_matrix(Y_valid, y_pred_knn)
print('KNeighborsRegressor confusion matrix: ')
print(confusion_matrix_knn)
print(classification_report(Y_valid, y_pred_knn))

#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
Dtree = DecisionTreeClassifier()
Dtree.fit(Xtrain, Ytrain)
y_pred_Dtree = Dtree.predict(X_valid)
Dtree_score = Dtree.score(X_valid, Y_valid)
print('score for DecisionTreeClassifier: ', Dtree_score)
print('Accuracy: ', metrics.accuracy_score(Y_valid, y_pred_Dtree))
confusion_matrix_Dtree = confusion_matrix(Y_valid, y_pred_Dtree)
print('DecisionTreeClassifier confusion matrix: ')
print(confusion_matrix_Dtree)
print(classification_report(Y_valid, y_pred_Dtree))

#Support Vector Machine
from sklearn import svm
SVM = svm.SVC()
SVM.fit(Xtrain, Ytrain)
y_pred_svm = SVM.predict(X_valid)
svm_score = SVM.score(X_valid, Y_valid)
print('score for Supporter Vector Machine: ', svm_score)
print('Accuracy: ', metrics.accuracy_score(Y_valid, y_pred_svm))
confusion_matrix_svm = confusion_matrix(Y_valid, y_pred_svm)
print('Support Vector Machine confusion matrix: ')
print(confusion_matrix_svm)
print(classification_report(Y_valid, y_pred_svm))

#Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(Xtrain, Ytrain)
y_pred_nb = nb.predict(X_valid)
nb_score = nb.score(X_valid, Y_valid)
print('score for Gaussian Naive Bayes: ', nb_score)
print('Accuracy: ', metrics.accuracy_score(Y_valid, y_pred_nb))
confusion_matrix_nb = confusion_matrix(Y_valid, y_pred_nb)
print('Gaussian Naive Bayes confusion matrix')
print(confusion_matrix_nb)
print(classification_report(Y_valid, y_pred_nb))





















