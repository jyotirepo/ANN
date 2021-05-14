# -*- coding: utf-8 -*-
"""
Created on Fri May 14 12:24:32 2021

@author: jysethy
"""

# Artifical Neural Network pratice programing
# Used a test cases - whether a cusotmer will leav ethe bank or not in shot churn dataset

# Import required libraries

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Importing dataset 

df = pd.read_csv('Churn_Modelling.csv')

# Removal of unsed columsn like 'RowNumber', 'CustomerId', 'Surname' as these are unique fields

df_bank = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# differentiating X and Y datafframe from master dataframe df_bank

X = df_bank.iloc[:, 0:10]
y = df_bank.iloc[:, 10]

# Create Dummy Variables for Geography and Gender

geography = pd.get_dummies(X["Geography"],drop_first=True)
gender = pd.get_dummies(X["Gender"],drop_first=True)


## Concatenate the Data Frames

X = pd.concat([X,geography,gender], axis=1)

## Drop Unnecessary columns from dataframe X
X = X.drop(['Geography','Gender'],axis=1)

#Splitting the datatset into X_train,X_test,y_train,y_test

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


#Feature scaling of daataset using standardscaler

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

## importing libraries to use ANN

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#Initializing Sequential library for ANN

classifier = Sequential()

# Adding the input layer and the first hidden layer for ANN
classifier.add(Dense(units= 6, kernel_initializer= 'he_uniform', activation='relu', input_dim=11 ))

# Adding the second hidden layer
classifier.add(Dense(units= 6, kernel_initializer= 'he_uniform', activation='relu'))

#adding the third output layer
classifier.add(Dense(units= 1, kernel_initializer= 'glorot_uniform', activation='sigmoid'))

#Compiling ANN
classifier.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set
model_history = classifier.fit(X_train,y_train,validation_split=0.33,batch_size= 10, epochs=100)

# list all data in history

print(model_history.history.keys())
# summarize history for accuracy
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Calculate the Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)
print(score)

