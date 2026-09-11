# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 11:28:00 2024

@author: Admin
"""

#Deploying machine learning model using FLASK
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

#Read data to build a regression model
df=pd.read_csv(r"D:\ODD 26-27\TY-EXTC-ML\ML_LAB\ML EXP7\modeldeploy_localhost\files_for_training_model\Salary_Data.csv")

#Separate predictors and response variables
X = df.iloc[:, :1]

y=df.iloc[:, -1]

#Define regression model object
regressor = LinearRegression()

#Fit model with  data
regressor.fit(X.values, y)

#print model parameters
print("Model parameters are:\n")
print("Intercept,b0,is:", regressor.intercept_)
print("Slope,b1,is:",regressor.coef_)

#To print path where pickle file will be saved
import os

#Get path of working directory
os.getcwd()

# Saving model to current directory
pickle.dump(regressor, open('reg_model.pkl','wb'))


# Loading model to compare the results
ourmodel = pickle.load(open('reg_model.pkl','rb'))

print("Predicted value:",ourmodel.predict([[1.2]]))

