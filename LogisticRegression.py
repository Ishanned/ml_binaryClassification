#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


df=pd.read_csv('diabetes.csv')


X = df[['Glucose','BloodPressure','Insulin','BMI','Age']] #assigning independent variables to x
y = df['Outcome'] #assigning dependent variable to y

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size =0.25,random_state=0) #to split the data

logistic_regression=LogisticRegression()
logistic_regression.fit(X_train,y_train)
y_pred=logistic_regression.predict(X_test)


from sklearn import metrics
cnf_matrix=metrics.confusion_matrix(y_test,y_pred)
print(cnf_matrix)

import seaborn as sn 
get_ipython().run_line_magic('matplotlib', 'inline')
sn.heatmap(pd.DataFrame(cnf_matrix))
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted')


print("Accuracy:", metrics.accuracy_score(y_test,y_pred))







