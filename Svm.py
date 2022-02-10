#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
# Import svm classifier
from sklearn import svm
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import scikit learn metrics module for accuracy calculation
from sklearn import metrics

df=pd.read_csv('diabetes.csv')
X = df[['Glucose','BloodPressure','Insulin','BMI','Age']] #assigning independent variables to x
y = df['Outcome'] #assigning dependent variable to y

#splitting data as test data and train data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size =0.25,random_state=0)

# Create svm classifier object
clf = svm.SVC(kernel='linear')
# Train the model
clf = clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)

#confusion matrix
cnf_matrix=metrics.confusion_matrix(y_test,y_pred)
print(cnf_matrix)


import matplotlib.pyplot as plt
import seaborn as sn 
get_ipython().run_line_magic('matplotlib', 'inline')
sn.heatmap(pd.DataFrame(cnf_matrix))
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted')

#Accuracy
print("Accuracy:", metrics.accuracy_score(y_test,y_pred))


# In[ ]:




