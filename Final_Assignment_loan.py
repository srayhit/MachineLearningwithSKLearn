#!/usr/bin/env python
# coding: utf-8

# <a href="https://www.bigdatauniversity.com"><img src="https://ibm.box.com/shared/static/cw2c7r3o20w9zn8gkecaeyjhgw3xdgbj.png" width="400" align="center"></a>
# 
# <h1 align="center"><font size="5">Classification with Python</font></h1>

# In this notebook we try to practice all the classification algorithms that we learned in this course.
# 
# We load a dataset using Pandas library, and apply the following algorithms, and find the best one for this specific dataset by accuracy evaluation methods.
# 
# Lets first load required libraries:

# In[1]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# ### About dataset

# This dataset is about past loans. The __Loan_train.csv__ data set includes details of 346 customers whose loan are already paid off or defaulted. It includes following fields:
# 
# | Field          | Description                                                                           |
# |----------------|---------------------------------------------------------------------------------------|
# | Loan_status    | Whether a loan is paid off on in collection                                           |
# | Principal      | Basic principal loan amount at the                                                    |
# | Terms          | Origination terms which can be weekly (7 days), biweekly, and monthly payoff schedule |
# | Effective_date | When the loan got originated and took effects                                         |
# | Due_date       | Since it’s one-time payoff schedule, each loan has one single due date                |
# | Age            | Age of applicant                                                                      |
# | Education      | Education of applicant                                                                |
# | Gender         | The gender of applicant                                                               |

# Lets download the dataset

# In[2]:


get_ipython().system('wget -O loan_train.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv')


# ### Load Data From CSV File  

# In[4]:


df = pd.read_csv('loan_train.csv')
df.head()


# In[5]:


df.shape


# ### Convert to date time object 

# In[6]:


df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()


# # Data visualization and pre-processing
# 
# 

# Let’s see how many of each class is in our data set 

# In[7]:


df['loan_status'].value_counts()


# 260 people have paid off the loan on time while 86 have gone into collection 
# 

# Lets plot some columns to underestand data better:

# In[ ]:


# notice: installing seaborn might takes a few minutes
get_ipython().system('conda install -c anaconda seaborn -y')


# In[8]:


import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[9]:


bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# # Pre-processing:  Feature selection/extraction

# ### Lets look at the day of the week people get the loan 

# In[10]:


df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


# We see that people who get the loan at the end of the week dont pay it off, so lets use Feature binarization to set a threshold values less then day 4 

# In[11]:


df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()


# ## Convert Categorical features to numerical values

# Lets look at gender:

# In[12]:


df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# 86 % of female pay there loans while only 73 % of males pay there loan
# 

# Lets convert male to 0 and female to 1:
# 

# In[13]:


df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()


# ## One Hot Encoding  
# #### How about education?

# In[14]:


df.groupby(['education'])['loan_status'].value_counts(normalize=True)


# #### Feature befor One Hot Encoding

# In[15]:


df[['Principal','terms','age','Gender','education']].head()


# #### Use one hot encoding technique to conver categorical varables to binary variables and append them to the feature Data Frame 

# In[16]:


Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
#Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()


# ### Feature selection

# Lets defind feature sets, X:

# In[17]:


X = Feature
X[0:5]
print(X.shape)


# What are our lables?

# In[18]:


y = df['loan_status'].values
y = df['loan_status'].replace(to_replace=['PAIDOFF','COLLECTION'], value=[0,1]).values
y[0:5]


# ## Normalize Data 

# Data Standardization give data zero mean and unit variance (technically should be done after train test split )

# In[19]:


X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]
print(X.shape)


# In[21]:


# Split data into train and test set

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
#X_train, X_test, y


# In[22]:


print(np.shape(X_train))
print(np.shape(X_test))
print(np.shape(y_train))
print(np.shape(y_test))
#print(X_test[1])


# # Classification 

# Now, it is your turn, use the training set to build an accurate model. Then use the test set to report the accuracy of the model
# You should use the following algorithm:
# - K Nearest Neighbor(KNN)
# - Decision Tree
# - Support Vector Machine
# - Logistic Regression
# 
# 
# 
# __ Notice:__ 
# - You can go above and change the pre-processing, feature selection, feature-extraction, and so on, to make a better model.
# - You should use either scikit-learn, Scipy or Numpy libraries for developing the classification algorithms.
# - You should include the code of the algorithm in the following cells.

# # K Nearest Neighbor(KNN)
# Notice: You should find the best k to build the model with the best accuracy.  
# **warning:** You should not use the __loan_test.csv__ for finding the best k, however, you can split your train_loan.csv into train and test to find the best __k__.

# In[27]:


#import libraries for the inbuilt model functions
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn import metrics


# In[28]:


#Train model

# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors = 7) #we initialize with 7 neighbors, will change with time
# Fit the classifier to the data
knn.fit(X_train,y_train)


# In[29]:


#Testing the model

#show first 5 model predictions on the test data
y_pred_knn = knn.predict(X_test)
print(y_pred_knn.shape)


# In[30]:


#check accuracy of our model on the test data
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_knn))


# In[31]:


#create a new KNN model for 5 fold cross validation
knn_cv = KNeighborsClassifier(n_neighbors=7)
#train model with cv of 5 
cv_scores = cross_val_score(knn_cv, X, y, cv=5)


# In[32]:


#print each cv score (accuracy) and average them
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# In[33]:


#create new a knn model
knn2 = KNeighborsClassifier()
#create a dictionary of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(1, 25)}
#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
#fit model to data
knn_gscv.fit(X, y)


# In[34]:


#check top performing n_neighbors value
print(knn_gscv.best_params_)
#check mean score for the top performing value of n_neighbors
print(knn_gscv.best_score_)


# # Decision Tree

# In[35]:


#import libraries for the inbuilt model functions
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


# In[90]:


#Build Decision Tree model
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# Create Decision Tree classifer object
DT_clf = DecisionTreeClassifier(criterion="entropy", max_depth=1)

# Train Decision Tree Classifer
DT_clf.fit(X_train,y_train)



# In[91]:


#Prediction accuracy
#Predict the response for test dataset
y_pred_dt = DT_clf.predict(X_test)

# Model Accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_dt))


# In[92]:


#create a new DT model for 5 fold cross validation
DT_clf2 = DecisionTreeClassifier(criterion="entropy", max_depth=1)
#train model with cv of 5 
cv_scores_DT = cross_val_score(DT_clf2, X, y, cv=5)


# In[93]:


#print each cv score (accuracy) and average them
print(cv_scores_DT)
print('cv_scores mean:{}'.format(np.mean(cv_scores_DT)))


# In[95]:


#create new a DT model
DT_clf3 = DecisionTreeClassifier(criterion="entropy")
#create a dictionary of all values we want to test for n_neighbors
param_grid = {'max_depth': np.arange(1, 25)}
#use gridsearch to test all values for n_neighbors
dt_gscv = GridSearchCV(DT_clf3, param_grid, cv=5)
#fit model to data
dt_gscv.fit(X, y)


# In[96]:


#check top performing max depth
print(dt_gscv.best_params_)
#check mean score for the top performing value of max depth
print(dt_gscv.best_score_)


# # Support Vector Machine

# In[42]:


#Import svm model
from sklearn import svm


# In[85]:


#Build SVM model
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
#Create a svm Classifier
svm_clf = svm.SVC(kernel='poly',C=0.1,gamma=0.01,probability=True) 

#Train the model using the training sets
svm_clf.fit(X_train, y_train)


# In[86]:


y_pred_svm = svm_clf.predict(X_test)


# In[87]:


# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_svm))


# In[88]:


#create a new DT model for 5 fold cross validation
svm_clf2 = svm.SVC(kernel='poly',C=0.1,gamma=0.01)
#train model with cv of 5 
cv_scores_svm = cross_val_score(svm_clf2, X, y, cv=5)


# In[89]:


#print each cv score (accuracy) and average them
print(cv_scores_svm)
print('cv_scores mean:{}'.format(np.mean(cv_scores_svm)))


# In[ ]:





# # Logistic Regression

# In[48]:


# import the class
from sklearn.linear_model import LogisticRegression


# In[83]:


#Build Logistic Regression
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# instantiate the model (using the default parameters)
logreg = LogisticRegression(solver='sag',penalty='l2',C=0.01,max_iter=100,warm_start=True)

# fit the model with data
logreg.fit(X_train,y_train)

#
y_pred_log=logreg.predict(X_test)


# In[75]:


# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_log))


# # Model Evaluation using Test set

# In[51]:


from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# First, download and load the test set:

# In[52]:


get_ipython().system('wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv')


# ### Load Test set for evaluation 

# In[53]:


test_df = pd.read_csv('loan_test.csv')
test_df.head()


# In[54]:


print(test_df.shape)


# In[55]:


# Preprocess the testing data

#convert to date time object
test_df['due_date'] = pd.to_datetime(test_df['due_date'])
test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])

#Preprocess rest days of the week to weekend and weekdays
test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek
test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
test_df.head()

#Replace gender to 0 1
#test_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
test_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
#test_df.head()

#Quantify category parameters
Feature_eval = test_df[['Principal','terms','age','Gender','weekend']]
Feature_eval = pd.concat([Feature_eval,pd.get_dummies(test_df['education'])], axis=1)
#Feature_eval.drop(['Master or Above'], axis = 1,inplace=True)
Feature_eval.head()


# In[56]:


print(Feature_eval.shape)


# In[57]:


#Set the feature and labels for the test data
#set features
X_eval = Feature_eval
#normaliza features
X_eval= preprocessing.StandardScaler().fit(X_eval).transform(X_eval)
print(X_eval.shape)
#X_eval.head()
#set labels
#Replace labels to 0 and 1
y_eval = test_df['loan_status'].replace(to_replace=['PAIDOFF','COLLECTION'], value=[0,1]).values
#y_eval = test_df['loan_status'].values
print(y_eval.shape)
print(y_eval)


# In[79]:


#Get the performance for all the algorithms

#KNN evaluation
y_knn = knn.predict(X_eval)
#print(y_knn)
jaccard_knn = jaccard_similarity_score(y_eval, y_knn)
f1score_knn = f1_score(y_eval, y_knn,)


print('Jaccard is : {}'.format(jaccard_knn))
print('F1-score is : {}'.format(f1score_knn))
#print('Log Loss is : {}'.format(logloss_knn))


# In[97]:


#Decision Tree evaluation
y_dt = DT_clf.predict(X_eval)
jaccard_DT = jaccard_similarity_score(y_eval, y_dt)
f1score_DT = f1_score(y_eval, y_dt,average='weighted')

print('Jaccard is : {}'.format(jaccard_DT))
print('F1-score is : {}'.format(f1score_DT))
#print('Log Loss is : {}'.format(logloss_DT))


# In[81]:


#SVM evaluation
y_svm = svm_clf.predict(X_eval)
jaccard_SVM = jaccard_similarity_score(y_eval, y_svm)
f1score_SVM = f1_score(y_eval, y_svm,average='weighted')



print('Jaccard is : {}'.format(jaccard_SVM))
print('F1-score is : {}'.format(f1score_SVM))


# In[84]:


#LogReg evaluation
y_logreg = logreg.predict(X_eval)
jaccard_LogReg = jaccard_similarity_score(y_eval, y_logreg)
f1score_LogReg = f1_score(y_eval, y_logreg,average='weighted')
y_logreg_proba = logreg.predict_proba(X_eval)
logloss_LogReg = log_loss(y_eval, y_logreg_proba)

print('Jaccard is : {}'.format(jaccard_LogReg))
print('F1-score is : {}'.format(f1score_LogReg))
print('Log Loss is : {}'.format(logloss_LogReg))


# # Report
# You should be able to report the accuracy of the built model using different evaluation metrics:

# In[ ]:





# | Algorithm          | Jaccard | F1-score | LogLoss |
# |--------------------|---------|----------|---------|
# | KNN                | 0.7222  | 0.4828   | NA      |
# | Decision Tree      | 0.7407  | 0.6304   | NA      |
# | SVM                | 0.7407  | 0.6304   | NA      |
# | LogisticRegression | 0.7407  | 0.6304   | 0.4938  |

# <h2>Want to learn more?</h2>
# 
# IBM SPSS Modeler is a comprehensive analytics platform that has many machine learning algorithms. It has been designed to bring predictive intelligence to decisions made by individuals, by groups, by systems – by your enterprise as a whole. A free trial is available through this course, available here: <a href="http://cocl.us/ML0101EN-SPSSModeler">SPSS Modeler</a>
# 
# Also, you can use Watson Studio to run these notebooks faster with bigger datasets. Watson Studio is IBM's leading cloud solution for data scientists, built by data scientists. With Jupyter notebooks, RStudio, Apache Spark and popular libraries pre-packaged in the cloud, Watson Studio enables data scientists to collaborate on their projects without having to install anything. Join the fast-growing community of Watson Studio users today with a free account at <a href="https://cocl.us/ML0101EN_DSX">Watson Studio</a>
# 
# <h3>Thanks for completing this lesson!</h3>
# 
# <h4>Author:  <a href="https://ca.linkedin.com/in/saeedaghabozorgi">Saeed Aghabozorgi</a></h4>
# <p><a href="https://ca.linkedin.com/in/saeedaghabozorgi">Saeed Aghabozorgi</a>, PhD is a Data Scientist in IBM with a track record of developing enterprise level applications that substantially increases clients’ ability to turn data into actionable knowledge. He is a researcher in data mining field and expert in developing advanced analytic methods like machine learning and statistical modelling on large datasets.</p>
# 
# <hr>
# 
# <p>Copyright &copy; 2018 <a href="https://cocl.us/DX0108EN_CC">Cognitive Class</a>. This notebook and its source code are released under the terms of the <a href="https://bigdatauniversity.com/mit-license/">MIT License</a>.</p>
