
# coding: utf-8

# In[58]:


import pandas as pd
import numpy as np


# In[59]:


import matplotlib.pyplot as plt
import datetime


# In[60]:


import seaborn as sns


# In[61]:


sns.set(style="white")


# In[62]:


rawdf = pd.read_csv('C:\\amit\\per\\kag\\2.DonorsChoose\\train\\train.csv')


# In[7]:


#len(rawdf)


# In[7]:


#rawdf.head()


# In[8]:


#len(rawdf.teacher_id.unique())


# # Nulls

# In[11]:


#rawdf.isnull().sum()*100/rawdf.shape[0]


# In[23]:


#rawdf.drop(['project_essay_3','project_essay_4'], axis=1, inplace=True)


# In[12]:


#rawdf.project_submitted_datetime.value_counts()


# In[ ]:


#rawdf.drop('project_essay_3', axis=1, inplace=True)


# In[6]:


#rawdf.plot(x='project_submitted_datetime', kind='hist', )


# In[ ]:


#ax = rawdf.plot(x='project_submitted_datetime', kind='bar')


# In[5]:


rawdf.dtypes


# In[24]:


rawdf.project_is_approved.value_counts()


# In[25]:


sns.countplot(x='project_is_approved',data=rawdf)
plt.show()


# In[6]:


rawdf['datetime']=pd.to_datetime(rawdf['project_submitted_datetime'])


# In[31]:


c = rawdf.groupby([rawdf['datetime'].dt.date]).size()


# In[32]:


c


# In[37]:


rawdf.teacher_prefix.value_counts()


# In[41]:


c = rawdf.groupby(['teacher_prefix','project_is_approved']).size()


# In[42]:


c


# In[63]:


sns.factorplot("project_is_approved",col='teacher_prefix',data=rawdf, saturation=.5,kind="count", ci=None, aspect=.6)


# In[64]:


plt.show()


# In[68]:


sns.factorplot("project_is_approved",col='school_state',data=rawdf, saturation=.5,kind="count", ci=None, aspect=.6,col_wrap=4)


# In[69]:


plt.show()


# In[72]:


c = rawdf.groupby([rawdf['datetime'].dt.year,rawdf['datetime'].dt.month,'project_is_approved']).size()


# In[73]:


c


# In[74]:


rawdf.project_grade_category.value_counts()


# In[75]:


sns.factorplot("project_is_approved",col='project_grade_category',data=rawdf, saturation=.5,kind="count", ci=None, aspect=.6)


# In[76]:


plt.show()


# # Subject categories need to be cleaned

# In[77]:


rawdf.project_subject_categories.value_counts()


# In[78]:


sns.factorplot("project_is_approved",col='project_subject_categories',data=rawdf, saturation=.5,kind="count", ci=None, aspect=.6,col_wrap=4)


# In[79]:


plt.show()


# In[80]:


rawdf.project_subject_subcategories.value_counts()


# In[10]:


rawdf.teacher_number_of_previously_posted_projects.value_counts()


# In[69]:


rawdf.columns


# In[6]:


rawdf.drop(['project_title','project_essay_1','project_essay_2','project_essay_3','project_essay_4','project_resource_summary',], axis=1, inplace=True)


# In[7]:


rawdf.columns


# In[8]:


resources = pd.read_csv('C:\\amit\\per\\kag\\2.DonorsChoose\\train\\resources.csv')


# In[71]:


resources.head(n=5)


# In[9]:


price = pd.DataFrame({'resource_price' : resources.groupby(['id'])['price'].agg('sum')}).reset_index()


# In[73]:


price.head()


# In[10]:


rawdf = pd.merge(rawdf,price,on='id')


# In[11]:


rawdf.head()


# In[94]:


rawdf.dtypes


# # Encode categorical

# In[11]:


rawdf = pd.concat([rawdf,pd.get_dummies(rawdf['teacher_prefix'], prefix='teacher_prefix',drop_first=True)],axis=1)


# In[12]:


rawdf.drop(['teacher_prefix'],axis=1, inplace=True)


# In[13]:


rawdf = pd.concat([rawdf,pd.get_dummies(rawdf['school_state'], prefix='school_state',drop_first=True)],axis=1)


# In[14]:


rawdf.drop(['school_state'],axis=1, inplace=True)


# In[62]:


rawdf.head()


# In[15]:


rawdf = pd.concat([rawdf,pd.get_dummies(rawdf['project_grade_category'], prefix='project_grade_category',drop_first=True)],axis=1)


# In[16]:


rawdf.drop(['project_grade_category'],axis=1, inplace=True)


# In[17]:


rawdf = pd.concat([rawdf,pd.get_dummies(rawdf['project_subject_categories'], prefix='project_subject_categories',drop_first=True)],axis=1)


# In[18]:


rawdf.drop(['project_subject_categories'],axis=1, inplace=True)


# In[19]:


rawdf = pd.concat([rawdf,pd.get_dummies(rawdf['project_subject_subcategories'], prefix='project_subject_subcategories',drop_first=True)],axis=1)


# In[20]:


rawdf.drop(['project_subject_subcategories'],axis=1, inplace=True)


# In[21]:


# array of all the categorical variables 
rawdf.select_dtypes(include=['O']).columns.values


# In[22]:


rawdf['datetime']=pd.to_datetime(rawdf['project_submitted_datetime'])


# In[23]:


c = rawdf.groupby([rawdf['datetime'].dt.year,rawdf['datetime'].dt.month]).size()


# In[24]:


c


# In[24]:


rawdf['yyyymm'] = rawdf.datetime.map(lambda x: x.strftime('%Y%m'))


# In[25]:


rawdf = pd.concat([rawdf,pd.get_dummies(rawdf['yyyymm'], prefix='yyyymm',drop_first=True)],axis=1)


# In[26]:


rawdf.drop(['yyyymm'],axis=1, inplace=True)


# In[27]:


rawdf.drop(['project_submitted_datetime'],axis=1, inplace=True)


# In[28]:


rawdf.drop(['datetime'],axis=1, inplace=True)


# In[30]:


rawdf.head()


# In[29]:


rawdf.select_dtypes(include=['O']).columns.values


# In[96]:


rawdf.hist(column='resource_price',bins=10)


# In[97]:


plt.show()


# In[56]:


#newdf = rawdf.drop(['id','teacher_id'], axis=1)


# In[30]:


col = 'project_is_approved'


# In[31]:


rawdf = pd.concat([rawdf.drop(col,axis=1),rawdf[col],], axis=1)


# In[32]:


rawdf.head()


# # Feature Selection

# In[33]:


from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


# In[33]:


logreg = LogisticRegression()


# In[34]:


data_final_vars=rawdf.columns.values.tolist()


# In[35]:


y=['project_is_approved']


# In[36]:


X=[i for i in data_final_vars if i not in ['id','teacher_id','project_is_approved']]


# In[37]:


rfe = RFE(logreg, 50)


# In[38]:


rfe = rfe.fit(rawdf[X], rawdf[y] )


# In[39]:


print(rfe.support_)


# In[40]:


print(rfe.ranking_)


# In[41]:


a= np.array(X)


# In[42]:


b= np.array(rfe.support_)


# In[43]:


a[b == True]


# In[44]:


cols = a[b == True]


# In[45]:


cols


# In[37]:


X=rawdf[X]


# In[38]:


X.shape


# In[39]:


y=rawdf[y]


# In[49]:


import statsmodels.api as sm
logit_model=sm.Logit(y,X)


# In[51]:


result=logit_model.fit()
print(result.summary())


# In[40]:


from sklearn.model_selection import train_test_split


# In[41]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0,stratify=y)


# In[125]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[126]:


logreg = LogisticRegression()


# In[127]:


logreg.fit(X_train, y_train)


# In[128]:


y_pred = logreg.predict(X_test)


# In[129]:


print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# In[131]:


from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'


# In[133]:


results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)


# In[134]:


print("10-fold cross validation average accuracy: %.3f" % (results.mean()))


# In[135]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# In[136]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[137]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# # XGBoost

# In[42]:


import xgboost as xgb


# In[43]:


import timeit


# In[44]:


model1 = xgb.XGBClassifier()
model2 = xgb.XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.1, subsample=0.5)


# In[45]:


from sklearn.metrics import classification_report


# In[51]:


train_model1 = model1.fit(X_train, y_train)


# In[52]:


pred1 = train_model1.predict(X_test)


# In[54]:


print('Model 1 XGboost Report %r' % (classification_report(y_test, pred1)))


# In[55]:


from sklearn.metrics import accuracy_score


# In[56]:


print("Accuracy for model 1: %.2f" % (accuracy_score(y_test, pred1) * 100))


# In[58]:


print(classification_report(y_test, pred1))


# In[59]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, pred1)
print(confusion_matrix)


# In[60]:


model2 = xgb.XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.1, subsample=0.5)


# In[63]:


train_model2 = model2.fit(X_train, y_train)


# In[64]:


pred2 = train_model2.predict(X_test)


# In[76]:


print(classification_report(y_test, pred2))


# In[74]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, pred2)
print(confusion_matrix)


# In[70]:


y_test


# In[73]:


pred2.shape


# In[77]:


print("Accuracy for model 2: %.2f" % (accuracy_score(y_test, pred2) * 100))


# In[39]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


model2


# In[80]:


param_test = {
 'max_depth':[4,5,6],
 'min_child_weight':[4,5,6]
}


# In[81]:


gsearch = GridSearchCV(estimator = xgb.XGBClassifier(n_estimators=140,subsample=0.5, colsample_bytree=0.8,nthread=4), 
                       param_grid = param_test, scoring='roc_auc',n_jobs=4,iid=False, cv=5)


# In[ ]:


train_model3 = gsearch.fit(X_train, y_train)


# In[ ]:


#14:27


# In[38]:


# Tuning no. of estimators


# In[57]:


model = xgb.XGBClassifier()


# In[46]:


n_estimators = range(50, 400, 50)


# In[47]:


param_grid = dict(n_estimators=n_estimators)


# In[48]:


param_grid


# In[51]:


gsearch = GridSearchCV(model, param_grid = param_grid, scoring='roc_auc',n_jobs=-1)


# In[54]:


train_model1 = gsearch.fit(X_train, y_train)


# In[55]:


# summarize results
print("Best: %f using %s" % (train_model1.best_score_, train_model1.best_params_))


# In[56]:


max_depth = [2, 4, 6, 8]


# In[58]:


param_grid = dict(max_depth=max_depth)


# In[68]:


model = xgb.XGBClassifier(n_estimators = 350)


# In[69]:


gsearch = GridSearchCV(model, param_grid = param_grid, scoring='roc_auc',n_jobs=-1)


# In[70]:


train_model1 = gsearch.fit(X_train, y_train)


# In[71]:


# summarize results
print("Best: %f using %s" % (train_model1.best_score_, train_model1.best_params_))


# In[42]:


model = xgb.XGBClassifier(n_estimators=350,max_depth=4, scale_pos_weight=0.18)


# In[43]:


train_model = model.fit(X_train, y_train)


# In[44]:


pred2 = model.predict(X_test)


# In[45]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, pred2)
print(confusion_matrix)


# In[46]:


from sklearn.metrics import accuracy_score


# In[47]:


print(classification_report(y_test, pred2))


# In[48]:


print("Accuracy for model: %.2f" % (accuracy_score(y_test, pred2) * 100))


# In[39]:


max_depth = [2, 4, 6, 8]


# In[40]:


n_estimators = range(50, 400, 50)


# In[41]:


model = xgb.XGBClassifier(scale_pos_weight=0.18)


# In[49]:


param_grid = dict(n_estimators=n_estimators,max_depth=max_depth)


# In[52]:


gsearch = GridSearchCV(model, param_grid = param_grid, scoring='roc_auc',n_jobs=-1,cv=5)


# In[ ]:


train_model = gsearch.fit(X_train, y_train)


# In[78]:


print("Best: %f using %s" % (train_model.best_score_, train_model.best_params_))


# In[64]:


pred2 = model.predict(X_test)


# In[58]:


from sklearn.metrics import confusion_matrix


# In[70]:


confusion_matrix = confusion_matrix(y_test, pred2)


# In[73]:


print(confusion_matrix)


# In[71]:


print("Accuracy for model: %.2f" % (accuracy_score(y_test, pred2) * 100))


# In[72]:


print(classification_report(y_test, pred2))


# In[46]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0,stratify=y)


# In[47]:


X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_train, y_train, test_size=0.9, random_state=0, stratify = y_train)


# In[49]:


max_depth = [2, 4, 6, 8]


# In[50]:


n_estimators = range(50, 400, 50)


# In[51]:


param_grid = dict(n_estimators=n_estimators,max_depth=max_depth)


# In[55]:


gsearch = GridSearchCV(model, param_grid = param_grid, scoring='roc_auc',n_jobs=4,cv=5)


# In[ ]:


train_model = gsearch.fit(X_train_1, y_train_1)


# In[57]:


train_model


# In[59]:


print("Best: %f using %s" % (train_model.best_score_, train_model.best_params_))


# In[ ]:


pred2 = model.predict(y_train_1)


# In[56]:


c = rawdf.groupby(['project_subject_subcategories','project_is_approved']).size()


# In[57]:


c

