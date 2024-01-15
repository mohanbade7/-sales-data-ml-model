#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd
df=pd.read_csv("salesdaily.csv")
df


# In[34]:


df.columns


# In[35]:


medication_columns = df.columns[1:9] 
df['Target'] = df[medication_columns].sum(axis=1).shift(-1)
print(df.head())


# In[36]:


df.columns


# In[37]:


(df['Target'])


# In[38]:


df.isnull().sum()


# In[39]:


df=df.fillna(method='bfill')
df= df.fillna(method='ffill')


# In[40]:


df.isnull().sum()


# In[41]:


df.drop_duplicates(inplace=True)


# In[42]:


df['datum'] = pd.to_datetime(df['datum'])


# In[43]:


df


# In[44]:


df.columns


# In[45]:


from sklearn.preprocessing import LabelEncoder


ordinal=['datum','Month','Hour']
nominal=['Weekday Name']

model=LabelEncoder()

for col in ordinal:
    df[col]=model.fit_transform(df[col])


# In[46]:


dff=pd.get_dummies(df[nominal])


# In[47]:


dff


# In[48]:


print("***",dff.columns,"*****",df.columns)


# In[49]:


dff[['datum','M01AB', 'M01AE', 'N02BA', 'N02BE', 'N05B', 'N05C', 'R03','R06', 'Year', 'Month', 'Hour','Target']]=df[['datum','M01AB', 'M01AE', 'N02BA', 'N02BE', 'N05B', 'N05C', 'R03','R06', 'Year', 'Month', 'Hour','Target']]


# In[50]:


dff


# In[51]:


dff=dff.astype('int')


# In[52]:


dff


# In[53]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

x=dff.drop(['Target'],axis=1)
y=dff['Target']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

model=LogisticRegression()
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred=model.predict(x_test)
print(type(model).__name__)
print("******* train data******")

print("accuracy score:",accuracy_score(y_train,train_pred))
print("precision score:",precision_score(y_train,train_pred,average='weighted'))
print("recall score:",recall_score(y_train,train_pred,average='weighted'))
print("f1 score:",f1_score(y_train,train_pred,average='weighted'))



print("******* test data******")

print("accuracy score:",accuracy_score(y_test,test_pred))
print("precision score:",precision_score(y_test,test_pred,average='weighted'))
print("recall score:",recall_score(y_test,test_pred,average='weighted'))
print("f1 score:",f1_score(y_test,test_pred,average='weighted'))


# In[54]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.model_selection import train_test_split

x=dff.drop(['Target'],axis=1)
y=dff['Target']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

model=KNeighborsClassifier()
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred=model.predict(x_test)
print(type(model).__name__)
print("********** train data********")
print("accuracy score:",accuracy_score(y_train,train_pred))
print("precision score:",precision_score(y_train,train_pred,average='weighted'))
print("recall score:",recall_score(y_train,train_pred,average='weighted'))
print("f1 score:",f1_score(y_train,train_pred,average='weighted'))


print("********** test data********")
print("acuracy score:",accuracy_score(y_train,train_pred))
print("precision score:",precision_score(y_train,train_pred,average='weighted'))
print("recall score:",recall_score(y_train,train_pred,average='weighted'))
print("f1 score:",f1_score(y_train,train_pred,average='weighted'))


# In[55]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

x=dff.drop(['Target'],axis=1)
y=dff['Target']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

model=DecisionTreeClassifier()
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred= model.predict(x_test)

print(type(model).__name__)
print("******** train data***********")
print("accuracy score:",accuracy_score(y_train,train_pred))
print("precision score:",precision_score(y_train,train_pred,average='weighted'))
print(" recall score:",recall_score(y_train,train_pred,average='weighted'))
print("f1 score:",f1_score(y_train,train_pred,average='weighted'))


print("******** test data***********")
print("accuracy score:",accuracy_score(y_test,test_pred))
print("precision score:",precision_score(y_test,test_pred,average='weighted'))
print(" recall score:",recall_score(y_test,test_pred,average='weighted'))
print("f1 score:",f1_score(y_test,test_pred,average='weighted'))


# In[56]:





from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

x=dff.drop(['Target'],axis=1)
y=dff['Target']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

model=RandomForestClassifier()
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred= model.predict(x_test)

print(type(model).__name__)
print("******** train data***********")
print("accuracy score:",accuracy_score(y_train,train_pred))
print("precision score:",precision_score(y_train,train_pred,average='weighted'))
print(" recall score:",recall_score(y_train,train_pred,average='weighted'))
print("f1 score:",f1_score(y_train,train_pred,average='weighted'))


print("******** test data***********")
print("accuracy score:",accuracy_score(y_test,test_pred))
print("precision score:",precision_score(y_test,test_pred,average='weighted'))
print(" recall score:",recall_score(y_test,test_pred,average='weighted'))
print("f1 score:",f1_score(y_test,test_pred,average='weighted'))


# In[57]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

x=dff.drop(['Target'],axis=1)
y=dff['Target']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

model=SVC()
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred= model.predict(x_test)

print(type(model).__name__)
print("******** train data***********")
print("accuracy score:",accuracy_score(y_train,train_pred))
print("precision score:",precision_score(y_train,train_pred,average='weighted'))
print(" recall score:",recall_score(y_train,train_pred,average='weighted'))
print("f1 score:",f1_score(y_train,train_pred,average='weighted'))


print("******** test data***********")
print("accuracy score:",accuracy_score(y_test,test_pred))
print("precision score:",precision_score(y_test,test_pred,average='weighted'))
print(" recall score:",recall_score(y_test,test_pred,average='weighted'))
print("f1 score:",f1_score(y_test,test_pred,average='weighted'))


# In[58]:


from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier

x=dff.drop(['Target'],axis=1)
y=dff['Target']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

model=GaussianNB()
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred= model.predict(x_test)

print(type(model).__name__)
print("******** train data***********")
print("accuracy score:",accuracy_score(y_train,train_pred))
print("precision score:",precision_score(y_train,train_pred,average='weighted'))
print(" recall score:",recall_score(y_train,train_pred,average='weighted'))
print("f1 score:",f1_score(y_train,train_pred,average='weighted'))


print("******** test data***********")
print("accuracy score:",accuracy_score(y_test,test_pred))
print("precision score:",precision_score(y_test,test_pred,average='weighted'))
print(" recall score:",recall_score(y_test,test_pred,average='weighted'))
print("f1 score:",f1_score(y_test,test_pred,average='weighted'))




# In[59]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

x=dff.drop(['Target'],axis=1)
y=dff['Target']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

model=GradientBoostingClassifier()
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred= model.predict(x_test)

print(type(model).__name__)
print("******** train data***********")
print("accuracy score:",accuracy_score(y_train,train_pred))
print("precision score:",precision_score(y_train,train_pred,average='weighted'))
print(" recall score:",recall_score(y_train,train_pred,average='weighted'))
print("f1 score:",f1_score(y_train,train_pred,average='weighted'))


print("******** test data***********")
print("accuracy score:",accuracy_score(y_test,test_pred))
print("precision score:",precision_score(y_test,test_pred,average='weighted'))
print(" recall score:",recall_score(y_test,test_pred,average='weighted'))
print("f1 score:",f1_score(y_test,test_pred,average='weighted'))


# In[ ]:





# In[ ]:




