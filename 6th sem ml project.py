#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
data = pd.read_csv(r"C:\Users\HARISH BJ\Downloads\laptop_data.csv")


# In[6]:


data.head()


# # data cleaning

# In[7]:


data.shape


# In[8]:


data.head()


# In[9]:


data.info()


# In[10]:


data.duplicated().sum()


# In[11]:


data.isnull().sum()


# In[12]:


data.drop(columns=['Unnamed: 0'],inplace=True)


# In[13]:


data.head()


# In[14]:


data['Ram'].str.replace('GB','')


# In[15]:


data.info()


# In[16]:


data['Weight'] = data['Weight'].str.replace('kg','')
data['Ram'] = data['Ram'].str.replace('GB','')


# In[17]:


data.head()


# In[18]:


data['Weight'] = data['Weight'].astype('float32')
data['Ram'] = data['Ram'].astype('int32')


# In[19]:


data.info()


# # exploratory data analysis

# In[20]:


import seaborn as sns
sns.distplot(data['Price'])


# In[21]:


data['Company'].value_counts().plot(kind='bar')


# In[22]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.barplot(x=data['Company'],y=data['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[23]:


data['TypeName'].value_counts().plot(kind='bar')


# In[24]:


import matplotlib.pyplot as plt
sns.barplot(x=data['TypeName'],y=data['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[25]:


sns.distplot(data['Inches'])


# In[26]:


sns.scatterplot(x=data['Inches'],y=data['Price'])


# In[27]:


data['ScreenResolution'].value_counts()


# In[28]:


data['Touchscreen'] = data['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)


# In[29]:


data.sample(5)


# In[30]:


data['Touchscreen'].value_counts().plot(kind='bar')


# In[31]:


sns.barplot(x=data['Touchscreen'],y=data['Price'])


# In[32]:


data['Ips'] = data['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)


# In[33]:


data.head()


# In[34]:


data['Ips'].value_counts().plot(kind='bar')


# In[35]:


sns.barplot(x=data['Ips'],y=data['Price'])


# In[36]:


new = data['ScreenResolution'].str.split('x',n=1,expand=True)


# In[37]:


data['X_res'] = new[0]
data['Y_res'] = new[1]


# In[38]:


data.sample(5)


# In[39]:


data['X_res'] = data['X_res'].str.replace(',','').str.findall(r'(\d+\.?\d+)').apply(lambda x:x[0])


# In[40]:


data.head()


# In[41]:


data['X_res'] = data['X_res'].astype('int')
data['Y_res'] = data['Y_res'].astype('int')


# In[42]:


data.info()


# In[43]:


data.corr()['Price']


# In[44]:


data['ppi'] = (((data['X_res']**2) + (data['Y_res']**2))**0.5/data['Inches']).astype('float')


# In[45]:


data.corr()['Price']


# In[46]:


data.drop(columns=['ScreenResolution'],inplace=True)


# In[47]:


data.head()


# In[48]:


data.drop(columns=['Inches','X_res','Y_res'],inplace=True)


# In[49]:


data.head()


# In[50]:


data['Cpu'].value_counts()


# In[51]:


data['Cpu Name'] = data['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))


# In[52]:


data.head()


# In[53]:


def fetch_processor(text):
    if text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3':
        return text
    else:
        if text.split()[0] == 'Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'


# In[54]:


data['Cpu brand'] = data['Cpu Name'].apply(fetch_processor)


# In[55]:


data.head()


# In[56]:


data['Cpu brand'].value_counts().plot(kind='bar')


# In[401]:


sns.barplot(x=data['Cpu brand'],y=data['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[57]:


data.drop(columns=['Cpu','Cpu Name'],inplace=True)


# In[58]:


data.head()


# In[59]:


data['Ram'].value_counts().plot(kind='bar')


# In[60]:


sns.barplot(x=data['Ram'],y=data['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[61]:


data['Memory'].value_counts()


# In[62]:


data['Memory'] = data['Memory'].astype(str).replace('\.0', '', regex=True)
data["Memory"] = data["Memory"].str.replace('GB', '')
data["Memory"] = data["Memory"].str.replace('TB', '000')
new = data["Memory"].str.split("+", n = 1, expand = True)

data["first"]= new[0]
data["first"]=data["first"].str.strip()

data["second"]= new[1]

data["Layer1HDD"] = data["first"].apply(lambda x: 1 if "HDD" in x else 0)
data["Layer1SSD"] = data["first"].apply(lambda x: 1 if "SSD" in x else 0)
data["Layer1Hybrid"] = data["first"].apply(lambda x: 1 if "Hybrid" in x else 0)
data["Layer1Flash_Storage"] = data["first"].apply(lambda x: 1 if "Flash Storage" in x else 0)

data['first'] = data['first'].str.replace(r'\D', '')

data["second"].fillna("0", inplace = True)
data["Layer2HDD"] = data["second"].apply(lambda x: 1 if "HDD" in x else 0)
data["Layer2SSD"] = data["second"].apply(lambda x: 1 if "SSD" in x else 0)
data["Layer2Hybrid"] = data["second"].apply(lambda x: 1 if "Hybrid" in x else 0)
data["Layer2Flash_Storage"] = data["second"].apply(lambda x: 1 if "Flash Storage" in x else 0)

data['second'] = data['second'].str.replace(r'\D', '')

data["first"] = data["first"].astype(int)
data["second"] = data["second"].astype(int)

data["HDD"]=(data["first"]*data["Layer1HDD"]+data["second"]*data["Layer2HDD"])
data["SSD"]=(data["first"]*data["Layer1SSD"]+data["second"]*data["Layer2SSD"])
data["Hybrid"]=(data["first"]*data["Layer1Hybrid"]+data["second"]*data["Layer2Hybrid"])
data["Flash_Storage"]=(data["first"]*data["Layer1Flash_Storage"]+data["second"]*data["Layer2Flash_Storage"])

data.drop(columns=['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid',
       'Layer1Flash_Storage', 'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid',
       'Layer2Flash_Storage'],inplace=True)


# In[63]:


data.sample(5)


# In[64]:


data.drop(columns=['Memory'],inplace=True)


# In[65]:


data.head()


# In[66]:


data.corr()['Price']


# In[67]:


data.drop(columns=['Hybrid','Flash_Storage'],inplace=True)


# In[68]:


data.head()


# In[69]:


data['Gpu'].value_counts()


# In[70]:


data['Gpu brand'] = data['Gpu'].apply(lambda x:x.split()[0])


# In[71]:


data.head()


# In[72]:


data['Gpu brand'].value_counts()


# In[73]:


data = data[data['Gpu brand'] != 'ARM']


# In[74]:


data['Gpu brand'].value_counts()


# In[75]:


import numpy as np
sns.barplot(x=data['Gpu brand'],y=data['Price'],estimator=np.median)
plt.xticks(rotation='vertical')
plt.show()


# In[76]:


data.drop(columns=['Gpu'],inplace=True)


# In[77]:


data.head()


# In[78]:


data['OpSys'].value_counts()


# In[79]:


sns.barplot(x=data['OpSys'],y=data['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[80]:


def cat_os(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'Windows'
    elif inp == 'macOS' or inp == 'Mac OS X':
        return 'Mac'
    else:
        return 'Others/No OS/Linux'


# In[81]:


data['os'] = data['OpSys'].apply(cat_os)


# In[82]:


data.head()


# In[83]:


data.drop(columns=['OpSys'],inplace=True)


# In[84]:


sns.barplot(x=data['os'],y=data['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[85]:


sns.distplot(data['Weight'])


# In[86]:


sns.scatterplot(x=data['Weight'],y=data['Price'])


# In[87]:


data.corr()['Price']


# In[88]:


sns.heatmap(data.corr())


# In[89]:


sns.distplot(np.log(data['Price']))


# In[90]:


X = data.drop(columns=['Price'])
y = np.log(data['Price'])


# In[91]:


X


# In[92]:


y


# In[93]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=2)


# In[94]:


X_train


# In[95]:


X_test


# In[96]:


y_train


# In[97]:


y_test


# In[98]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error


# In[99]:


from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor


# # linear regression
# 

# In[100]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = LinearRegression()

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # ridge regression
# 

# In[101]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = Ridge(alpha=10)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # lasso
# 

# In[102]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = Lasso(alpha=10)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # knn

# In[103]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = KNeighborsRegressor(n_neighbors=3)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # decesion tree

# In[104]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = DecisionTreeRegressor(max_depth=8)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # svm

# In[105]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = SVR(kernel='rbf',C=10000,epsilon=0.1)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # Random Forest

# In[106]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = RandomForestRegressor(n_estimators=100,
                             random_state=3,
                             max_samples=0.5,
                             max_features=0.75,
                             max_depth=15)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # ExtraTrees

# In[107]:


from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# Assuming you have X_train, X_test, y_train, y_test defined

step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse=False, drop='first'), [0, 1, 7, 10, 11])
], remainder='passthrough')

step2 = ExtraTreesRegressor(n_estimators=100,
                            random_state=3,
                            max_samples=0.5,  # Correct parameter name is max_samples
                            max_features=0.75,
                            max_depth=15,
                            bootstrap=True)  # Set bootstrap=True if you want to use max_samples

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

print('R2 score', r2_score(y_test, y_pred))
print('MAE', mean_absolute_error(y_test, y_pred))


# # gradient boost

# In[108]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = GradientBoostingRegressor(n_estimators=500)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # xg boost

# In[109]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = XGBRegressor(n_estimators=45,max_depth=5,learning_rate=0.5)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# #  voting regressor

# In[110]:


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np

# Assuming you have X_train, X_test, y_train, y_test defined

step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse=False, drop='first'), [0, 1, 7, 10, 11])
], remainder='passthrough')

rf = RandomForestRegressor(n_estimators=350, random_state=3, max_samples=0.5, max_features=0.75, max_depth=15, bootstrap=True)
gbdt = GradientBoostingRegressor(n_estimators=100, max_features=0.5)
xgb = XGBRegressor(n_estimators=25, learning_rate=0.3, max_depth=5)
et = ExtraTreesRegressor(n_estimators=100, random_state=3, max_samples=0.5, max_features=0.75, max_depth=10, bootstrap=True)

step2 = VotingRegressor([('rf', rf), ('gbdt', gbdt), ('xgb', xgb), ('et', et)], weights=[5, 1, 1, 1])

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

print('R2 score', r2_score(y_test, y_pred))
print('MAE', mean_absolute_error(y_test, y_pred))


# # stacking

# In[111]:


from sklearn.ensemble import VotingRegressor,StackingRegressor

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')


estimators = [
    ('rf', RandomForestRegressor(n_estimators=350,random_state=3,max_samples=0.5,max_features=0.75,max_depth=15)),
    ('gbdt',GradientBoostingRegressor(n_estimators=100,max_features=0.5)),
    ('xgb', XGBRegressor(n_estimators=25,learning_rate=0.3,max_depth=5))
]

step2 = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=100))

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # exporting the model

# In[112]:


import pickle
pickle.dump(data,open('mydata.pkl','wb'))
pickle.dump(pipe,open('mypipe.pkl','wb'))


# In[113]:


data


# In[ ]:





# In[ ]:




