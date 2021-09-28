import numpy as np
import pandas as pd
#import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels    # library for regression models
import statsmodels.api as sm
import sklearn       # library for regression models
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from plotly.offline import iplot
from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor

desired_width=320
pd.set_option('display.width',desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',40)

day=pd.read_csv("day.csv",encoding = "ISO-8859-1")
print(day.head(5))

print(day.shape) # to see rows and columns in a table
print(day.info())  #to see if there are any missing values in the dataset
print(day.describe()) # for getting the summary statistics of the dataset

bike=day
print(bike.shape)

drop_col=['instant','dteday','atemp','casual','registered']
bike=bike.drop(drop_col,axis=1)
print(bike.shape)
#visualise the dataset
sns.pairplot(bike) #pair plot to see numeric variables
#plt.show()

#boxplot for target and categorical variable
plt.figure(figsize=(20,12))
plt.subplot(2,3,1)
sns.boxplot(x='season',y='cnt',data=day)
plt.subplot(2,3,2)
sns.boxplot(x='yr',y='cnt',data=day)
plt.subplot(2,3,3)
sns.boxplot(x='mnth',y='cnt',data=day)
plt.subplot(2,3,4)
sns.boxplot(x='weekday',y='cnt',data=day)
plt.subplot(2,3,5)
sns.boxplot(x='workingday',y='cnt',data=day)
plt.subplot(2,3,6)
sns.boxplot(x='weathersit',y='cnt',data=day)
#plt.show()

print(bike.dtypes)

#DUMMY VARIABLES
#Create Dummy variables for 4 categorical variables- mnth, 'weekday','season' and 'weathersit
#Before creating the dummy variables, convert these to category

bike['season']=bike['season'].astype('category')
bike['mnth']=bike['mnth'].astype('category')
bike['weekday']=bike['weekday'].astype('category')
bike['weathersit']=bike['weathersit'].astype('category')
print(bike.dtypes)

#Now create dummy variables
dum_col=['season','mnth','weekday','weathersit']
dum_var=pd.get_dummies(bike[dum_col],drop_first=True)
print(bike.head(15))

#concat dum_var with existing bike dataset
bike=pd.concat([bike,dum_var],axis=1)
#print(bike.dtypes)

bike=bike.drop(['mnth','season','weekday','weathersit'],axis=1)
print(bike.head(5))

#Correlation matrix
plt.figure(figsize=(25,20))
sns.heatmap(bike.corr(),annot=True,cmap='RdBu')
#sns.heatmap(bike_new.corr(), annot = True, cmap="RdBu")
#plt.show()

#Splitting into train test dataset
bike_train,bike_test=train_test_split(bike,train_size=0.7,random_state=100)
print(bike_train.shape)
print(bike_test.shape)

#Rescaling of dataset
#1.Instantiate an object
scaler=MinMaxScaler()

#Scale only numeric variables and not binary variables
#create a list of only numeric variables
num_vars=['yr','holiday','workingday','temp','hum','windspeed','cnt']

#Fit on data
#scaler.fit() #This will learn max and min value when you fit it on data , thus, learns xmax, xmin
#transform(): x-xmin/xmax-xmin
#fit_transform()

#scaler.fit_transform()
bike_train[num_vars]=scaler.fit_transform(bike_train[num_vars])
print(bike_train.head(5))
print(bike_train[num_vars].describe()) #to check if the numeric variables are between 0 and 1

#Model Building or Training the model
#Heatmap
plt.figure(figsize=(18,14))
sns.heatmap(bike_train.corr(),annot=True,cmap='YlGnBu')
#plt.show()

#Model Building- Approach 1
#X_train,y_train
y_train=bike_train.pop('cnt')
X_train=bike_train

#print(X_train.head(3))
#print(y_train.head(3))

#add a const
X_train_sm=sm.add_constant(X_train['temp'])

#create the 1st model
lr=sm.OLS(y_train,X_train_sm)
lr_model=lr.fit()
print(lr_model.params)
print(lr_model.summary())

#add another variable 'yr' and fit the model
X_train_sm=X_train[['temp','yr']]
X_train_sm=sm.add_constant(X_train_sm)
lr=sm.OLS(y_train,X_train_sm)
lr_model=lr.fit()
print(lr_model.params)
print(lr_model.summary())

#Adding all variables to the Model
print(bike.columns)

#Build a model with all variables
X_train_sm=sm.add_constant(X_train)

#create model
lr=sm.OLS(y_train,X_train_sm)
lr_model=lr.fit()
print(lr_model.summary())

#VIF
vif=pd.DataFrame()
vif['Features']=X_train.columns
vif['VIF']=[variance_inflation_factor(X_train.values,i) for i in range(X_train.shape[1])]
vif['VIF']=round(vif['VIF'],2)
vif=vif.sort_values(by='VIF',ascending=False)
print(vif)

#Remove hum variable based on high VIF
a=['hum','weekday_4','weekday_5','weekday_3','weekday_2','weekday_1','mnth_11','mnth_12','mnth_4','mnth_6','mnth_7','holiday']
x=X_train.drop(a,axis=1)

#create another model
X_train_sm=sm.add_constant(x)
#create model
lr=sm.OLS(y_train,X_train_sm)
lr_model=lr.fit()
print(lr_model.summary())

#Lets Check VIF
vif=pd.DataFrame()
vif['Features']=x.columns
vif['VIF']=[variance_inflation_factor(x.values,i) for i in range(x.shape[1])]
vif['VIF']=round(vif['VIF'],2)
vif=vif.sort_values(by='VIF',ascending=False)
print(vif)

#Drop unwanted columns
b=['season_3','mnth_2','mnth_5']
x1=x.drop(b,axis=1)

#create another model
X_train_sm=sm.add_constant(x1)
#create model
lr=sm.OLS(y_train,X_train_sm)
lr_model=lr.fit()
print(lr_model.summary())

#Lets Check VIF
vif=pd.DataFrame()
vif['Features']=x1.columns
vif['VIF']=[variance_inflation_factor(x1.values,i) for i in range(x1.shape[1])]
vif['VIF']=round(vif['VIF'],2)
vif=vif.sort_values(by='VIF',ascending=False)
print(vif)

#Drop unwanted columns
x2=x1.drop('mnth_3',axis=1)

#create another model
X_train_sm=sm.add_constant(x2)
#create model
lr=sm.OLS(y_train,X_train_sm)
lr_model=lr.fit()
print(lr_model.summary())

#Lets Check VIF
vif=pd.DataFrame()
vif['Features']=x2.columns
vif['VIF']=[variance_inflation_factor(x2.values,i) for i in range(x2.shape[1])]
vif['VIF']=round(vif['VIF'],2)
vif=vif.sort_values(by='VIF',ascending=False)
print(vif)

#Drop unwanted columns
x3=x2.drop('mnth_10',axis=1)

#create another model
X_train_sm=sm.add_constant(x3)
#create model
lr=sm.OLS(y_train,X_train_sm)
lr_model=lr.fit()
print(lr_model.summary())

#Lets Check VIF
vif=pd.DataFrame()
vif['Features']=x3.columns
vif['VIF']=[variance_inflation_factor(x3.values,i) for i in range(x3.shape[1])]
vif['VIF']=round(vif['VIF'],2)
vif=vif.sort_values(by='VIF',ascending=False)
print(vif)


#Drop unwanted columns
x4=x3.drop('mnth_8',axis=1)

#create another model
X_train_sm=sm.add_constant(x4)
#create model
lr=sm.OLS(y_train,X_train_sm)
lr_model=lr.fit()
print(lr_model.summary())

#Lets Check VIF
vif=pd.DataFrame()
vif['Features']=x4.columns
vif['VIF']=[variance_inflation_factor(x4.values,i) for i in range(x4.shape[1])]
vif['VIF']=round(vif['VIF'],2)
vif=vif.sort_values(by='VIF',ascending=False)
print(vif)

#Residual Analysis
y_train_pred=lr_model.predict(X_train_sm)
res=y_train-y_train_pred
sns.displot(res)
#plt.show()

#Predictions and Evaluation on Test set
##Before making predictions, we need to follow the same pre-processings steps for test set as we did on training set

num_vars=['yr','holiday','workingday','temp','hum','windspeed','cnt']

#scaler.fit_transform()
bike_test[num_vars]=scaler.transform(bike_test[num_vars])
print(bike_test.head(5))
print(bike_test[num_vars].describe()) #to check if the numeric variables are between 0 and 1

y_test=bike_test.pop('cnt')
X_test=bike_test

#add a const to X_test to make predictions
X_test_sm=sm.add_constant(X_test)
#print(X_test_sm.head(6))

d=['hum','weekday_4','weekday_5','weekday_3','weekday_2','weekday_1','mnth_11','mnth_12','mnth_4','mnth_6','mnth_7','holiday',
   'mnth_8','mnth_10','season_3','weekday_1','weekday_2','weekday_3','weekday_4','weekday_5','mnth_2','mnth_3','mnth_5']

X_test_sm=X_test_sm.drop(d,axis=1)
#print(X_test_sm.head(6))

#make predictions
y_test_pred=lr_model.predict(X_test_sm)

#evaluate the model
r2=r2_score(y_true=y_test,y_pred=y_test_pred)
print(r2)

#Model Evaluation
fig = plt.figure()
plt.scatter(y_test, y_test_pred, alpha=.5)
fig.suptitle('y_test vs y_test_pred', fontsize = 20)              # Plot heading
plt.xlabel('y_test', fontsize = 18)                          # X-label
plt.ylabel('y_pred', fontsize = 16)
plt.show()
