import pandas as pd
import matplotlib as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


df=pd.read_csv("/home/kukanishka/Downloads/train.csv")   #reading of training data into dataframe
df.head(10)


# In[ ]:


df.apply(lambda x: sum(x.isnull()),axis=0)


# In[ ]:


df.describe()


# In[ ]:


df['Property_Area'].value_counts()


# In[ ]:


df['LoanAmount'].fillna(df['LoanAmount'].mean(),inplace=True)  #Filling the missing data with mean of the entire column
df['LoanAmount'][0:10]  


# In[ ]:


df['Self_Employed'].value_counts()


# In[ ]:


df['Self_Employed'].fillna('No',inplace=True)  #Filling missing data in favour of majority
df['Self_Employed'].value_counts()


# In[ ]:


df['ApplicantIncome'].hist(bins=50)


# In[5]:


new_data=df.dropna(axis=0)  #Dropping of rows whose values are 'NaN' or '0' due to their non-numerical nature
new_data                    #This may cause poor results due to reduction in dataset


# In[16]:


var=['Education','Gender','Married','Dependents','Self_Employed','Property_Area','Loan_Status']
LE=LabelEncoder()       #Labelling of non-numerical values in terms of '0' and '1'  as in '0 for No and 1 for Yes'
for i in var:
    new_data[i]=LE.fit_transform(new_data[i])
new_data[:10]


# In[97]:


predictors=['Credit_History','ApplicantIncome','Education','Married','Dependents','Self_Employed','Property_Area']
X=new_data[predictors]
y=new_data.Loan_Status
my_model=RandomForestClassifier(n_estimators=90)
train_X,val_X,train_Y,val_Y=train_test_split(X,y,random_state=0)
my_model.fit(train_X,train_Y)
prediction=my_model.predict(val_X)
accuracy=metrics.accuracy_score(prediction,val_Y)
print (accuracy)

