import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import numpy


from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
#practical

gender_df = pd.read_csv('datasets//gender_submission.csv')
gender_df.head()
gender_df.nunique() # PassengerID 418/ survived 2
gender_df.PassengerId.count() # 418, indicates only unique passenger IDs 
gender_df.groupby(['Survived']).count() # 0:266, 1: 152 
print(train_df.describe())

train_df = pd.read_csv('datasets//train.csv')
train_df.columns # Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
     #  'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
   #   dtype='object')
   #survived 0 or 1 
   #Pclass 1, 2 or 3
   #Sex male, or female
   #SibSp 0-8
   #Parch: 0-6
   ##Fair: Float, 15 paid no fair (248 distinct values)
   ##ticket some are just Ints, some have characters W./C. or W.E.P etc (681 distinct values)
   ## bar chart with survived vs other column counts
   #cabin letter+int or just Letter
 #some ages are missing, think of replacing missing data with an average based on gender or Miss , Mrs., Master for kids
 #males below 15 are named Master. 
average_mrs_age = train_df[train_df['Name'].str.contains('Mrs.')]['Age'].mean()
average_master_age = train_df[train_df['Name'].str.contains('Master')]['Age'].mean()
average_mr_age = train_df[train_df['Name'].str.contains('Mr.')]['Age'].mean()
average_miss_age = train_df[train_df['Name'].str.contains('Miss.')]['Age'].mean()
def fill_missing_age(row):
  if pd.isna(row['Age']):
    if 'Mrs.' in row['Name']:
      return average_mrs_age
    elif 'Master' in row['Name']:
      return average_master_age
    elif 'Mr.' in row['Name']:
      return average_mr_age
    else:
      return average_miss_age
  else:
    return row['Age']


train_df['Age'] = train_df.apply(fill_missing_age,axis=1)
train_df.hist()
#scatter_matrix(train_df)

pipe = make_pipeline(StandardScaler(),LogisticRegression()
)
y_train = pd.DataFrame()
y_train['Survived'] = train_df['Survived']
x_train = pd.DataFrame()
x_train = train_df
x_train=x_train.drop('Survived', axis=1)

#le = preprocessing.LabelEncoder()# need to transform strings to int
#x_train['Sex']=le.fit(x_train['Sex'])
def df_encoder(column,df):
  le = preprocessing.LabelEncoder()
  le.fit(df[column])
  df[column] = le.transform(df[column])

df_encoder('Sex',x_train)
df_encoder('Name',x_train)
df_encoder('Ticket',x_train)
df_encoder('Cabin',x_train)
df_encoder('Embarked',x_train)

x_test = pd.read_csv('datasets//test.csv')
x_test['Age'] = x_test.apply(fill_missing_age,axis=1)
df_encoder('Sex',x_test)
df_encoder('Name',x_test)
df_encoder('Ticket',x_test)
df_encoder('Cabin',x_test)
df_encoder('Embarked',x_test)
y_test = pd.DataFrame()
y_test = gender_df


#le_sex = preprocessing.LabelEncoder()
#le_sex.fit(x_train['Sex'])
#x_train['Sex'] = le_sex.transform(x_train['Sex'])

pipe.fit(x_train, y_train)
##account for missing age
#x_test['Age'] = x_test['Age'].astype('str')

#x_test['Age'] = x_test['Age'].astype('float')
x_test['Fare'].fillna(value=x_test['Fare'].mean(), inplace=True)
accuracy_score(pipe.predict(x_test), y_test['Survived']) #.9186 accuracy score
df_submission = pd.DataFrame()
df_submission['PassengerId'] = x_test['PassengerId']
df_submission['Survived'] = pipe.predict(x_test)