import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing


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


pipe = make_pipeline(StandardScaler(),LogisticRegression()
)

#le = preprocessing.LabelEncoder()# need to transform strings to int
#x_train['Sex']=le.fit(x_train['Sex'])
def df_encoder(column,df):
  le = preprocessing.LabelEncoder()
  le.fit(df[column])
  df[column] = le.transform(df[column])
x_df = pd.DataFrame()
y_df = pd.DataFrame()
y_df['Survived']= train_df['Survived']
x_df = train_df.drop(columns=['Survived','Ticket', 'Name', 'Cabin'])
df_encoder('Sex',x_df)
#df_encoder('Name',x_df)
#df_encoder('Ticket',x_df)
#df_encoder('Cabin',x_df)
df_encoder('Embarked',x_df)
#.789 with logistic regression, .7982, .8026



#le_sex = preprocessing.LabelEncoder()
#le_sex.fit(x_train['Sex'])
#x_train['Sex'] = le_sex.transform(x_train['Sex'])
x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, random_state=42)

pipe.fit(x_df, y_df)
##account for missing age
#x_test['Age'] = x_test['Age'].astype('str')

#x_test['Age'] = x_test['Age'].astype('float')
#x_test['Fare'].fillna(value=x_test['Fare'].mean(), inplace=True)
#accuracy_score(pipe.predict(x_test), y_test['Survived']) #.9186 accuracy score
df_submission = pd.read_csv('datasets//test.csv')
df_encoder('Sex',df_submission)
df_encoder('Embarked',df_submission)
df_submission['Age'] = df_submission.apply(fill_missing_age,axis=1)
df_submission = df_submission.drop(columns=['Ticket', 'Name', 'Cabin'])
df_submission['Fare'].fillna((df_submission['Fare'].mean()), inplace=True)
df_submission['Survived'] = pipe.predict(df_submission)
df_final_est = pd.DataFrame()
df_final_est['PassengerId'] = df_submission['PassengerId']
df_final_est['Survived'] = df_submission['Survived']
df_final_est.to_csv('jv_submission_7_19_21.csv')
# regr = linear_model.LinearRegression()
# fitted = regr.fit(x_train, y_train)

