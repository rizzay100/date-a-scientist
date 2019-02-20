#Import libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from sklearn import linear_model

from sklearn.model_selection import train_test_split

#Create Dataframe
df_age_income = pd.read_csv("profiles.csv")
df_age_income['income'].replace(-1, np.nan,inplace=True)
df_age_income['income'].replace('income', np.nan,inplace=True)
df_age_income['income'].replace('no', np.nan,inplace=True)
df_age_income.dropna(axis=0, how='any',inplace=True)


#Normalize and extract age and name
#Average_income_by_age = df.groupby('age').income.mean().reset_index()
X = df_age_income['age']
y = df_age_income['income']


#Split data between training and test
x_train, x_test, y_train, y_test  = train_test_split(X, y, test_size = 0.2, random_state = 1)

#Resize independent variables
X_train = x_train.values.reshape(-1, 1)
X_test = x_test.values.reshape(-1, 1)
Y_train = y_train.values.reshape(-1, 1)
Y_test = y_test.values.reshape(-1, 1)

#Plot training data
plt.scatter(X_train, Y_train)
regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)
y_predict = regr.predict(X_train)
plt.xlabel("Age")
plt.ylabel("Income")
plt.plot(X_train, y_predict)


#Predict future dataset with new fitted linear regression
X_future = np.array(range(40, 110))
X_future = X_future.reshape(-1, 1)
future_predict = regr.predict(X_future)
plt.plot(X_future,future_predict, '-', color="g",)

plt.show()

#Print Score
print('Train Score:', regr.score(X_train, y_train))
print('Test Score:', regr.score(X_test, y_test))
