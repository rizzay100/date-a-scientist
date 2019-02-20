
#Height and Age vs. Income

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
X = df_age_income[['age','height']]
y = df_age_income[['income']]


#Split data between training and test
x_train, x_test, y_train, y_test  = train_test_split(X, y, test_size = 0.2, random_state = 1)

#Plot fit the data
regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)
y_predict = regr.predict(x_test)


#Find score to test for veracity
print("Train score:")
print(regr.score(x_train, y_train))
print("Test score:")
print(regr.score(x_test, y_test))

#Describe the plot
plt.ylabel("Income")
plt.xlabel("Height & Age ")
plt.title('Height & Age vs. Income')

#Make the plot
plt.scatter(y_test, y_predict)
plt.plot(range(20000), range(20000))
plt.show()

