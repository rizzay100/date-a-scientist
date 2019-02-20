
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("profiles.csv")
df.fillna('no_answer', inplace=True)

body_type_map = {
"no_answer": 0,
"rather not say": 1,
"used up": 2,
"overweight": 3,
"curvy": 4,
"full figured": 5,
"a little extra": 6,
"skinny": 7,
"thin": 8,
"average": 9,
"fit": 10,
"athletic": 11,
"jacked": 12
}

drug_type_map = {
"no_answer": 0,
"never": 1,
"sometimes": 2,
"often": 3
}

drink_type_map = {
"no_answer": 0,
"not at all": 1,
"rarely": 2,
"socially": 3,
"often": 4,
"very often": 5,
"desperately": 6
}

smoke_type_map = {
"no_answer": 0,
"no": 1,
"trying to quit": 2,
"when drinking": 3,
"sometimes": 4,
"yes": 5
}

diet_type_map = {
"no_answer": 0,
"strictly vegan": 1,
"vegan": 2,
"mostly vegan": 3,
"strictly vegetarian": 4,
"vegetarian": 5,
"mostly vegetarian": 6,
"mostly anything": 7,
"anything": 8,
"strictly anything": 9
}

education_type_map = {
"no_answer": 0,
"working on high school	": 1,
"graduated from high school	":1,
"high school":1,
"two-year college":2,
"working on two-year college":2,
"graduated from two-year college":2,
"working on college/university	":3,
"college/university	":3,
"graduated from college/university	":3,
"working on masters program	":4,
"graduated from masters program	":4,
"masters program":	4,
"working on law school	":5,
"working on med school	":5,
"working on ph.d program":5,
"graduated from law school"	:5,
"graduated from med school"	:5,
"graduated from ph.d program":5,
"law school	":5,
"med school	":5,
"ph.d program":5
}

attributes = df[["diet", "drinks", "drugs", "body_type", "smokes", "income", "age","education"]]
attributes["diet_code"] = attributes.diet.map(diet_type_map)
attributes["drinks_code"] = attributes.drinks.map(drink_type_map)
attributes["smokes_code"] = attributes.smokes.map(smoke_type_map)
attributes["drugs_code"] = attributes.drugs.map(drug_type_map)
attributes["body_code"] = attributes.body_type.map(body_type_map)
attributes["education_code"] = attributes.education.map(education_type_map)
attributes.dropna(inplace=True)
attributes = attributes[attributes.income != -1]
attributes.drop(labels=["diet", "drinks", "drugs", "body_type", "body_code", "smokes", "education"], axis=1, inplace=True)
feature_values = attributes.values

