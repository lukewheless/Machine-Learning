from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Data Frames
classes =  pd.read_csv("animal_classes.csv", usecols=["Class_Number", "Class_Type"])
df1 = pd.DataFrame(classes)
train =  pd.read_csv("animals_train.csv")
df2 = pd.DataFrame(train)
test =  pd.read_csv("animals_test.csv")

#New Column
df1.sort_index().sort_index(axis=1) == df2.sort_index().sort_index(axis=1)

df2['Target'] = np.where(df1['Class_Number'] == df2['class_number'], df1['Class_Type'])
                                                   
print(train)          


'''
for i in train["class_number"]:
    if i == 1:
        train["Target"] = "Mammal"
    if i == 2:
        train["Target"] = "Bird"
    if i == 3:
        train["Target"] = "Reptile"
    if i == 4:
        train["Target"] = "Fish"
    if i == 5:
        train["Target"] = "Amphibian"
    if i == 6:
        train["Target"] = "Invertebrate"

print(train)


conditions = ["Mammal" if i == 1
                    "Bird" if i == 2
                    "Reptile" elif i == 3
                    "Fish" elif i == 4
                    "Amphibian" elif i == 5
                    "Bug" elif i == 6
                    "Invertebrate" elif i == 7
                    for i in train["class_number"]]
train["Target"] = np.

x_train, x_test, y_train, y_test = train_test_split(train.Target.values, test.Name.values, random_state=11)

lr = LinearRegression()

lr.fit(X=x_train, y=y_train)

predicted = lr.predict(x_test)

expected = y_test

for p,e in zip(predicted[::5], expected[::5]):
    #print(f"predicted: {p: .2f}, expected: {e: .2f}")

df = pd.DataFrame(predicted, expected, columns = "Predicted", "Expected")
df.to_csv("predictions.csv")
print('Done!')
'''