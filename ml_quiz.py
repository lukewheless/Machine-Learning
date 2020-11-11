from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Data Frames
classes =  pd.read_csv("animal_classes.csv", usecols=["Class_Number", "Class_Type"])

train =  pd.read_csv("animals_train.csv")

test =  pd.read_csv("animals_test.csv")

#New Column
train["Target"] = ["Mammal" if i == 1
                    "Bird" elif i == 2
                    "Reptile" elif i == 3
                    "Fish" elif i == 4
                    "Amphibian" elif i == 5
                    "Bug" elif i == 6
                    "Invertebrate" elif i == 7
                    for i in train["Class_Number"]]
print(train)

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