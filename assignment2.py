import pandas as pd   
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
import numpy as np  
import matplotlib.pyplot as plt

nyc = pd.read_csv('ave_yearly_temp_nyc_1895-2017.csv')

x_train, x_test, y_train, y_test = train_test_split(
    nyc.Date.values.reshape(-1,1), nyc.Value.values, random_state=11)

lr = LinearRegression()

lr.fit(X=x_train, y=y_train)

predicted = lr.predict(x_test)

expected = y_test

for p,e in zip(predicted[::5], expected[::5]):
    print(f"predicted: {p: .2f}, expected: {e: .2f}")


predict = (lambda x: lr.coef_ * x + lr.intercept_) #mx + b

print(predict(2021))

axes = sns.scatterplot(
    data=nyc,
    x="Date",
    y="Value",
    hue="Value",
    palette="winter",
    legend=False
)

axes.set_ylim(10,70)

x = np.array([min(nyc.Date.values), max(nyc.Date.values)])
print(x)

y = predict(x)
print(y)

line = plt.plot(x,y)
plt.show()

'''
When comparing the two regression models we 
see the high temperatures in January below 40 
degrees, however, the other model representing December 
looks to be above 50 degrees over the years with less variation
along the regression line

'''