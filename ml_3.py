from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

cali = fetch_california_housing()

pd.set_option('max_columns', 9)
pd.set_option('precision', 4)
pd.set_option('display.width', None)

cali_df = pd.DataFrame(cali.data, columns=cali.feature_names)
cali_df['MedHouseValue'] = pd.Series(cali.target)
cali_df = cali_df.sample(frac = 0.1, random_state = 17)           
print(cali_df.head())

for feature in cali.feature_names:
    plt.figure(figsize=(16, 9))
    sns.set(font_scale=2)
    sns.set_style('whitegrid')
    sns.scatterplot(data=cali_df,
                    x=feature,
                    y='MedHouseValue',
                    hue='MedHouseValue',
                    palette='cool',
                    legend=False)
plt.show()

x_train, x_test, y_train, y_test = train_test_split(cali.data, cali.target, random_state=11)

lr = LinearRegression()

lr.fit(X=x_train, y=y_train)

predicted = lr.predict(x_test)

expected = y_test

for p,e in zip(predicted[::5], expected[::5]):
    print(f"predicted: {p: .2f}, expected: {e: .2f}")