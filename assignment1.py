from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('max_columns', 9)
pd.set_option('precision', 4)
pd.set_option('display.width', None)

cali = fetch_california_housing()

cali_df = pd.DataFrame(cali.data, columns=cali.feature_names)
cali_df['MedHouseValue'] = pd.Series(cali.target)           

cali_df = cali_df.sample(frac=0.1, random_state=17)

sns.set(font_scale=1.1)
sns.set_style('whitegrid')
grid = sns.pairplot(data=cali_df, vars=cali_df.columns[:8], hue='MedHouseValue') 
plt.show()