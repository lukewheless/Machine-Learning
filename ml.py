from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_digits
import pandas as pd    
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2

digits = load_digits 

#print(digits.DESCR)
#print(digits.data[:2])
#print(digits.data.shape)
#print(digits.target[100:120])
#print(digits.target.shape)
#print(digits.images[:13])

fig, axes = plt.subplots(nrows = 4, ncols = 6, figsize = (6,4))

#takes 3 iterable things and interates through all of them
for item in zip(axes.ravel(), digits.images, digits.target):
    axes, image, target = item
    axes.imshow(image, cmap = plt.cm.gray_r)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(target)

plt.tight_layout()
plt.show()

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, random_state=11)
#data = x's    and   target = y's

print(x_train.shape) #data (1347, 64)
print(y_train.shape) #target (1347,0)
print(x_test.shape)
print(y_test.shape)

knn = KNeighborsClassifier()

knn.fit(X = x_train, Y = y_train)

predicted = knn.predict(X = x_test) #predicts target

expected = y_test

print(predicted[:20])
print(expected[:20])

#11/2/20 
wrong = [(p,e) for (p,e in zip (predicted, expected): if p != e] #iterates through two lists and compares if they are not equal values

print(format(knn.score(x_test,y_test), '.2%'))

cf = confusion_matrix(y_true=expected, y_pred=predicted)

cf_df = pd.DataFrame(cf, index=range(10), columns=range(10)) # 0 - 10

fig = plt2.figure(figsize = (7,6))
axes = sns.heatmap(cf_df, annot = True, cmap = plt2.cm.nipy_spectral_r)
plt2.show()















