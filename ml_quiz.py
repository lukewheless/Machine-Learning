from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
knn = KNeighborsClassifier()

#Data Frames
classes =  pd.read_csv("animal_classes.csv", usecols=["Class_Number", "Class_Type"])
classes_df = pd.DataFrame(classes)
train =  pd.read_csv("animals_train.csv") #df without without target
train_df = pd.DataFrame(train)
test =  pd.read_csv("animals_test.csv")
test_df = pd.DataFrame(test)

#train.rename(columns={"class_number":"Class_Number"}, inplace=True)
#merge = pd.merge(train,classes, how='left', on='Class_Number')

train = train_df.iloc[:,:16] #without target class_type
target = train_df.iloc[:,16] #with target
test = test_df.iloc[:,1:] #without animal_name
name = test_df.iloc[:,0] #With just names
types = classes_df.iloc[:,1]

#Train Model
knn.fit(X=train, y=target)
predicted = knn.predict(X=test) #predicts target

l = 0
outfile = open("predictions.csv", 'w')
h = 'animal_name, predicted'
outfile.write(h + '\n')

for p in predicted:
    line = f"{name[l]},{types[int(p)-1]}\n"
    print(line)
    outfile.write(line)
    l += 1