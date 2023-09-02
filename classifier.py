import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df=pd.read_csv("/content/mlops/data/Iris.csv")
features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
taget = 'Species'

X_train, X_test, Y_train, Y_test = train_test_split(df[features], df[taget], test_size=0.3, shuffle=True)
#step 1: initialise the model class
clf = DecisionTreeClassifier(criterion="entropy")
#step 2: train the model on training set
clf.fit(X_train, Y_train)
#evaluate the data on testing set
Y_pred = clf.predict(X_test)

print(f"Accuracy of the model is {accuracy_score(Y_test,Y_pred)*100}")
# this is test accuracy
