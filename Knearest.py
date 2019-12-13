import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn import svm
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style
style.use('ggplot')


df = pd.read_csv('breast-cancer-wisconsin.data.txt')

df.replace('?',-99999, inplace=True)
df.drop(['id'],1, inplace=True)


X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=42)

clf = KNeighborsClassifier()

clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))
example_measures = np.array([7,6,6,3,2,10,7,1,1])
example_measures = example_measures.reshape(1, -1)
prediction = clf.predict(example_measures)
print(prediction)
