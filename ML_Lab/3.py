import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0)
print(" k-NN Classification Algorithm")
print(" Dataset: Social Networking Ads")
print('\nDataSet', dataset)
print('\nTraining data', X_train)
print('\n Testing  data', X_test)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print('\n Purchase prediction', y_pred)

cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
print('Confusion matrix', cm)
print('Accuracy', ac)