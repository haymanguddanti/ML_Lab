from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

msg = pd.read_csv('messagetext.csv', names=['message', 'label'])
print('Total instances in the dataset:', msg.shape[0])
msg['labelnum'] = msg.label.map({'pos': 1, 'neg': 0})
X = msg.message
Y = msg.labelnum

xtrain, xtest, ytrain, ytest = train_test_split(X, Y)
print('\nDataset is split into Training and Testing samples')
print('Total training instances :', xtrain.shape[0])
print('Total testing instances :', xtest.shape[0])
count_vect = CountVectorizer()
xtrain_dtm = count_vect.fit_transform(xtrain)
xtest_dtm = count_vect.transform(xtest)

clf = MultinomialNB().fit(xtrain_dtm, ytrain)
predicted = clf.predict(xtest_dtm)
print('\nAccuracy metrics')
print('Accuracy of the classifer is', metrics.accuracy_score(ytest, predicted))
print('Recall :', metrics.recall_score(ytest, predicted))
print('\nPrecison :', metrics.precision_score(ytest, predicted))
print('Confusion matrix', metrics.confusion_matrix(ytest, predicted))
