# This file takes Mfcc data from a csv and uses it to train a SVM classifier to detect if
# the audio has sound from an UAV(Unmaned Aerial Vehicle) of from another audio source.
# This file also calculate the accuracy of the model and plots a confussion matrix that helps to
# measure the accuracy of the model. The libray used for the creation of the SVM model is sklearn.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.model_selection import cross_val_score #new
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from scipy.sparse import coo_matrix

#reads the data placed in the csv file, by audiodataextraction.py
data = pd.read_csv("results.csv")
shape = data.shape
print(shape)
head = data.head()
print(head)


#divide data in attributes

x = data.drop('label', axis=1)
y = data['label']
print("this is x pre shuffle")
print(x.head())
print("this is y pre shuffle")
print(y.head())
#shuffle
X_sparse = coo_matrix(x)
x, X_sparse, y = shuffle(x, X_sparse, y, random_state=0)
print("this is x after shuffle")
print(x.head())
print("this is y after shuffle")
print(y.head())
#divide data into training and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20) 

#training
svclassifier = SVC(kernel='linear')
#svclassifier = SVC(kernel='poly', degree=8)
svclassifier.fit(x_train, y_train)

#cross validation
#shows the classes in Y
print(np.unique(y))
scores = cross_val_score(svclassifier, x_train, y_train,cv=2, scoring='accuracy')
#gives a score for cross validation
print(scores)
print("Mean of accuracy of Cross Validation: {}".format(np.mean(scores)))

#predicting
y_pred = svclassifier.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print(cm)  
print(classification_report(y_test,y_pred))  

#plt.plot(x_test,y_test, linestyle='-')
#plt.plot(x_train,y_train, linestyle=':')
#plt.show()

# creates confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False,
						title= 'Confusion matrix', 
						cmap = plt.cm.Blues):

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion marix, without normalization')

	print(cm)

	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i,j],
			horizontalalignment = "center",
			color="white" if cm[i, j] > thresh else "black")
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.show()

cm_plot_labels = ['drones','notdrones']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')
