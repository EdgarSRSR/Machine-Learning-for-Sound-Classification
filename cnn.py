# This file works with Convolutional neural network to create a model that gets fed with images of spectograms
# to train it to differentiate between spectrograms of UAV (Unmaned Aerial Vehicle) and of other sound sources
# This file has two cnn models, one untrained and another based on Keras library pre-trained model vg166.
# To make these models works there has to be three depositories one for testing, one for training and another
# for validation. Each depository should have two depositories inside, one with jpg images of spectrograms of UAV
# another with images of spectrograms that are not UAVs. This model is done wit the the idea of recognizing between
# two options. This file also analysies the results and plots a confusion matrix.

import numpy as np
import keras
from keras import backend as k
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import itertools
import matplotlib.pyplot as plt
#%matplotlib inline

train_path = 'soundtest/train'
valid_path = 'soundtest/valid'
test_path = 'soundtest/test'
# 106 total samples 73 drones 27 not drones
# valid samples: 72 drones 27 not drones
# 8 test 4 drones 4 no drones
train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224, 224), classes=['drones', 'notdrones'], batch_size = 7)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224, 224), classes=['drones', 'notdrones'], batch_size = 7)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224, 224), classes=['drones', 'notdrones'], batch_size = 8)

def plots(ims, figsize = (12,8), rows=1, interp = False, titles=None):
	if type(ims[0]) is np.ndarray:
		ims = np.array(ims).astype(np.uint8)
		if (ims.shape[-1] != 3):
			ims = ims.transpose((0,2,3,1))
	f = plt.figure(figsize=figsize)
	cols = len(ims)// rows if len(ims) % 2 == 0 else len(ims)//rows + 1
	for i in range(len(ims)):
		sp = f.add_subplot(rows, cols, i + 1)
		sp.axis('Off')
		if titles is not None:
			sp.set_title(titles[i], fontsize=16)
		plt.imshow(ims[1], interpolation= None if interp else 'none')
	plt.show()

imgs, labels = next(train_batches)
plots(imgs, titles=labels)

# this is a simple CNN model without pre training, it is not as precise as the one from Keras below, comment this section out if you don't
# want to use it, since it is there only for learning purposes
model = Sequential([
	Conv2D(32,(3, 3), activation='relu', input_shape=(224,224,3)),
	Flatten(),
	Dense(2, activation='softmax'),
	])
model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
steps_per_epoch is calculated by the total amount of samples/ train_batches size
model.fit_generator(train_batches, steps_per_epoch=7,
	validation_data=valid_batches, validation_steps=7, epochs=5, verbose=2)
#tests usng the model
test_imgs, test_labels = next(test_batches)
plots(test_imgs, titles=test_labels)

test_labels = test_labels[:,0]
print(test_labels)
# creates predictions using the model
predictions = model.predict_generator(test_batches, steps=1, verbose=0)
print(predictions)
#creates a confusion matrix for the model
cm = confusion_matrix(test_labels, predictions[:,0])

# a function that creates a confusion matrix
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

#cm_plot_labels = ['drones','notdrones']
#plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')


# this is another CNN model that is more powerful than the one above, it is pre trained and come from a Kerala library
vgg16_model = keras.applications.vgg16.VGG16()
vgg16_model.summary()
type(vgg16_model)
model = Sequential()
for layer in vgg16_model.layers:
	model.add(layer)
model.summary()
model.layers.pop()
model.summary()
for layer in model.layers:
	layer.trainable = False
model.add(Dense(2, activation='softmax'))
model.summary()
model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_batches, steps_per_epoch=5, validation_data = valid_batches, validation_steps = 5, epochs = 6, verbose = 2)

#tests for the model
test_imgs, test_labels = next(test_batches)
plots(test_imgs, titles=test_labels)

test_labels = test_labels[:,0]
print(test_labels)

predictions = model.predict_generator(test_batches, steps=1, verbose=0)
cm_plot_labels = ['drones','notdrones']

y_pred_bool = np.argmax(predictions, axis=1)

#prints report
print(classification_report(y_pred_bool, test_labels ))

#creates confusion matrix
cm = confusion_matrix(test_labels, np.round(predictions[:,0]))

#plots confusion matrix
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')
