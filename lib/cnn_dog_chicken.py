from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, adam
from keras.utils import np_utils

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import theano
from PIL import Image
from numpy import *

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
#%%
img_rows, img_cols = 200, 200
#%%
path1 = "G:\Columbia\STAT GR5243\project03\images"
path2 = "G:\Columbia\STAT GR5243\project03\images_resized"

listing = os.listdir(path1)
num_samples = size(listing)
print(num_samples)

for file in listing:
    im = Image.open(path1 + '\\' + file)
    img = im.resize((img_rows, img_cols))
    gray = img.convert('L')
    gray.save(path2 + '\\' + file, "JPEG")

imlist = os.listdir(path2)

im1 = array(Image.open(path2 + '\\' + imlist[0]))
m, n = im1.shape[0:2]
imnbr = len(imlist)

immatrix = array([array(Image.open(path2 + '\\' + im2)).flatten()
              for im2 in imlist],'f')

label = np.ones((num_samples,), dtype = int)
label[0:1000] = 1
label[1000:] = 0

data, Label = shuffle(immatrix, label, random_state = 2)
train_data = [data, Label]

img = immatrix[500].reshape(img_rows, img_cols)
plt.imshow(img)
plt.imshow(img, cmap = 'gray')
print(train_data[0].shape)
print(train_data[1].shape)
#%%
batch_size = 32
nb_classes = 2
nb_epoch = 20
img_channels = 1
nb_filters = 32
nb_pool = 2
nb_conv = 3
#%%
(X, y) = (train_data[0], train_data[1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 4)
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

X_train /= 255
X_test /=255

print("X_train shape:", X_train.shape)
print(X_train.shape[0], "train samples")
print(X_test.shape[0], "test samples")

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

i = 200
plt.imshow(X_train[i, 0], interpolation = "nearest")
print("label: ", Y_train[i, :])
#%%
model = Sequential()
model.add(Convolution2D(nb_filters, nb_conv, nb_conv, 
                        border_mode = 'valid',
                        input_shape = (1, img_rows, img_cols)))
convout1 = Activation('relu')
model.add(convout1)
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
convout2 = Activation('relu')
model.add(convout2)
model.add(MaxPooling2D(pool_size = (nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('sigmoid'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop')
#%%
model.fit(X_train, Y_train, batch_size = batch_size, nb_epoch = nb_epoch, 
          verbose = 1, validation_data = (X_test, Y_test))
#%%
score = model.evaluate(X_test, Y_test, verbose = 0)
print("test score: ", score[0])
print("test accuracy: ", score[1])