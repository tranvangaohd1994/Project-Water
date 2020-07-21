from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
from models import *
import numpy as np
import os
import glob
from scipy.stats import entropy
import cv2
from keras.utils.np_utils import to_categorical
img_width = 32
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
def resize2SquareKeepingAspectRation(img, size, interpolation):
    h, w = img.shape[:2]
    c = None if len(img.shape) < 3 else img.shape[2]
    if h == w: return cv2.resize(img, (size, size), interpolation)
    if h > w: dif = h
    else:     dif = w
    x_pos = int((dif - w)/2.)
    y_pos = int((dif - h)/2.)
    if c is None:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
    return cv2.resize(mask, (size, size), interpolation)

# Training parameters
epochs = 200
data_augmentation = True
num_classes = 10

# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True

n = 3

version = 2

if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2

# Model name, depth and version
model_type = 'ResNet%dv%d' % (depth, version)
input_shape = (img_width,img_width,3)
batch_size = 128
num_classes = 10

train_data_dir = "/home/vti/Documents/CRAFT-pytorch-master/classed"
X = []
Y = []

for root, subdirectories, files in os.walk(train_data_dir):
    for subdirectory in subdirectories:
        pathsub = os.path.join(root, subdirectory)
        listfiles = glob.glob(pathsub + "/*.jpg")
        for fil in listfiles:
            img = cv2.imread(fil).astype('float32') / 255
            img = resize2SquareKeepingAspectRation(img,img_width,cv2.INTER_AREA)
            X.append(img)
            Y.append([int(subdirectory)])
X = np.asarray(X)
Y = np.asarray(Y)

def preData():
    global X, Y
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X = X[idx]
    Y = Y[idx]
    (x_train, y_train), (x_test, y_test) = (X[:4000], Y[:4000]),(X[4000:],Y[4000:])

    # x_train_mean = np.mean(x_train, axis=0)
    # x_train -= x_train_mean
    # x_test -= x_train_mean
    # Convert class vectors to binary class matrices.
    y_test = to_categorical(y_test, num_classes)
    y_train = to_categorical(y_train, num_classes)
    # remove the initial data from the training dataset
    # remove the initial data from the training dataset
    
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = preData()

def lr_schedule(epoch):
    lr = 0.01
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    return lr


if version == 2:
    model = resnet_v2(input_shape=input_shape, depth=depth)
else:
    model = resnet_v1(input_shape=input_shape, depth=depth)

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=lr_schedule(0), decay=1e-5, momentum=0.9),
              metrics=['accuracy'])
model.summary()
print(model_type)

# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

filepath = os.path.join(save_dir, model_name)


lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)
checkpoint = ModelCheckpoint("resnet_retrain.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
callbacks = [lr_scheduler,lr_reducer,checkpoint]


datagen = ImageDataGenerator(
    )

model.load_weights("/home/vti/Documents/OCR_CAPCHA/resnet_retrain.h5")
# model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
#                          validation_data=(x_test, y_test),
#                         epochs=epochs, verbose=1, workers=4,
#                         steps_per_epoch = len(x_train)//batch_size,
#                         callbacks=callbacks)

# scores = model.evaluate(x_test, y_test, verbose=1)
# print(scores)
keras.models.load_model("resnet_retrain_1.h5")

for img in X:
    result = np.argmax(model.predict(np.array([img])))
    print(str(result))
    cv2.imshow("a",img)
    cv2.waitKey()