import glob
import cv2
from skimage.filters import threshold_local
import numpy as np
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import tensorflow as tf
from models import *
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
img_width, img_height = 32, 32

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

n = 3
version = 2

if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2
input_shape = (img_width,img_width,3)

if version == 2:
    model = resnet_v2(input_shape=input_shape, depth=depth)
else:
    model = resnet_v1(input_shape=input_shape, depth=depth)

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.03, decay=1e-5, momentum=0.9),
              metrics=['accuracy'])

model.load_weights("/home/vti/Documents/OCR_CAPCHA/resnet_retrain.h5")

listimages = glob.glob("/home/vti/Documents/CRAFT-pytorch-master/images/*.jpg")

for i,file in enumerate(listimages):
    img = cv2.imread(file)
    V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))[2]
    T = threshold_local(V, 9, offset=15, method='gaussian')
    thresh = (V > T).astype('uint8') * 255
    thresh = cv2.bitwise_not(thresh)
    blur = cv2.GaussianBlur(thresh,(45,45),0)
    blur = cv2.threshold(blur,40,255,cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hei, wid = img.shape[:2]
    X = []
    corrs = []
    for j, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < 20:
            continue
        y = 0
        x = max(0,x - 3)
        w = min(wid,w+6)
        h = hei
        tmp = img[y:y+h,x:x+w,:]
        tmp = resize2SquareKeepingAspectRation(tmp,img_width,cv2.INTER_AREA)
        # cv2.imwrite(f"/home/vti/Documents/CRAFT-pytorch-master/dumps/1/{i}_{j}.jpg",tmp)
        # tmp = cv2.imread(f"/home/vti/Documents/CRAFT-pytorch-master/dumps/1/{i}_{j}.jpg")
        X.append(tmp)
        corrs.append([x, y, w, h])
    X = np.array(X)
    X = X.astype('float32') / 255

    # x_train_mean = np.mean(X, axis=0)
    # X -= x_train_mean
    result = model.predict(X)
    result = np.argmax(result,axis=1)
    for corr, y_pred in zip(corrs,result):
        x, y, w, h = corr
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
        img = cv2.putText(img, str(y_pred), (x, y+20), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255) , 2, cv2.LINE_AA) 
    # cv2.imshow("img",img)
    cv2.imwrite(f"imgs/{i}.jpg",img)
    # cv2.waitKey()