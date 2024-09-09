import pickle

import numpy as np
import cv2
import os

from keras.layers import Dropout,Dense,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam


test_ratio=0.2
valid_ratio=0.2
imagesDimensions=(32,32,3)
batchSizeVal=50
epochsVal=10
stepsPerEpochsVal=2000
path="myData"
pathlabel="label.csv"
images=[]
myList=os.listdir(path)
noOfClass=len(myList)
classNo=[]

for x in range(0,noOfClass):
    myPiList=os.listdir(path+"/"+str(x))
    for y in myPiList:
        curImg=cv2.imread(path+"/"+str(x)+"/"+y)
        curImg=cv2.resize(curImg,(32,32))
        images.append(curImg)
        classNo.append(x)

images=np.array(images)
classNo=np.array(images)

x_train,x_test,y_train,y_test=train_test_split(images,classNo,test_size=test_ratio)
x_train,x_validation,y_train,y_validation=train_test_split(x_train,y_train,test_size=valid_ratio)

def preprocess(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img)
    img=img/255
    return img

img=preprocess(x_train[30])
img=cv2.resize(img,(300,300))
cv2.imshow("preprocess",img)
cv2.waitKey(0)

x_train=np.array(list(map(preprocess(x_train))))
x_test=np.array(list(map(preprocess(x_test))))
x_validation=np.array(list(map(preprocess(x_validation))))

x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_test=x_train.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
x_validation=x_train.reshape(x_validation.shape[0],x_validation.shape[1],x_validation.shape[2],1)

dataGen=ImageDataGenerator(width_shift_range=0.1,
                           height_shift_range=0.1,
                           zoom_range=0.2,
                           shear_range=0.1,
                           rotation_range=10)
dataGen.fit(x_train)
y_train=to_categorical(y_train,noOfClass)

def MyModel():
    noOfFilter=60
    sizeOfFilter1=(5,5)
    sizeOfFilter2 = (3,3)
    sizeOfPool=(2,2)
    noOfNode=500

    model=Sequential()
    model.add((Conv2D(noOfFilter,sizeOfFilter1,input_shape=(imagesDimensions[0],
                                                            imagesDimensions[1],
                                                            1),activation='relu')))
    model.add((Conv2D(noOfFilter, sizeOfFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(noOfFilter//2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(noOfFilter // 2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNode,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfNode, activation='softmax'))
    model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    return model

model=MyModel()


history=model.fit_generator(dataGen.flow(x_train,y_train,
                                         batch_size=batchSizeVal,
                                         steps_per_epoch=stepsPerEpochsVal,
                                         epochs=epochsVal,
                                         validation_data=(x_validation,y_validation)),
                            shuffle=1)


score=model.evaluate(x_test,y_test,verbose=0)
pickle_out=open("model_train.p","wb")
pickle.dump(model,pickle_out)
pickle_out.close()