import numpy as np
import cv2
import pickle


width=640
height=480
threshold=0.65

cap=cv2.VideoCapture(1)
cap.set(3,width)
cap.set(4,height)

pickle_in=open("model_trained.p","rb")
model=pickle.load(pickle_in)

def preprocess(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img)
    img=img/255
    return img

while True:
    success,imgOrigin=cap.read()
    img=np.asarray(imgOrigin)
    img=cv2.resize(img,(320,320))
    img=preprocess(img)
    cv2.imshow("precessed img",img)
    img=img.reshape(1,32,32,1)

    classIndex=int(model.predict_classes(img))
    #print(classIndex)
    predictions=model.predict(img)
    probVal=np.amax(predictions)
    if probVal>threshold:
        cv2.putText(imgOrigin,str(classIndex)+" "+str(probVal),
                    (50,50),cv2.FONT_HERSHEY_TRIPLEX,1,(0.255,0),1)

    cv2.imshow("Origin Image",imgOrigin)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break