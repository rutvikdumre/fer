
__author__ = "Rutvi Pramod Dumre"
# HCI CSE 528 SBU Final Project
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2

def overlay_transparent(background, overlay, x, y):

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background


emotion =  ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

model = keras.models.load_model("my_model.h5")
font = cv2.FONT_HERSHEY_SIMPLEX
cam = cv2.VideoCapture(0)
face_cas = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')

while True:
    ret, frame = cam.read()
    
    if ret==True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #gray = cv2.flip(gray,1)
        faces = face_cas.detectMultiScale(gray, 1.3,5)
        pred={}

        pred['Neutral'] = cv2.imread("nue.png")
        pred['Happy']= cv2.imread("happy.png")
        pred['Disgust']= cv2.imread('disgust.png')
        pred['Anger']= cv2.imread('anger.png')
        pred['Fear']= cv2.imread('fear.png')
        pred['Sad']= cv2.imread('sad.png')
        pred['Surprise']= cv2.imread('surprise.png')

        for (x, y, w, h) in faces:
            face_component = gray[y:y+h, x:x+w]
            fc = cv2.resize(face_component, (48, 48))
            inp = np.reshape(fc,(1,48,48,1)).astype(np.float32)
            inp = inp/255.
            prediction = model.predict(inp)
            em = emotion[np.argmax(prediction)]
            score = np.max(prediction)
            cv2.putText(frame, em+"  "+str(score*100)+'%', (x, y), font, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            try:
                frame= overlay_transparent(frame,pred[em],0,0)
            except:
                frame= overlay_transparent(frame,pred['Neutral'],0,0)
        cv2.imshow("image", frame)
        
        if cv2.waitKey(1) == 27:
            break
    else:
        print ('Error')

cam.release()
cv2.destroyAllWindows()
