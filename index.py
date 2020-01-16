import cv2
from keras.models import load_model
import numpy as np 
from keras.preprocessing.image import img_to_array

orig_frame = cv2.imread('test.jpg')
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
emotion_detector = load_model('_mini_XCEPTION.106-0.65.hdf5',compile=False)

EMOTIONS = ["angry","disgust","scared", "happy", "sad", "surprised","neutral"]

frame = cv2.imread('test.jpg',0)

faces = face_detector.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)

for face in faces:
    (fX, fY, fW, fH) = face
    roi = frame[fY:fY + fH, fX:fX + fW]
    roi = cv2.resize(roi, (48, 48))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    preds = emotion_detector.predict(roi)[0]
    emotion_probability = np.max(preds)
    label = EMOTIONS[preds.argmax()]
    cv2.putText(orig_frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.rectangle(orig_frame, (fX, fY), (fX + fW, fY + fH),(0, 0, 255), 2)

cv2.imshow('Emotions Detector',orig_frame)
cv2.imwrite('output.jpg',orig_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()