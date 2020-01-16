### 
**Detects Face using Haarcascades and further detects emotion in bounded face (trained a CNN emotion detector model)**

The full application is explained in the following blog-post. https://appliedmachinelearning.blog/2018/11/28/demonstration-of-facial-emotion-recognition-on-real-time-video-using-cnn-python-keras/

```
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
emotion_detector = load_model('_mini_XCEPTION.106-0.65.hdf5',compile=False)


EMOTIONS = ["angry","disgust","scared", "happy", "sad", "surprised","neutral"]

faces = face_detector.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)

preds = emotion_detector.predict(roi)[0]
```
![output](https://user-images.githubusercontent.com/54398533/72501812-834ac800-3840-11ea-9d33-25deb1c568ac.jpg)
