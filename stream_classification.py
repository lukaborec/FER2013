import numpy as np
import cv2
from keras.models import model_from_json

#Loading the classifier model
#Load the JSON file
json_file = open('/Users/lukaborec/Downloads/model.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

#Load the model weights
model.load_weights('model.h5')

#Expression dictionary
expression_dict = {0:'Angry', 
           1:'Fear', 
           2:'Happy', 
           3:'Sad', 
           4:'Surprise', 
           5:'Neutral'
          }

#Font for writing the expression on the webcam stream
font = cv2.FONT_HERSHEY_SIMPLEX

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Create a VideoCapture object to read live-stream from a camera
cap = cv2.VideoCapture(0) 

while True:
    ret, frame = cap.read() #retval, image
    
    #Operations on the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # type = numpy array
    
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        #Iterate over all of the faces found in the frame
        for (x,y,w,h) in faces:
            #Draw a rectange around the face
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            #Crop each of the faces, resize, and use the classifier to classify the facial expression
            cropped_image = gray[y:y+h,x:x+w]
            cropped_image = cv2.resize(cropped_image, dsize=(48,48))
            reshaped = np.reshape(cropped_image, (1, 48,48,1))
            reshaped = np.divide(reshaped, 255.0)
            #Make a prediction
            expression = np.argmax(model.predict(reshaped))
            cv2.putText(frame,expression_dict[expression],(x+10,y-10), font, 1,(100,100,100),2,cv2.LINE_AA)
               
    #Display the frame
    cv2.imshow('Facial expression classifier', frame)
         
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()


