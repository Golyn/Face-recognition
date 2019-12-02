from joblib import load
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()

#loading recognizer 
recognizer.read('recognizer.yaml')
#loading the labels
labels = load('labels.joblib')

#dictionaries have key and value, for eg. Eugene is the key and 0 is the value
# makes the value(the ID comes before the key)
labels = {v:k for k,v in labels.items() }
# the key is now the ID and the name is the value

cap = cv2.VideoCapture(0)
seth_counter = 6

while True:
    check, frame = cap.read()  

#the recognizer can pic the image if it's in grayscale
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.05)

    for x,y,w,h in faces:
        roi = gray_image[y:y+h, x:x+w]
#conf is the confidence in the prediction
        id,conf = recognizer.predict(roi)

# Conf is how sure the person's ID assigned to it
# if the confidence is more than 50% it should predict it to that confident level
#and it means that the image is recognized and predicted
        if conf >=50 and conf <=100:
            #gives the image a caption
            cv2.putText(frame,labels[id], (x,y), cv2.FONT_HERSHEY_COMPLEX,2, cv2.LINE_AA)

            if labels[id] == "seth":
                print('Hello,' + labels[id] + 'you are signed in')

            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

        cv2.imshow('Real time face recognition', frame)    

    if cv2.waitKey(0) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()        

