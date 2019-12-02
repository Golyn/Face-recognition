#navigating directory
import os
import cv2
#importing images from PILLOW
from PIL import Image
import numpy as np
#joblib import dump for saving the models
from joblib import dump

#give us access to the base directory (face recognition folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#joining the base folder (FACE-RECOGNITION) to the images folder to get the full path
image_dir = os.path.join(BASE_DIR, 'Images')

#face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

#creating empty dictionary for storing the persons name and ID. The ID is unique
# starting the ID at 0
current_id = 0
labels_id = {
    
}

features = []
labels = []

#root is the images folder, dirs is the folder of Eugene, Rita and Seth
#os.walk(image_dir) goes through the files to get the images
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith('jpg') or file.endswith('jpeg'):
            #joining the image files to the images folder
            path = os.path.join(root, file)
            #getting the label names(Eugene, Rita,Seth)
            label = os.path.basename(os.path.dirname(path)).lower()
            #print(label,path)

            #If no label, then create a label ID for it
            if not label in labels_id:
                labels_id[label] = current_id
                current_id +=1

               # print(labels_id)

            id = labels_id[label] 

            # loading the images and converting into array 
            image = Image.open(path).convert('L')  
            #the image itself
            #converting the images into a numpy array 
            image_array = np.array(image, 'uint8')

            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.05)
            # getting the face region of interest
            for x,y,w,h in faces:
                roi = image_array[y: y+h, x: x+w]
                #appending the region of interest of the faces to the list
                features.append(roi)
                labels.append(id)

dump(labels_id,'labels.joblib')

# calling or importing OpenCv model that recognizes the face
# initializing the recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
# training the recognizer
recognizer.train(features, np.array(labels))

recognizer.save('recognizer.yaml')