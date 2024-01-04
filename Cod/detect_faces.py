import cv2


def is_there_a_face(imagePath,cascPath= 'face_cascade.xml'):


    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)

    # Read the image
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    return len(faces)

    # Draw a rectangle around the faces
    #for (x, y, w, h) in faces:
    #    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

imagePath = 'triplete_fete/1/1.jpg'

#print(is_there_a_face(imagePath))
import os

for i in os.listdir('triplete_fete'):
    for j in os.listdir(os.path.join('triplete_fete',i)):
        print(is_there_a_face(os.path.join('triplete_fete',i,j)))
    break

