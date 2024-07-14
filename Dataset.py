import cv2
import numpy as np
import os
import pickle 

video = cv2.VideoCapture(0)
cass_path = "C:/Users/nithi/Desktop/facereg/haarcascade_frontalface_default.xml"
facedetect = cv2.CascadeClassifier(cass_path)

face_data = []
i = 0  # Initialize frame counter
name = input("Enter name:")

while True:
    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w]
        resize_img = cv2.resize(crop_img, (50, 50))

        # Check if face_data has less than 100 images and i is a multiple of 10
        if len(face_data) < 100 and i % 10 == 0:
            face_data.append(resize_img)  # Append resized image to face_data
            cv2.putText(frame, str(len(face_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
        
        i += 1  # Increment frame counter

    cv2.imshow("frame", frame)
    k = cv2.waitKey(1)
    if len(face_data) == 100:  # Stop capturing after 100 images
        break

video.release()
cv2.destroyAllWindows()

face_data = np.array(face_data)
face_data = face_data.reshape(100, -1)  # Ensure face_data is of shape (100, -1)

if not os.path.exists('data'):
    os.makedirs('data')  # Create 'data' directory if it does not exist

# Save or append the name data
if "names.pkl" not in os.listdir('data/'):
    names = [name] * 100
    with open("data/names.pkl", "wb") as f:
        pickle.dump(names, f)
else:
    with open('data/names.pkl', 'rb') as f:
        names = pickle.load(f)
    names = names + [name] * 100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

# Save or append the face data
if "face_data.pkl" not in os.listdir('data/'):
    with open('data/face_data.pkl', 'wb') as f:
        pickle.dump(face_data, f)
else:
    with open('data/face_data.pkl', 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, face_data, axis=0)  # Append new face data to existing data
    with open('data/face_data.pkl', 'wb') as f:
        pickle.dump(faces, f)
