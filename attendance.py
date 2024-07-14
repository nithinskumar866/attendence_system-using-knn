import cv2
import numpy as np
import os
import csv
import time
import pickle

from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime

video = cv2.VideoCapture(0)
cass_path = r"C:\Users\nithi\Desktop\facereg\haarcascade_frontalface_default.xml"
facedetect = cv2.CascadeClassifier(cass_path)

with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)

with open('data/face_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Ensure 'bg.png' exists in the specified path or adjust the path accordingly
pic_path = r'C:\Users\nithi\Desktop\facereg\bg.png'
imgbackground = cv2.imread(pic_path)
if imgbackground is None:
    raise FileNotFoundError(f"Could not open or read 'bg.png'. Please check the file path.")

COL_NAMES = ['NAME', 'TIME']

S = 1

# Create the attendance directory if it does not exist
attendance_dir = 'attendance'
os.makedirs(attendance_dir, exist_ok=True)

attendance_recorded = False  # To ensure attendance is recorded only once

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w, :]
        resize_img = cv2.resize(crop_img, (50, 50))  # Corrected resizing of crop_img directly
        resize_img = resize_img.flatten().reshape(1, -1)  # Flattening and reshaping after resize

        output = knn.predict(resize_img)
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime('%d-%m-%y')
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2.putText(frame, text=str(output[0]), org=(x, y - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(50, 50, 255), thickness=2)
        attendance = [str(output[0]), str(timestamp)]
        imgbackground[162:162 + 480, 55:55 + 640] = frame

    cv2.imshow('frame', imgbackground)
    k = cv2.waitKey(1)
    if k == ord('t'):
        time.sleep(S)
        print("Saving attendance...")

        # Write the attendance data to the file
        attendance_file = f"{attendance_dir}/attendance_{date}.csv"
        if not os.path.isfile(attendance_file):
            with open(attendance_file, "w", newline='') as csvfile:  # Use 'w' for writing header
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)
                print(f"Created new attendance file and added attendance: {attendance}")
        else:
            with open(attendance_file, "a", newline='') as csvfile:  # Use newline='' to avoid blank lines
                writer = csv.writer(csvfile)
                writer.writerow(attendance)
                print(f"Appended attendance: {attendance}")

        # Mark attendance as recorded
        attendance_recorded = True

        # Break the loop and stop the webcam after saving attendance
        break

    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

# Inform the user if attendance was not recorded
if not attendance_recorded:
    print("No attendance recorded.")
else:
    print("Attendance successfully recorded.")
