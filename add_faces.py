import cv2
import pickle
import numpy as np
import os

# Ensure data directory exists
if not os.path.exists('data'):
    os.makedirs('data')

video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

faces_data = []
i = 0

# Take user input for Name and Registration Number
name = input("Enter Your Name: ")
reg_no = input("Enter Your Registration Number: ")

while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Unable to access the camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50))
        
        if len(faces_data) < 100 and i % 10 == 0:
            faces_data.append(resized_img)

        i += 1
        cv2.putText(frame, f"Collected: {len(faces_data)}/100", (50, 50), 
                    cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(1) == ord('q') or len(faces_data) == 100:
        break

video.release()
cv2.destroyAllWindows()

faces_data = np.asarray(faces_data).reshape(100, -1)

# Save Name and Registration Number Data
names_file = 'data/names.pkl'
reg_no_file = 'data/reg_no.pkl'
faces_file = 'data/faces_data.pkl'

if not os.path.exists(names_file):
    names = [name] * 100
    reg_nos = [reg_no] * 100
    with open(names_file, 'wb') as f:
        pickle.dump(names, f)
    with open(reg_no_file, 'wb') as f:
        pickle.dump(reg_nos, f)
else:
    with open(names_file, 'rb') as f:
        names = pickle.load(f)
    with open(reg_no_file, 'rb') as f:
        reg_nos = pickle.load(f)

    names.extend([name] * 100)
    reg_nos.extend([reg_no] * 100)

    with open(names_file, 'wb') as f:
        pickle.dump(names, f)
    with open(reg_no_file, 'wb') as f:
        pickle.dump(reg_nos, f)

# Save Face Data
if not os.path.exists(faces_file):
    with open(faces_file, 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open(faces_file, 'rb') as f:
        faces = pickle.load(f)
        faces = np.append(faces, faces_data, axis=0)
    with open(faces_file, 'wb') as f:
        pickle.dump(faces, f)

print("✅ Face data successfully saved!")
