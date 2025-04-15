import cv2
import pickle
import numpy as np
import mysql.connector
from datetime import datetime

# Load Haarcascade face detector
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load Name and Registration Number Data
with open('data/names.pkl', 'rb') as name_file, open('data/reg_no.pkl', 'rb') as reg_file:
    names = pickle.load(name_file)
    reg_nos = pickle.load(reg_file)

# Load Face Data
with open('data/faces_data.pkl', 'rb') as face_file:
    faces = pickle.load(face_file).astype(np.float32)  # Convert to float32 for better matching

# Initialize MySQL connection
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="2804",
    database="attendance_system"
)
cursor = conn.cursor()

# Prevent duplicate attendance in one session
attendance_taken = set()

def store_attendance(name, reg_no):
    """Store attendance in MySQL only if 'O' is pressed and not already recorded in this session."""
    if reg_no not in attendance_taken:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        query = "INSERT INTO attendance_records (name, reg_no, timestamp) VALUES (%s, %s, %s)"
        cursor.execute(query, (name, reg_no, timestamp))
        conn.commit()
        attendance_taken.add(reg_no)
        print(f"✅ Attendance stored: {name} | {reg_no} | {timestamp}")
    else:
        print(f"⚠ {name} ({reg_no}) already marked present in this session.")

# Start video capture
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Camera issue.")
        break

    # Face Detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = facedetect.detectMultiScale(gray, 1.3, 5)

    detected_person = None

    for (x, y, w, h) in faces_detected:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1).astype(np.float32)

        # Identify person
        index = np.argmin(np.linalg.norm(faces - resized_img, axis=1))
        name = names[index]
        reg_no = reg_nos[index]

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display name on top of face
        cv2.rectangle(frame, (x, y - 30), (x + w, y), (0, 255, 0), -1)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        detected_person = (name, reg_no)

    cv2.imshow("Smart Attendance System", frame)

    key = cv2.waitKey(1)
    
    if key == ord('o') and detected_person:  # Press 'O' to mark attendance
        store_attendance(*detected_person)

    if key == ord('q'):  # Press 'Q' to quit
        break

video.release()
cv2.destroyAllWindows()
cursor.close()
conn.close()
