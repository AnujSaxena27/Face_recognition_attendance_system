import tkinter as tk
from tkinter import messagebox, scrolledtext
import cv2
import pickle
import numpy as np
import os
import mysql.connector
from datetime import datetime

# MySQL Connection
conn = mysql.connector.connect(
    host="localhost", 
    user="root",
    password="2804",
    database="attendance_system"
)
cursor = conn.cursor()

# Load face detector
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ------------------------ Add New Face ------------------------ #
def add_face():
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    faces_data = []
    i = 0

    name = name_entry.get()
    reg_no = reg_entry.get()

    if not name or not reg_no:
        messagebox.showwarning("Missing Info", "Please enter both name and registration number.")
        return

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

        cv2.imshow("Adding Face", frame)
        if cv2.waitKey(1) == ord('q') or len(faces_data) == 100:
            break

    video.release()
    cv2.destroyAllWindows()

    faces_data = np.asarray(faces_data).reshape(100, -1)

    # Save files
    names_file = 'data/names.pkl'
    reg_no_file = 'data/reg_no.pkl'
    faces_file = 'data/faces_data.pkl'

    if not os.path.exists(names_file):
        names = [name] * 100
        reg_nos = [reg_no] * 100
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

    if not os.path.exists(faces_file):
        with open(faces_file, 'wb') as f:
            pickle.dump(faces_data, f)
    else:
        with open(faces_file, 'rb') as f:
            faces = pickle.load(f)
            faces = np.append(faces, faces_data, axis=0)
        with open(faces_file, 'wb') as f:
            pickle.dump(faces, f)

    messagebox.showinfo("Success", "✅ Face data successfully saved!")

# ------------------------ Mark Attendance ------------------------ #
def mark_attendance():
    with open('data/names.pkl', 'rb') as name_file, open('data/reg_no.pkl', 'rb') as reg_file:
        names = pickle.load(name_file)
        reg_nos = pickle.load(reg_file)

    with open('data/faces_data.pkl', 'rb') as face_file:
        faces = pickle.load(face_file).astype(np.float32)

    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    attendance_taken = set()

    while True:
        ret, frame = video.read()
        if not ret:
            print("Error: Camera issue.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = facedetect.detectMultiScale(gray, 1.3, 5)

        detected_person = None

        for (x, y, w, h) in detected_faces:
            crop_img = frame[y:y+h, x:x+w]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1).astype(np.float32)
            index = np.argmin(np.linalg.norm(faces - resized_img, axis=1))
            name = names[index]
            reg_no = reg_nos[index]
            detected_person = (name, reg_no)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y - 30), (x + w, y), (0, 255, 0), -1)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Marking Attendance", frame)
        key = cv2.waitKey(1)

        if key == ord('o') and detected_person:
            # Check if attendance for the student is already marked today
            cursor.execute("""
                SELECT COUNT(*) FROM attendance_records
                WHERE reg_no = %s AND DATE(timestamp) = CURDATE()
            """, (detected_person[1],))
            result = cursor.fetchone()

            if result[0] == 0:  # No attendance recorded yet
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                try:
                    query = "INSERT INTO attendance_records (name, reg_no, timestamp) VALUES (%s, %s, %s)"
                    cursor.execute(query, (*detected_person, timestamp))
                    conn.commit()
                    attendance_taken.add(detected_person[1])
                    print(f"✅ Attendance stored: {detected_person[0]} | {detected_person[1]} | {timestamp}")
                except mysql.connector.Error as err:
                    print(f"Error: {err}")
            else:
                print("⚠️ Duplicate attendance detected. Skipping entry.")
        elif key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


# ------------------------ View Attendance ------------------------ #
def view_attendance():
    cursor.execute("SELECT id, name, timestamp FROM attendance_records ORDER BY id DESC")
    rows = cursor.fetchall()

    win = tk.Toplevel(root)
    win.title("📋 Recorded Attendance")
    text = scrolledtext.ScrolledText(win, width=80, height=20)
    text.pack()

    text.insert(tk.END, "S.NO.  | NAME       | TIMESTAMP\n")
    text.insert(tk.END, "-" * 60 + "\n")

    for row in rows:
        text.insert(tk.END, f"{row[0]:<6} | {row[1]:<10} | {row[2]}\n")

# ------------------------ Exit ------------------------ #
def exit_app():
    cursor.close()
    conn.close()
    root.quit()

# ------------------------ GUI ------------------------ #
root = tk.Tk()
root.title("Smart Attendance System")

tk.Label(root, text="Name:").grid(row=0, column=0, padx=10, pady=10)
name_entry = tk.Entry(root)
name_entry.grid(row=0, column=1)

tk.Label(root, text="Reg No:").grid(row=1, column=0, padx=10)
reg_entry = tk.Entry(root)
reg_entry.grid(row=1, column=1)

tk.Button(root, text="➕ Add Face", width=20, command=add_face).grid(row=2, column=0, columnspan=2, pady=10)
tk.Button(root, text="📷 Mark Attendance", width=20, command=mark_attendance).grid(row=3, column=0, columnspan=2, pady=10)
tk.Button(root, text="📄 View Attendance", width=20, command=view_attendance).grid(row=4, column=0, columnspan=2, pady=10)
tk.Button(root, text="❌ Exit", width=20, command=exit_app).grid(row=5, column=0, columnspan=2, pady=10)

root.mainloop()
