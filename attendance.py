import mysql.connector

# Connect to MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="2804",
    database="attendance_system"
)
cursor = conn.cursor()

cursor.execute("SELECT * FROM attendance_records")
rows = cursor.fetchall()

print("\n📌 Attendance Records:")
print("---------------------------------------------------------")
print("| S.NO.         | NAME            | Timestamp           |")
print("---------------------------------------------------------")
for row in rows:
    print(f"| {row[0]:<13} | {row[1]:<15} | {row[2]} |")
print("---------------------------------------------------------")

cursor.close()
conn.close()
