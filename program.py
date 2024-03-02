import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
import dlib

# Constants
KNOWN_FACES_DIR = "photos/"
CSV_FILE_PREFIX = "attendance"
CSV_FILE_EXTENSION = "csv"

# Function to initialize known faces and encodings
def initialize_known_faces():
    negi_image = face_recognition.load_image_file(KNOWN_FACES_DIR + "negi.jpg")
    negi_encoding = face_recognition.face_encodings(negi_image)[0]

    dhama_image = face_recognition.load_image_file(KNOWN_FACES_DIR + "dhama.jpg")
    dhama_encoding = face_recognition.face_encodings(dhama_image)[0]

    rajeev_image = face_recognition.load_image_file(KNOWN_FACES_DIR + "rajeev.jpg")
    rajeev_encoding = face_recognition.face_encodings(rajeev_image)[0]

    shreya_image = face_recognition.load_image_file(KNOWN_FACES_DIR + "shreya.jpg")
    shreya_encoding = face_recognition.face_encodings(shreya_image)[0]

    shubham_image = face_recognition.load_image_file(KNOWN_FACES_DIR + "shubham.jpg")
    shubham_encoding = face_recognition.face_encodings(shubham_image)[0]
    
    shyam_image = face_recognition.load_image_file(KNOWN_FACES_DIR + "shyam.jpg")
    shyam_encoding = face_recognition.face_encodings(shyam_image)[0]

    # Add more faces as needed

    known_face_encoding = [negi_encoding, dhama_encoding, rajeev_encoding, shreya_encoding, shubham_encoding, shyam_encoding]  # Add more as needed
    known_faces_names = ["Saurabh Negi", "Saurabh Dhama", "Rajeev Verma", "Shreya Dixit", "Shubham Bhatt", "Shyam K.G."]  # Add more as needed

    return known_face_encoding, known_faces_names

# Function to capture video frames and perform face recognition
def capture_and_recognize(detector, known_face_encoding, known_faces_names, students, csv_writer):
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()

        if not ret:
            print("Error capturing video.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
            name = ""
            face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
            best_match_index = np.argmin(face_distance)

            if matches[best_match_index]:
                name = known_faces_names[best_match_index]

            if name and name in students:
                students.remove(name)
                print(students)
                current_time = datetime.now().strftime("%H-%M-%S")
                csv_writer.writerow([name, current_time])

            for face in faces:
                x, y = face.left(), face.top()
                hi, wi = face.right(), face.bottom()
                cv2.rectangle(frame, (x, y), (hi, wi), (0, 0, 255), 2)
                cv2.putText(frame, f"{name}", (x-12, y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Attendance System", frame)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Function to create a CSV file for attendance
def create_csv_file():
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    file_name = f"{CSV_FILE_PREFIX}_{current_date}.{CSV_FILE_EXTENSION}"
    return open(file_name, 'w+', newline='')

if __name__ == "__main__":
    detector = dlib.get_frontal_face_detector()
    known_face_encoding, known_faces_names = initialize_known_faces()
    students = known_faces_names.copy()

    csv_file = create_csv_file()
    csv_writer = csv.writer(csv_file)

    try:
        capture_and_recognize(detector, known_face_encoding, known_faces_names, students, csv_writer)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        csv_file.close()