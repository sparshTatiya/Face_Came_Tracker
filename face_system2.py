import os
import cv2
import sqlite3
import numpy as np
import urllib.request

from datetime import datetime


class FaceSystem:
    def __init__(self, db_path="face_database.db", confidence_threshold=0.5,
    similarity_threshold=0.6, models_dir="models"):
        self.db_path = db_path
        self.confidence_threshold = confidence_threshold
        self.similarity_threshold = similarity_threshold 
        self.models_dir = models_dir

        # Intialize database
        self._init_database()

        # Download models if they don't exist 
        self._ensure_models_exists()

        # Load face detection model (DNN based face detector)
        prototxt_path = os.path.join(self.models_dir, "deploy.prototxt")
        caffemodel_path = os.path.join(self.models_dir, "res10_300x300_ssd_iter_140000.caffemodel")

        # Load face detection face Model (DNN based face detector)
        self.face_detector = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

        # Load face recoginition model (face embeddings generator)
        openface_path = os.path.join(self.models_dir, "openface_nn4.small2.v1.t7")
        self.face_recognizer = cv2.dnn.readNetFromTorch(openface_path)

    def _ensure_models_exists(self):
        models = {
            "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
            "res10_300x300_ssd_iter_140000.caffemodel": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
            "openface_nn4.small2.v1.t7": "https://github.com/pyannote/pyannote-data/raw/master/openface.nn4.small2.v1.t7"
        }

        for model_file, url in models.items():
            model_path = os.path.join(self.models_dir, model_file)

            if not os.path.exists(model_path):
                print(f"Downloading {model_file}...")
                try:
                    urllib.request.urlretrieve(url, model_path)
                    print(f"\033[34;1mDownloaded {model_file} successfully.\033[0;0m")
                except Exception as e:
                    print(f"Failed to download {model_file}: {e}")
                    print(f"Please download {model_file} manually from {url} and save it {model_path}")

    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # create tables if they don't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER,
            embedding BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (person_id) REFERENCES persons (id)                          
        )
        ''') 

        conn.commit()
        conn.close()

    def detect_faces(self, image):
        h, w = image.shape[:2]

        # Prepare image for face detection
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )

        # Detect Faces
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()

        face_boxes = []

        # Process detection results
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self.confidence_threshold:
                # Get face bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                start_x, start_y, end_x, end_y = box.astype("int")

                # Ensure box is within image boundaries
                start_x, start_y = max(0, start_x), max(0, start_y)
                end_x, end_y = min(w, end_x), min(h, end_y)

                # Calculate width and height
                face_width = end_x - start_x
                face_height = end_y - start_y

                # Only include face with reasonable dimensions
                if face_width > 20 and face_height > 20:
                    face_boxes.append((start_x, start_y, face_width, face_height))

        return face_boxes
    
    def extract_face_embedding(self, image, face_box):
        x, y, w, h = face_box
        face_roi = image[y:y+h, x:x+w]

        # Prepare face for embedding extraction
        face_blob = cv2.dnn.blobFromImage(
            face_roi, 1.0/255, (96, 96),
            (0, 0, 0), swapRB=True, crop=False
        )

        # Extract Embedding 
        self.face_recognizer.setInput(face_blob)
        embedding = self.face_recognizer.forward()

        return embedding.flatten()
    
    def register_face(self, image, face_box, person_name):
        # Extract face Embeddings 
        embedding = self.extract_face_embedding(image, face_box)

        # Connect to a database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if the person already exists
        cursor.execute("SELECT id FROM persons WHERE name = ?", (person_name,))
        result = cursor.fetchone()

        if result:
            person_id = result[0]
        else:
            # Insert new person
            cursor.execute("INSERT INTO persons (name) VALUES (?)", (person_name,))
            person_id = cursor.lastrowid 

        # Insert embedding 
        embedding_bytes = embedding.tobytes()
        cursor.execute(
            "INSERT INTO face_embeddings (person_id, embedding) VALUES (?, ?)",
            (person_id, embedding_bytes)
        )

        conn.commit()
        conn.close()

        return person_id
    
    def recognize_face(self, image, face_box):
        # Extract face embeddings
        query_embedding = self.extract_face_embedding(image, face_box)

        # Connect to database 
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get all embeddings
        cursor.execute("""
SELECT fe.id, fe.person_id, fe.embedding, p.name
FROM face_embeddings fe 
JOIN persons p ON fe.person_id = p.id                      
""")
        
        best_match = None
        highest_similarity = -1

        for row in cursor.fetchall():
            _, person_id, embedding_bytes, person_name = row

            # Convert the bytes into Numpy array 
            db_embeddings = np.frombuffer(embedding_bytes, dtype=np.float32)

            # Calculate cosine similarity
            similarity = np.dot(query_embedding, db_embeddings) / (np.linalg.norm(query_embedding) * 
            np.linalg.norm(db_embeddings))

            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = (person_id, person_name, similarity)

        conn.close()

        if best_match and best_match[2] > self.similarity_threshold:
            return best_match
        
        return None
    
    def get_all_persons(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT id, name FROM persons")
        persons = cursor.fetchall()

        conn.close()

        return persons
    
    def draw_face_box(self, image, face_box, label=None, color=(255, 0, 0)):
        x, y, w, h = face_box
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)

        if label:
            cv2.putText(image, label, (x, y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return image
    

def main():
    # Intialize face system 
    face_system = FaceSystem()

    # Open webcam 
    cap = cv2.VideoCapture(0)

    # Operation modes 
    mode = "detection" # "detection" or "registration"
    registration_name = ""

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Detect Faces 
        face_boxes = face_system.detect_faces(frame)

        for face_box in face_boxes:
            if mode == "detection":
                # Try to recognize the face 
                recognition_result = face_system.recognize_face(frame, face_box)

                if recognition_result:
                    person_id, person_name, similarity = recognition_result
                    label = f"{person_name} ({similarity:.2f})"
                    face_system.draw_face_box(frame, face_box, label, (0, 0, 255))
                else:
                    face_system.draw_face_box(frame, face_box, "Unknown", (0, 0, 255))

            elif mode == "registration" and registration_name:
                face_system.draw_face_box(frame, face_box, "Register: " + registration_name, (255, 0, 0))

        
        # Show current mode 
        cv2.imshow("Face Detection and Recongnition:", frame)

        # Handle key events 
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("d"):
            mode = "detection"
        elif key == ord("r"):
            mode = "registration"
            registration_name = input("Please enter name for registration: ")
        elif key == ord("s") and mode == "registration" and registration_name and face_boxes:
            # Register the first detected face 
            person_id = face_system.register_face(frame, face_box=face_boxes[0], 
            person_name=registration_name)
            print(f"Registered {registration_name} with ID {person_id}.")
            mode = "detection"

    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()