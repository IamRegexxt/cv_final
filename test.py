# recognize_faces.py
import cv2
import pickle
import face_recognition
import numpy as np

# Load trained classifier
with open('classifier.pkl', 'rb') as f:
    model, le = pickle.load(f)

cap = cv2.VideoCapture(0)

print("🔍 Press 'q' to quit.")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    frame_count += 1

    if frame_count % 2 == 0:  # skip every other frame for speed
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        scaled_locations = []
        for (top, right, bottom, left) in face_locations:
            scaled_locations.append((top*2, right*2, bottom*2, left*2))

        recognized_faces = []
        for face_encoding in face_encodings:
            preds = model.predict_proba([face_encoding])[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j] if proba > 0.6 else "Unknown"
            recognized_faces.append((name, proba))

    # draw detections from last processed frame
    if 'scaled_locations' in locals():
        for (loc, (name, proba)) in zip(scaled_locations, recognized_faces):
            top, right, bottom, left = loc
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f'{name} ({proba:.2f})', (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
