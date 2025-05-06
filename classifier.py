# extract_embeddings.py
import os
import cv2
import numpy as np
import pickle
import face_recognition

dataset_dir = 'dataset/cropped_faces'
embeddings = []
labels = []

for person_name in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, person_name)
    if not os.path.isdir(person_dir):
        continue

    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        image = cv2.imread(img_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb, model='hog')
        if len(face_locations) == 0:
            continue  # skip if no face detected

        encodings = face_recognition.face_encodings(rgb, face_locations)

        for encoding in encodings:
            embeddings.append(encoding)
            labels.append(person_name)

print(f"Collected {len(embeddings)} face embeddings.")

# Save to file
with open('embedd.pkl', 'wb') as f:
    pickle.dump({'embeddings': embeddings, 'labels': labels}, f)

print("pkl saved")
