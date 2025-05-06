import cv2
import os
from datetime import datetime
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# Folder to save cropped faces
output_folder = 'dataset/cropped_faces'
os.makedirs(output_folder, exist_ok=True)

# Start webcam
cap = cv2.VideoCapture(0)
face_id = input("Enter name for this person: ")
person_folder = os.path.join(output_folder, face_id)
os.makedirs(person_folder, exist_ok=True)

frame_count = 0
max_faces = 50

print("Press 's' to save face, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 expects BGR image (OpenCV format)
    results = model(frame, verbose=False)[0]

    # Loop through detections
    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Crop face
        face_crop = frame[y1:y2, x1:x2]

        # Save on key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and face_crop.size > 0 and frame_count < max_faces:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'{face_id}_{timestamp}.jpg'
            face_path = os.path.join(person_folder, filename)
            cv2.imwrite(face_path, face_crop)
            frame_count += 1
            print(f"[✔] Saved face {frame_count}/{max_faces}")

    # Show webcam feed
    cv2.imshow('gather datasets', frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or frame_count >= max_faces:
        break

cap.release()
cv2.destroyAllWindows()
