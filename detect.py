import cv2
import numpy as np
import face_recognition
import pickle

# Load the face encodings and names from the file
with open('face_encodings.pkl', 'rb') as f:
    known_face_encodings, known_face_names = pickle.load(f)

# Specify the paths to the model files
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Function to process frames
def process_frame(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, 
                                 (300, 300), (104.0, 177.0, 123.0))
    
    net.setInput(blob)
    detections = net.forward()
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            face_offset = frame[startY:endY, startX:endX]

            if face_offset.size > 0:
                rgb_face = cv2.cvtColor(face_offset, cv2.COLOR_BGR2RGB)
                face_encodings = face_recognition.face_encodings(rgb_face)

                if face_encodings:  # Ensure at least one face encoding is found
                    face_names = []
                    for face_encoding in face_encodings:
                        # Calculate face distances and determine the best match
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        
                        # Logging distances for debugging
                        # print(f"Face distances: {face_distances}")
                        # print(f"Best match index: {best_match_index}")
                        # print(f"Best match distance: {face_distances[best_match_index]}")

                        # Set a threshold for face distance to determine a match
                        if face_distances[best_match_index] < 0.5:  # Adjust threshold as needed
                            matches = face_recognition.compare_faces([known_face_encodings[best_match_index]], face_encoding)
                            if matches[0]:
                                name = known_face_names[best_match_index]
                            else:
                                name = "Unknown"
                        else:
                            name = "Unknown"

                        face_names.append(name)

                    for name in face_names:
                        cv2.putText(frame, name, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
    return frame

cap = cv2.VideoCapture(0)  # Change to 0 to use the webcam
skip_frames = 5  

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    if frame_count % skip_frames == 0:
        processed_frame = process_frame(frame)
        cv2.imshow("Face Detection", processed_frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
