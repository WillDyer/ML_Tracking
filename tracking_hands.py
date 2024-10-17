import cv2
import mediapipe as mp
import socket
import struct
import json
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

#hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def vector(point_1, point_2):
    return np.array([point_2.x - point_1.x, point_2.y - point_1.y, point_2.z - point_1.z])

def hand_orientation(landmarks):
    wrist = landmarks[0]
    index_base = landmarks[5]
    pinky_base = landmarks[17]

    index_vector = vector(wrist, index_base)
    pinky_vector = vector(wrist, pinky_base)
    
    palm_normal = np.cross(index_vector, pinky_vector)
    palm_normal = palm_normal / np.linalg.norm(palm_normal) 

    hand_rotation = np.arctan2(palm_normal[1], palm_normal[0])
    
    return {
        "rotation_x": float(hand_rotation),
        "palm_normal": palm_normal.tolist()
    }

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Failed to capture image")
            break
        
        image.flags.writeable = False
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                rotation_data = hand_orientation(hand_landmarks.landmark)

                landmarks_data = {
                    "landmarks": []
                }

                for landmark in hand_landmarks.landmark:
                    landmarks_data["landmarks"].append({
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z
                    })

                json_data = json.dumps(landmarks_data)
                #json_data = json.dumps(rotation_data)
                #print(json_data)

                sock.sendto(json_data.encode('utf-8'), (UDP_IP, UDP_PORT))

        # Display the image in OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Handtracker", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()