import cv2
import mediapipe as mp
import socket
import json

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# UDP settings
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# pose landmarks to ignore
EXCLUDED_LANDMARKS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20, 21, 22, }
CUSTOM_POSE_CONNECTIONS = [
    (start, end) for start, end in mp_pose.POSE_CONNECTIONS
    if start not in EXCLUDED_LANDMARKS and end not in EXCLUDED_LANDMARKS
]

webcam = False
if webcam is False:
    video_source = "/run/media/will/Will_s SSD1/University_Projects/YR3/Twelvefold/previsfootage/cm30_reversed.mp4"
else:
    video_source = 0

cap = cv2.VideoCapture(video_source)
if not cap.isOpened():
    print("ERROR: could not open camera.")
    exit()

with mp_pose.Pose() as pose_detector:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to capture image")
            break

        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

        frame.flags.writeable = False

        pose_results = pose_detector.process(frame)

        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        landmarks_data = {
                "landmarks": []
            }

        if pose_results.pose_landmarks:
            for index, landmark in enumerate(pose_results.pose_landmarks.landmark):
                if index not in EXCLUDED_LANDMARKS:
                    landmarks_data["landmarks"].append({
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z,
                        "visibility": landmark.visibility
                    })

                    cx, cy = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (cx, cy), 3, (128, 0, 255), thickness=5)

            for connection in CUSTOM_POSE_CONNECTIONS:
                start_idx, end_idx = connection
                start_landmark = pose_results.pose_landmarks.landmark[start_idx]
                end_landmark = pose_results.pose_landmarks.landmark[end_idx]

                start_cx, start_cy = int(start_landmark.x * frame.shape[1]), int(start_landmark.y * frame.shape[0])
                end_cx, end_cy = int(end_landmark.x * frame.shape[1]), int(end_landmark.y * frame.shape[0])

                cv2.line(frame, (start_cx, start_cy), (end_cx, end_cy), (99, 255, 107), 2)

            json_data = json.dumps(landmarks_data)
            sock.sendto(json_data.encode('utf-8'), (UDP_IP, UDP_PORT))

        cv2.imshow("Pose Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
