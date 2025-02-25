import cv2
import mediapipe as mp
import socket
import json

# init mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# UDP settings
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# pose landmarks to ignore
EXCLUDED_LANDMARKS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20, 21, 22,}
CUSTOM_POSE_CONNECTIONS = [
    (start, end) for start, end in mp_pose.POSE_CONNECTIONS
    if start not in EXCLUDED_LANDMARKS and end not in EXCLUDED_LANDMARKS
]

show_pose = 1
show_hands = 1
show_video_sources = True
webcam = False

video_source_1 = "/home/will/Documents/walk_cycle_1.mp4"
video_source_2 = "/home/will/Documents/walk_cycle_2.mp4"

if webcam is False:
    video_source_1 = video_source_1
    video_source_2 = video_source_2
else:
    video_source_1 = 0

capture_1 = cv2.VideoCapture(video_source_1)
capture_2 = cv2.VideoCapture(video_source_2)

if not capture_2.isOpened():
    print("ERROR: could not open camera.")
    exit()

def get_landmark_data(frame, pose_detector, hands_detector):
    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    frame.flags.writeable = False
    pose_results = pose_detector.process(frame)
    hand_results = hands_detector.process(frame)
    frame.flags.writeable = True

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    landmarks = {
        "pose_landmarks": [],
        "hand_landmarks": []
    }

    if show_pose == 1:
        if pose_results.pose_landmarks:
            for index, landmark in enumerate(pose_results.pose_landmarks.landmark):
                if index not in EXCLUDED_LANDMARKS:
                    landmarks["pose_landmarks"].append({
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

    if show_hands == 1:
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                hand_landmark_data = []
                for landmark in hand_landmarks.landmark:
                    hand_landmark_data.append({
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z
                    })
                landmarks["hand_landmarks"].append(hand_landmark_data)

                mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks, 
                            mp_hands.HAND_CONNECTIONS
                        )
        
    return landmarks

def average_landmarks(landmarks_1, landmarks_2):
    averaged_landmarks = {
        "pose_landmarks": [],
        "hand_landmarks": []
        }

    if landmarks_1["pose_landmarks"] and landmarks_2["pose_landmarks"]:
        for pose_1, pose_2 in zip(landmarks_1["pose_landmarks"], landmarks_2["pose_landmarks"]):
            average = {
                "x": (pose_1["x"] + pose_2["x"]) / 2,
                "y": (pose_1["y"] + pose_2["y"]) / 2,
                "z": (pose_1["z"] + pose_2["z"]) / 2,
                "visibility": (pose_1["visibility"] + pose_2["visibility"]) / 2
            }
            averaged_landmarks["pose_landmarks"].append(average)

    if landmarks_1["hand_landmarks"] and landmarks_2["hand_landmarks"]:
        for hand_1, hand_2 in zip(landmarks_1["hand_landmarks"], landmarks_2["hand_landmarks"]):
            average_hand = []
            for lm1, lm2 in zip(hand_1, hand_2):
                average = {
                    "x": (lm1["x"] + lm2["x"]) / 2,
                    "y": (lm1["y"] + lm2["y"]) / 2,
                    "z": (lm1["z"] + lm2["z"]) / 2
                }
                average_hand.append(average_hand)
            averaged_landmarks["hand_landmarks"].append(average_hand)

    return averaged_landmarks


with mp_pose.Pose() as pose_detector, mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands_detector:
    while capture_1.isOpened() and capture_2.isOpened():
        success_1, frame_1 = capture_1.read()
        success_2, frame_2 = capture_2.read()

        if not success_1 or not success_2:
            print("Failed to capture image")
            break

        landmarks_1 = get_landmark_data(frame_1, pose_detector, hands_detector)
        landmarks_2 = get_landmark_data(frame_2, pose_detector, hands_detector)

        averaged_landmarks = average_landmarks(landmarks_1, landmarks_2)

        all_landmarks_data = {
            "pose_landmarks": averaged_landmarks["pose_landmarks"],
            "hand_landmarks": averaged_landmarks["hand_landmarks"]
        }

        print(all_landmarks_data)

        json_data = json.dumps(all_landmarks_data)
        sock.sendto(json_data.encode('utf-8'), (UDP_IP, UDP_PORT))

        if show_video_sources is True:
            cv2.imshow("ML Tracking showing Source_1", frame_1)
            cv2.imshow("ML Tracking showing Source_2", frame_2)


        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

capture_1.release()
capture_2.release()
cv2.destroyAllWindows()