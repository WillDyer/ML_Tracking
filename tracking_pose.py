import cv2
import mediapipe as mp
import socket
import json

# Initialize Mediapipe drawing and pose detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Drawing specifications
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# UDP settings
UDP_IP = "127.0.0.1"  # Corrected to a valid IP address
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: could not open camera.")
    exit()

# Initialize Pose detection
with mp_pose.Pose() as pose_detector:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Failed to capture image")
            break

        # Convert the BGR image to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # Mark the image as not writable to improve performance
        image.flags.writeable = False

        # Process the image to detect pose landmarks
        results = pose_detector.process(image)

        # Set the image back to writable and convert to BGR for display
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Check if pose landmarks are detected
        if results.pose_landmarks:
            # Draw pose landmarks on the image
            mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

            # Prepare landmark data to send over UDP
            landmarks_data = {
                "landmarks": []
            }

            # Collect all pose landmarks
            for landmark in results.pose_landmarks.landmark:
                landmarks_data["landmarks"].append({
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                    "visibility": landmark.visibility  # Pose landmarks include visibility info
                })

            # Serialize the data as JSON and send via UDP
            json_data = json.dumps(landmarks_data)
            sock.sendto(json_data.encode('utf-8'), (UDP_IP, UDP_PORT))

        # Display the image with pose landmarks
        cv2.imshow("Pose Detection", image)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()