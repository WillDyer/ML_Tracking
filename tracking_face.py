import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

FACE_CONNECTIONS = frozenset([
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154), 
    (154, 155), (155, 133), (33, 246), (246, 161), (161, 160), (160, 159),
    (159, 158), (158, 157), (157, 173), (173, 133), (246, 30), (30, 29), 
    (29, 27), (27, 28), (28, 56), (56, 190), (190, 243), (243, 112), 
    (112, 26), (26, 22), (22, 23), (23, 24), (24, 110), (110, 25), (25, 130),
])

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: could not open camera.")
    exit()

with mp_face.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Failed to capture image")
            break

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = face_mesh.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image, 
                    face_landmarks, 
                    FACE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec
                )

        cv2.imshow("Face Masking", image)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()