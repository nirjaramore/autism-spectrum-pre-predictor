import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

image = cv2.imread("test.jpg")  # put any face image here

with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        print("Face + landmarks detected ✅")
    else:
        print("No face detected ❌")