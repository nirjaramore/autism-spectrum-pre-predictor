import cv2
from mtcnn import MTCNN

# ---------------- INITIALIZE DETECTOR ----------------

detector = MTCNN()

# ---------------- FACE DETECTION FUNCTION ----------------

def detect_face(image):

    # Convert BGR → RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces
    faces = detector.detect_faces(rgb_image)

    # No face found
    if len(faces) == 0:
        return None

    # First detected face
    x, y, width, height = faces[0]['box']

    # Prevent negative values
    x = max(0, x)
    y = max(0, y)

    # Crop face
    face = image[y:y + height, x:x + width]

    # Validate face crop
    if face.size == 0:
        return None

    return face