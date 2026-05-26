import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Labels
AGE_GROUPS = ["Child (0-12)", "Teen (13-19)", "Young Adult (20-35)", 
              "Adult (36-55)", "Senior (56+)"]
GENDER_LABELS = ["Male", "Female"]

# Colors (BGR format for OpenCV)
GENDER_COLORS = {
    "Male":   (219, 152, 52),   # blue
    "Female": (147, 20, 255)    # pink
}
AGE_COLOR = (255, 255, 255)     # white

# Load model
print("Loading model...")
model = load_model("models/pehchan_model.keras")
print("Model loaded ✓")

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def preprocess_face(face_img):
    face = cv2.resize(face_img, (64, 64))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)
    return face

def draw_label(frame, text, x, y, color, bg_color=(0,0,0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.6, 2
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(frame, (x, y-h-8), (x+w+8, y+4), bg_color, -1)
    cv2.putText(frame, text, (x+4, y), font, scale, color, thickness)

# Start webcam
print("Starting webcam... Press Q to quit")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        # Extract and predict
        face_img = frame[y:y+h, x:x+w]
        face_input = preprocess_face(face_img)
        gender_prob, age_prob = model.predict(face_input, verbose=0)

        gender_idx = int(gender_prob[0][0] > 0.5)
        age_idx    = int(np.argmax(age_prob[0]))

        gender = GENDER_LABELS[gender_idx]
        age    = AGE_GROUPS[age_idx]
        g_conf = float(gender_prob[0][0]) if gender_idx == 1 else 1 - float(gender_prob[0][0])
        a_conf = float(np.max(age_prob[0]))

        box_color = GENDER_COLORS[gender]

        # Draw face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)

        # Draw labels
        draw_label(frame, f"{gender} ({g_conf*100:.0f}%)", x, y-35, box_color)
        draw_label(frame, f"{age} ({a_conf*100:.0f}%)",    x, y-10, AGE_COLOR)

    # Header bar
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 35), (30, 30, 30), -1)
    cv2.putText(frame, "Pehchan AI  |  Press Q to quit",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

    cv2.imshow("Pehchan AI - Real Time Age & Gender Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam closed.")