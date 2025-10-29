import cv2
import mediapipe as mp
import numpy as np
import serial
import time
import tensorflow as tf
from tensorflow.keras.models import load_model

# === Serial Setup ===
try:
    ser = serial.Serial('COM6', 9600, timeout=1)
    time.sleep(2)
    print("✅ Arduino connected.")
except:
    ser = None
    print("⚠️ Arduino not connected.")

# === Load Trained Model ===
model_path = r'C:\Users\admin\Desktop\ll\hand_sign_iot_project\model\hand_sign_model.h5'
model = load_model(model_path)

# === MediaPipe Setup ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# === Finger Indices ===
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [2, 6, 10, 14, 18]

# === Gesture Rules ===
gesture_rules = {
    "hello": [1, 1, 1, 1, 1],
    "thanks": [0, 0, 0, 0, 0],
    "yes": [0, 0, 0, 0, 0],
    "no": [1, 1, 0, 0, 0],
    "Sorry": [1, 1, 0, 0, 1],
    "stop": [0, 1, 1, 1, 1],
    "wait": [0, 1, 1, 1, 0],
    "come": [1, 1, 0, 0, 0],
    "go": [1, 1, 0, 0, 0],
    "bye bye": [0, 1, 1, 0, 0],
    "eat": [0, 1, 0, 0, 0],
    "drink": [1, 0, 0, 0, 0],
    "okay": [2, 2, 1, 1, 1],
    "lets go ": [0, 0, 0, 0, 1]
}

# === State Tracking ===
last_sent = ""
last_time = time.time()

# === Webcam Reopen Logic ===
def open_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return None
    return cap

# === Extract Features for ML ===
def extract_features(lm):
    coords = np.array([[p.x, p.y, p.z] for p in lm]).flatten()
    return np.array([coords])

# === Get Rule-Based States ===
def get_finger_states(landmarks):
    states = []
    for tip, pip in zip(FINGER_TIPS, FINGER_PIPS):
        if tip == 4:  # Thumb (x comparison)
            state = int(landmarks[tip].x < landmarks[pip].x)
        else:
            state = int(landmarks[tip].y < landmarks[pip].y)
        states.append(state)
    return states

# === Match Rule-Based Gesture ===
def match_gesture(states):
    for gesture, rule in gesture_rules.items():
        if all((r == 2 or s == r) for s, r in zip(states, rule)):
            return gesture
    return None

# === Main Loop ===
print("✅ Rule-based + ML hybrid gesture recognition started...")

cap = open_camera()

try:
    while True:
        if cap is None or not cap.isOpened():
            cap = open_camera()
            continue

        ret, frame = cap.read()
        if not ret:
            continue

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            label = "No hand"

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    lm = hand_landmarks.landmark
                    states = get_finger_states(lm)

                    # Try rule-based first
                    gesture = match_gesture(states)

                    # If no match, fallback to ML
                    if gesture is None:
                        features = extract_features(lm)
                        preds = model.predict(features, verbose=0)
                        pred_idx = np.argmax(preds)
                        confidence = np.max(preds)
                        if confidence > 0.8:
                            gesture = str(pred_idx)

                    if gesture:
                        label = gesture
                        if label != last_sent and (time.time() - last_time) > 1:
                            if ser:
                                ser.write(label.encode())
                            print(f"[SENT] {label}")
                            last_sent = label
                            last_time = time.time()

                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Draw overlay
            cv2.putText(frame, f"Gesture: {label}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow("Hybrid Gesture Recognition", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        except Exception as e:
            print("⚠️ Frame processing error:", e)
            continue

except KeyboardInterrupt:
    print("🛑 Interrupted by user.")

cap.release()
cv2.destroyAllWindows()
if ser:
    ser.close()
