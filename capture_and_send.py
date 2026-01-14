# capture_and_send.py
import cv2
import mediapipe as mp
import socketio
from datetime import datetime
import time
import os
import numpy as np
import json
from dotenv import load_dotenv
import mediapipe as mp
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# Try to import TensorFlow with error handling
try:
    import tensorflow
    TENSORFLOW_AVAILABLE = True
    print(f"✅ TensorFlow detected")
except ImportError as e:
    print(f"⚠️ TensorFlow not available: {e}")
    TENSORFLOW_AVAILABLE = False

# ================= ENV =================
load_dotenv()
BACKEND_URL = os.getenv("BACKEND_SOCKET_URL", "http://localhost:7000")
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))
SEND_FPS = int(os.getenv("SEND_FPS", 30))

#================= LOAD SIGN MODEL & LABELS =================
MODEL_PATH = "models/asl_mediapipe_mlp_model.h5"
LABELS_PATH = "models/class_labels.json"

model = None
LABELS = []
label_encoder = None

try:
    # Load labels from JSON file
    with open(LABELS_PATH, "r") as f:
        LABELS = json.load(f)
    print(f"✅ Loaded {len(LABELS)} labels: {LABELS}")
    
    # Load model if TensorFlow is available
    if TENSORFLOW_AVAILABLE:
        from sklearn.preprocessing import LabelEncoder
        import tensorflow as tf
        
        print("⏳ Loading MediaPipe MLP model...")
        
        try:
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            print(f"✅ Model loaded with TensorFlow {tf.__version__}")
            print(f"   Input shape: {model.input_shape}")
            print(f"   Output shape: {model.output_shape}")
            
            # Check if model output matches number of labels
            num_classes = model.output_shape[-1]
            if len(LABELS) != num_classes:
                print(f"⚠️ Warning: Model has {num_classes} classes but {len(LABELS)} labels provided")
                print(f"   Using first {num_classes} labels from the list")
                LABELS = LABELS[:num_classes]
            
            # Create label encoder for class mapping
            label_encoder = LabelEncoder()
            label_encoder.fit(LABELS)
            
            print(f"✅ Model ready for predictions with {len(LABELS)} classes")
            
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")
            model = None
            print("⚠️ Using placeholder predictions")
        
    else:
        print("⚠️ TensorFlow not available - using placeholder predictions")
        
except FileNotFoundError as e:
    print(f"⚠️ File not found: {e}")
    print("⚠️ Using default labels and placeholder predictions")
    LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
              "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", 
              "del", "nothing", "space"]
except Exception as e:
    print(f"⚠️ Error loading model: {e}")
    print("⚠️ Using placeholder predictions")

# ================= SOCKET =================
sio = socketio.Client()

@sio.event
def connect():
    print("✅ Connected to backend")

@sio.event
def disconnect():
    print("❌ Disconnected")

@sio.event
def connect_error(err):
    print("❌ Connection error:", err)

# Tracking state
is_tracking_enabled = False

@sio.on("start_tracking")
def on_start():
    global is_tracking_enabled
    print("🚀 Tracking command received: START")
    is_tracking_enabled = True

@sio.on("stop_tracking")
def on_stop():
    global is_tracking_enabled
    print("🛑 Tracking command received: STOP")
    is_tracking_enabled = False

try:
    sio.connect(BACKEND_URL, auth={'isCapture': True})
except Exception as e:
    print(f"⚠️ Could not connect to backend: {e}")
    print("⚠️ Running in local mode only")

# ================= MEDIAPIPE =================
mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,  # Use complex model for better accuracy
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def lm_dict(lm):
    return {"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)}

# ================= LANDMARK-BASED SIGN PREDICTION =================
from collections import deque

def extract_landmark_features(hand_landmarks, handedness):
    """Extract and normalize 21 hand landmarks from MediaPipe (63 features total)"""
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    
    # Flip x-coordinates for right hand to match left-hand training data
    if handedness.classification[0].label == "Right":
        landmarks[:, 0] = 1 - landmarks[:, 0]
    
    return landmarks.flatten().reshape(1, -1)

def predict_sign_from_landmarks(hand_landmarks, handedness):
    """Predict sign from MediaPipe hand landmarks"""
    if model is None or not TENSORFLOW_AVAILABLE or label_encoder is None:
        # Return placeholder prediction
        import random
        if LABELS:
            letter = random.choice(LABELS)
            confidence = random.uniform(0.7, 0.95)
        else:
            letter = "?"
            confidence = 0.0
        return letter, confidence
    
    try:
        # Extract 63 features from hand landmarks
        landmark_features = extract_landmark_features(hand_landmarks, handedness)
        
        # Make prediction
        prediction = model.predict(landmark_features, verbose=0)
        
        # Get highest probability
        idx = np.argmax(prediction)
        confidence = float(prediction[0][idx])
        
        # Return corresponding label
        predicted_label = label_encoder.inverse_transform([idx])[0]
        return predicted_label, confidence
            
    except Exception as e:
        print(f"⚠️ Prediction error: {e}")
        return "?", 0.0

# ================= SENTENCE FORMATION LOGIC =================
predicted_sentence = ""
last_predicted_label = None
last_prediction_time = 0
cooldown_time = 5  # 5 seconds cooldown for repeated letters

# Stabilization buffer: stores last 5 predictions
stabilization_window = deque(maxlen=5)
stabilization_threshold = 4  # Must match for 4 out of 5 frames

cap = None
FRAME_DELAY = 1 / SEND_FPS
print("🎥 Capture system ready (Waiting for 'Start Tracking' from frontend...)")

# ================= MAIN LOOP =================
function_start_time = time.time()
frame_count = 0
last_sent_time = time.time()

while True:
    # 🛑 If tracking is disabled, release camera and wait
    if not is_tracking_enabled:
        if cap is not None:
            print("💤 Tracking stopped. Releasing camera...")
            cap.release()
            cv2.destroyAllWindows()
            cap = None
        time.sleep(0.5)
        continue

    # 🚀 If tracking is enabled, ensure camera is open
    if cap is None:
        print(f"🎬 Tracking started. Opening camera {CAMERA_INDEX}...")
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            print(f"❌ Could not open camera {CAMERA_INDEX}")
            is_tracking_enabled = False
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    start = time.time()
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process for landmarks
    holistic_result = holistic.process(rgb)
    
    # Process for hand detection (for better ROI)
    hands_result = hands.process(rgb)
    
    sign_letter = None
    sign_confidence = None
    hand_bbox = None

    data = {
        "timestamp": datetime.utcnow().isoformat(),
        "pose": {},
        "left_hand": {},
        "right_hand": {},
        "sign_language": {
            "letter": None,
            "confidence": None,
            "sentence": ""
        }
    }

    # ---- POSE ----
    if holistic_result.pose_landmarks:
    # Send ALL pose landmarks (33 points)
        pose_landmarks = holistic_result.pose_landmarks.landmark
        data["pose"] = {
        str(i): lm_dict(lm) for i, lm in enumerate(pose_landmarks)
        }
    
    # Also keep the specific ones for backward compatibility
        data["pose_specific"] = {
            "LEFT_SHOULDER": lm_dict(pose_landmarks[11]),
            "LEFT_ELBOW": lm_dict(pose_landmarks[13]),
            "LEFT_WRIST": lm_dict(pose_landmarks[15]),
            "RIGHT_SHOULDER": lm_dict(pose_landmarks[12]),
            "RIGHT_ELBOW": lm_dict(pose_landmarks[14]),
            "RIGHT_WRIST": lm_dict(pose_landmarks[16]),
            "LEFT_HIP": lm_dict(pose_landmarks[23]),
            "RIGHT_HIP": lm_dict(pose_landmarks[24]),
            "LEFT_KNEE": lm_dict(pose_landmarks[25]),
            "RIGHT_KNEE": lm_dict(pose_landmarks[26]),
            "LEFT_ANKLE": lm_dict(pose_landmarks[27]),
            "RIGHT_ANKLE": lm_dict(pose_landmarks[28]),
            "LEFT_FOOT_INDEX": lm_dict(pose_landmarks[31]),
            "RIGHT_FOOT_INDEX": lm_dict(pose_landmarks[32]),
        }
        mp_draw.draw_landmarks(frame, holistic_result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    # ---- LEFT HAND ----
    if holistic_result.left_hand_landmarks:
        for i, lm in enumerate(holistic_result.left_hand_landmarks.landmark):
            data["left_hand"][str(i)] = lm_dict(lm)
        mp_draw.draw_landmarks(frame, holistic_result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # ---- RIGHT HAND ----
    if holistic_result.right_hand_landmarks:
        for i, lm in enumerate(holistic_result.right_hand_landmarks.landmark):
            data["right_hand"][str(i)] = lm_dict(lm)
        mp_draw.draw_landmarks(frame, holistic_result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # 🔥 SIGN PREDICTION FROM HAND LANDMARKS
    # Only predict if exactly ONE hand is detected (avoid confusion with two hands)
    num_hands = 0
    if hands_result.multi_hand_landmarks:
        num_hands = len(hands_result.multi_hand_landmarks)
    
    if num_hands == 1:
        # Get the single hand
        hand_landmarks = hands_result.multi_hand_landmarks[0]
        handedness = hands_result.multi_handedness[0]
        
        # Predict sign from landmarks
        letter, confidence = predict_sign_from_landmarks(hand_landmarks, handedness)
        
        # Add to stabilization buffer
        stabilization_window.append(letter)
        
        # Check if prediction is stable (appears in 4 out of 5 frames)
        if len(stabilization_window) == stabilization_window.maxlen:
            letter_counts = {}
            for l in stabilization_window:
                letter_counts[l] = letter_counts.get(l, 0) + 1
            
            # Find most common letter
            most_common_letter = max(letter_counts, key=letter_counts.get)
            most_common_count = letter_counts[most_common_letter]
            
            # Only accept if it appears >= threshold times
            if most_common_count >= stabilization_threshold and confidence > 0.5:
                sign_letter = most_common_letter
                sign_confidence = confidence
                
                # Update data for real-time display
                data["sign_language"]["letter"] = sign_letter
                data["sign_language"]["confidence"] = sign_confidence
                
                # 📝 SENTENCE FORMATION LOGIC
                current_time = time.time()
                
                # Check cooldown for repeated letters
                if sign_letter == last_predicted_label:
                    if current_time - last_prediction_time < cooldown_time:
                        sign_letter = None  # Ignore repeated prediction
                else:
                    last_predicted_label = sign_letter
                    last_prediction_time = current_time
                
                # Build sentence
                if sign_letter:
                    if sign_letter == "space":
                        predicted_sentence += " "
                    elif sign_letter == "del":
                        predicted_sentence = predicted_sentence[:-1]  # Remove last character
                    elif sign_letter != "nothing":
                        predicted_sentence += sign_letter
                    
                    # Add sentence to data payload
                    data["sign_language"]["sentence"] = predicted_sentence
    
    elif num_hands > 1:
        # Display warning for two hands
        cv2.putText(
            frame,
            "⚠️ Use ONE hand only",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3
        )
    else:
        # 🧪 TESTING: Generate placeholder predictions even without hand detection
        # This helps test the data flow to frontend
        import random
        if frame_count % 30 == 0:  # Every 30 frames (once per second at 30 FPS)
            test_letter = random.choice([l for l in LABELS if l not in ["nothing", "del", "space"]])
            test_confidence = random.uniform(0.75, 0.95)
            
            # Update predicted sentence
            predicted_sentence += test_letter
            
            # Update data
            data["sign_language"]["letter"] = test_letter
            data["sign_language"]["confidence"] = test_confidence
            data["sign_language"]["sentence"] = predicted_sentence
            
            print(f"🧪 TEST: Generated sign: {test_letter} | Sentence: '{predicted_sentence}'")
    
    # Display prediction on frame
    if sign_letter and sign_letter not in ["nothing", "space", "del"]:
        display_text = f"Sign: {sign_letter}"
        if sign_confidence is not None:
            display_text += f" ({sign_confidence:.2f})"
        
        cv2.putText(
            frame,
            display_text,
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 255),
            3
        )
    
    # Display sentence on frame
    if predicted_sentence:
        cv2.putText(
            frame,
            f"Sentence: {predicted_sentence}",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # Display model status
    if model is not None:
        model_status = "Model: Ready"
        color = (0, 255, 0)
    else:
        model_status = "Model: Placeholder"
        color = (0, 165, 255)
    
    cv2.putText(
        frame,
        model_status,
        (30, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

    # Display FPS
    fps = 1.0 / (time.time() - start) if (time.time() - start) > 0 else 0
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (30, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # ---- SEND TO BACKEND (if connected) ----
    try:
        # Only send at the specified FPS
        current_time = time.time()
        if sio.connected and current_time - last_sent_time >= FRAME_DELAY:
            # Debug: Print what we're sending
            if sign_letter and sign_letter not in ["nothing", None]:
                print(f"📤 Sending sign: {sign_letter} (conf: {sign_confidence:.2f}) | Sentence: '{predicted_sentence}'")
            
            sio.emit("landmarks", data)
            last_sent_time = current_time
    except Exception as e:
        # Don't print error on every frame
        if frame_count % 30 == 0:
            print(f"⚠️ Failed to send data: {e}")

    cv2.imshow("ASL Sign Recognition", frame)

    elapsed = time.time() - start
    if elapsed < FRAME_DELAY:
        time.sleep(FRAME_DELAY - elapsed)

    if cv2.waitKey(1) & 0xFF == 27:
        break
    
    frame_count += 1

# ================= CLEANUP =================
cap.release()
cv2.destroyAllWindows()
try:
    hands.close()
except:
    pass

try:
    sio.disconnect()
except:
    pass

print("✅ Capture stopped")
