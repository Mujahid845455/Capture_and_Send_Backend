"""
Word-Level Sign Language Recognition - Headless Backend Service
Receives video frames from frontend and returns sign predictions
No camera window - runs as background service
"""

import cv2
import mediapipe as mp
import numpy as np
import socketio
import base64
from io import BytesIO
from PIL import Image
import math

# Socket.IO server
sio = socketio.Server(cors_allowed_origins='*', async_mode='threading')
app = socketio.WSGIApp(sio)

# MediaPipe setup
mp_holistic = mp.solutions.holistic

print("🚀 Sign Language Recognition Service Started")
print("📡 Waiting for connections on port 7001...")

class SignLanguageRecognizer:
    @staticmethod
    def calculate_distance(point1, point2):
        """Calculate Euclidean distance between two points"""
        if point1 is None or point2 is None:
            return 0
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    @staticmethod
    def recognize_hello(hand_landmarks, pose_landmarks):
        """Recognize 'Hello' - waving hand gesture"""
        if hand_landmarks is None:
            return 0.0
        wrist = hand_landmarks.landmark[0]
        if wrist.y < 0.5:  # Hand in upper half
            return 0.85
        return 0.0
    
    @staticmethod
    def recognize_i_love_you(hand_landmarks):
        """Recognize 'I love you' - thumb, index, and pinky extended"""
        if hand_landmarks is None:
            return 0.0
        
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        ring_tip = hand_landmarks.landmark[16]
        pinky_tip = hand_landmarks.landmark[20]
        palm = hand_landmarks.landmark[0]
        
        thumb_extended = SignLanguageRecognizer.calculate_distance(thumb_tip, palm) > 0.15
        index_extended = SignLanguageRecognizer.calculate_distance(index_tip, palm) > 0.2
        pinky_extended = SignLanguageRecognizer.calculate_distance(pinky_tip, palm) > 0.15
        middle_folded = SignLanguageRecognizer.calculate_distance(middle_tip, palm) < 0.15
        ring_folded = SignLanguageRecognizer.calculate_distance(ring_tip, palm) < 0.15
        
        if thumb_extended and index_extended and pinky_extended and middle_folded and ring_folded:
            return 0.90
        return 0.0
    
    @staticmethod
    def recognize_yes(hand_landmarks, pose_landmarks):
        """Recognize 'Yes' - fist nodding motion"""
        if hand_landmarks is None:
            return 0.0
        
        fingers_folded = 0
        palm = hand_landmarks.landmark[0]
        for finger_tip_id in [8, 12, 16, 20]:
            tip = hand_landmarks.landmark[finger_tip_id]
            if SignLanguageRecognizer.calculate_distance(tip, palm) < 0.12:
                fingers_folded += 1
        
        if fingers_folded >= 3:
            return 0.80
        return 0.0
    
    @staticmethod
    def recognize_no(hand_landmarks):
        """Recognize 'No' - hand waving side to side"""
        if hand_landmarks is None:
            return 0.0
        
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        palm = hand_landmarks.landmark[0]
        
        if SignLanguageRecognizer.calculate_distance(index_tip, palm) > 0.15 and \
           SignLanguageRecognizer.calculate_distance(middle_tip, palm) > 0.15:
            return 0.75
        return 0.0
    
    @staticmethod
    def recognize_please(hand_landmarks, pose_landmarks):
        """Recognize 'Please' - circular motion on chest"""
        if hand_landmarks is None:
            return 0.0
        palm = hand_landmarks.landmark[0]
        if 0.3 < palm.x < 0.7 and 0.4 < palm.y < 0.7:
            return 0.70
        return 0.0
    
    @staticmethod
    def recognize_thank_you(hand_landmarks):
        """Recognize 'Thank you' - hand from chin outward"""
        if hand_landmarks is None:
            return 0.0
        palm = hand_landmarks.landmark[0]
        if palm.y < 0.4:
            return 0.75
        return 0.0
    
    @staticmethod
    def recognize_ok(hand_landmarks):
        """Recognize 'OK' - thumb and index forming circle"""
        if hand_landmarks is None:
            return 0.0
        
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        distance = SignLanguageRecognizer.calculate_distance(thumb_tip, index_tip)
        if distance < 0.05:
            return 0.85
        return 0.0
    
    @staticmethod
    def recognize_help(hand_landmarks, pose_landmarks):
        """Recognize 'Help' - one hand raised"""
        if hand_landmarks is None:
            return 0.0
        wrist = hand_landmarks.landmark[0]
        if wrist.y < 0.3:
            return 0.70
        return 0.0
    
    @staticmethod
    def recognize_good(hand_landmarks):
        """Recognize 'Good' - thumbs up"""
        if hand_landmarks is None:
            return 0.0
        
        thumb_tip = hand_landmarks.landmark[4]
        palm = hand_landmarks.landmark[0]
        
        thumb_up = thumb_tip.y < palm.y
        fingers_folded = all(
            SignLanguageRecognizer.calculate_distance(hand_landmarks.landmark[tip_id], palm) < 0.12
            for tip_id in [8, 12, 16, 20]
        )
        
        if thumb_up and fingers_folded:
            return 0.85
        return 0.0
    
    @staticmethod
    def recognize_bad(hand_landmarks):
        """Recognize 'Bad' - thumbs down"""
        if hand_landmarks is None:
            return 0.0
        
        thumb_tip = hand_landmarks.landmark[4]
        palm = hand_landmarks.landmark[0]
        
        if thumb_tip.y > palm.y + 0.1:
            return 0.80
        return 0.0
    
    @staticmethod
    def predict_sign(results):
        """Predict sign language word from holistic results"""
        if results.left_hand_landmarks is None and results.right_hand_landmarks is None:
            return None, 0.0
        
        hand_landmarks = results.right_hand_landmarks or results.left_hand_landmarks
        pose_landmarks = results.pose_landmarks
        
        predictions = {
            "Hello": SignLanguageRecognizer.recognize_hello(hand_landmarks, pose_landmarks),
            "I love you": SignLanguageRecognizer.recognize_i_love_you(hand_landmarks),
            "Yes": SignLanguageRecognizer.recognize_yes(hand_landmarks, pose_landmarks),
            "No": SignLanguageRecognizer.recognize_no(hand_landmarks),
            "Please": SignLanguageRecognizer.recognize_please(hand_landmarks, pose_landmarks),
            "Thank you": SignLanguageRecognizer.recognize_thank_you(hand_landmarks),
            "OK": SignLanguageRecognizer.recognize_ok(hand_landmarks),
            "Help": SignLanguageRecognizer.recognize_help(hand_landmarks, pose_landmarks),
            "Good": SignLanguageRecognizer.recognize_good(hand_landmarks),
            "Bad": SignLanguageRecognizer.recognize_bad(hand_landmarks),
        }
        
        best_sign = max(predictions, key=predictions.get)
        best_confidence = predictions[best_sign]
        
        if best_confidence > 0.6:
            return best_sign, best_confidence
        
        return None, 0.0

# Socket.IO event handlers
@sio.event
def connect(sid, environ):
    print(f"✅ Client connected: {sid}")

@sio.event
def disconnect(sid):
    print(f"🔴 Client disconnected: {sid}")

@sio.event
def process_frame(sid, data):
    """Process video frame from frontend"""
    try:
        # Decode base64 image
        image_data = base64.b64decode(data['frame'].split(',')[1])
        image = Image.open(BytesIO(image_data))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Create new Holistic instance for each frame to avoid timestamp issues
        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=True  # Process each frame independently
        ) as holistic:
            # Process with MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = holistic.process(image_rgb)
            
            # Get prediction
            sign, confidence = SignLanguageRecognizer.predict_sign(results)
            
            if sign:
                # Emit prediction back to client
                sio.emit('sign_prediction', {
                    'word': sign,
                    'confidence': float(confidence),
                    'timestamp': data.get('timestamp', 0)
                }, room=sid)
            
    except Exception as e:
        print(f"⚠️ Frame processing error: {e}")

if __name__ == '__main__':
    import eventlet
    import eventlet.wsgi
    
    print("🎯 Starting server on port 7001...")
    # Run Socket.IO server on port 7001 (separate from main backend)
    eventlet.wsgi.server(eventlet.listen(('', 7001)), app)
