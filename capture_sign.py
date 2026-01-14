import cv2
import mediapipe as mp
from gtts import gTTS
import pygame
import threading
import time
import os
import warnings
import math
import queue

# Suppress warnings
warnings.filterwarnings("ignore")

# Initialize pygame mixer once at the beginning
try:
    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
    print("Audio initialized successfully")
except Exception as e:
    print(f"Audio initialization failed: {e}")

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Audio queue to manage playback
audio_queue = queue.Queue()
audio_playing = False

# --- Function to play audio in a separate thread ---
def audio_worker():
    global audio_playing
    while True:
        try:
            text = audio_queue.get()
            if text is None:  # Exit signal
                break
                
            audio_playing = True
            filename = f"voice_{text.replace(' ', '_').lower()}.mp3"
            
            try:
                # Generate TTS
                tts = gTTS(text=text, lang='en', slow=False)
                tts.save(filename)
                
                # Load and play
                sound = pygame.mixer.Sound(filename)
                sound.play()
                
                # Wait for sound to finish
                time.sleep(sound.get_length() + 0.1)
                
            except Exception as e:
                print(f"Audio error for '{text}': {e}")
            finally:
                # Cleanup
                if os.path.exists(filename):
                    os.remove(filename)
                    
            audio_playing = False
            audio_queue.task_done()
            
        except Exception as e:
            print(f"Audio worker error: {e}")
            audio_playing = False

# Start audio worker thread
audio_thread = threading.Thread(target=audio_worker, daemon=True)
audio_thread.start()

# --- Function to queue audio for playback ---
def play_audio(text):
    if not audio_playing and audio_queue.qsize() < 3:  # Limit queue size
        audio_queue.put(text)

# --- Function to calculate distance between two landmarks ---
def calculate_distance(landmark1, landmark2):
    x1, y1 = landmark1.x, landmark1.y
    x2, y2 = landmark2.x, landmark2.y
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# --- Gesture detection function ---
def detect_gesture(landmarks):
    try:
        # Get all landmarks
        thumb_tip = landmarks.landmark[4]
        thumb_mcp = landmarks.landmark[2]
        
        index_tip = landmarks.landmark[8]
        index_mcp = landmarks.landmark[5]
        
        middle_tip = landmarks.landmark[12]
        middle_mcp = landmarks.landmark[9]
        
        ring_tip = landmarks.landmark[16]
        ring_mcp = landmarks.landmark[13]
        
        pinky_tip = landmarks.landmark[20]
        pinky_mcp = landmarks.landmark[17]
        
        wrist = landmarks.landmark[0]

        # Gesture 1: "Hello" - Open hand
        if (thumb_tip.y < thumb_mcp.y and 
            index_tip.y < index_mcp.y and 
            middle_tip.y < middle_mcp.y and 
            ring_tip.y < ring_mcp.y and 
            pinky_tip.y < pinky_mcp.y):
            return "Hello"

        # Gesture 2: "Point" - Index finger pointing
        if (index_tip.y < index_mcp.y and 
            middle_tip.y > middle_mcp.y and 
            ring_tip.y > ring_mcp.y and 
            pinky_tip.y > pinky_mcp.y):
            return "Point"

        # Gesture 3: "Fist" - Closed fist
        if (index_tip.y > index_mcp.y and 
            middle_tip.y > middle_mcp.y and 
            ring_tip.y > ring_mcp.y and 
            pinky_tip.y > pinky_mcp.y):
            return "Fist"

        # Gesture 4: "Peace" - Victory sign
        if (index_tip.y < index_mcp.y and 
            middle_tip.y < middle_mcp.y and 
            ring_tip.y > ring_mcp.y and 
            pinky_tip.y > pinky_mcp.y):
            return "Peace"

        # Gesture 5: "OK" - Thumb and index touching
        thumb_index_dist = calculate_distance(thumb_tip, index_tip)
        if (thumb_index_dist < 0.05 and
            middle_tip.y > middle_mcp.y and
            ring_tip.y > ring_mcp.y and
            pinky_tip.y > pinky_mcp.y):
            return "OK"

        # Gesture 6: "Thumbs Up"
        if (thumb_tip.y < thumb_mcp.y and 
            index_tip.y > index_mcp.y and 
            middle_tip.y > middle_mcp.y and 
            ring_tip.y > ring_mcp.y and 
            pinky_tip.y > pinky_mcp.y):
            return "Thumbs Up"

        # Gesture 7: "Thumbs Down"
        if (thumb_tip.y > thumb_mcp.y and
            thumb_tip.y > wrist.y and
            index_tip.y > index_mcp.y and 
            middle_tip.y > middle_mcp.y and 
            ring_tip.y > ring_mcp.y and 
            pinky_tip.y > pinky_mcp.y):
            return "Thumbs Down"

        # Gesture 8: "Three" - Three fingers up
        if (index_tip.y < index_mcp.y and 
            middle_tip.y < middle_mcp.y and 
            ring_tip.y < ring_mcp.y and 
            pinky_tip.y > pinky_mcp.y):
            return "Three"

        # Gesture 9: "Four" - Four fingers up
        if (index_tip.y < index_mcp.y and 
            middle_tip.y < middle_mcp.y and 
            ring_tip.y < ring_mcp.y and 
            pinky_tip.y < pinky_mcp.y):
            return "Four"

        # Gesture 10: "Call Me" - Pinky and thumb extended
        if (thumb_tip.y < thumb_mcp.y and 
            pinky_tip.y < pinky_mcp.y and 
            index_tip.y > index_mcp.y and 
            middle_tip.y > middle_mcp.y and 
            ring_tip.y > ring_mcp.y):
            return "Call Me"

    except Exception as e:
        print(f"Gesture detection error: {e}")
        return None
    
    return None

# --- Main program ---
def main():
    # Try different camera indices if 0 doesn't work
    for camera_index in [0, 1, 2]:
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            print(f"Camera {camera_index} opened successfully")
            break
        else:
            cap.release()
    else:
        print("No camera found!")
        return

    last_gesture = None
    last_time = 0
    frame_count = 0
    fps = 0
    start_time = time.time()

    # Color scheme for gestures
    gesture_colors = {
        "Hello": (255, 0, 0),           # Blue
        "Point": (0, 255, 0),           # Green
        "Fist": (0, 165, 255),          # Orange
        "Peace": (255, 0, 255),         # Purple
        "OK": (255, 255, 0),           # Cyan
        "Thumbs Up": (0, 255, 255),     # Yellow
        "Thumbs Down": (139, 0, 0),     # Dark Red
        "Three": (128, 0, 128),         # Purple
        "Four": (0, 128, 128),          # Teal
        "Call Me": (30, 144, 255),      # Dodger Blue
    }

    default_color = (255, 255, 255)

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:
            
            print("Hand tracking started. Press 'ESC' to exit.")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                frame_count += 1
                frame = cv2.flip(frame, 1)
                
                # Calculate FPS
                if frame_count % 30 == 0:
                    fps = 30 / (time.time() - start_time)
                    start_time = time.time()
                
                # Convert to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame
                result = hands.process(rgb_frame)

                gesture = None
                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        # Draw landmarks
                        mp_drawing.draw_landmarks(
                            frame, 
                            hand_landmarks, 
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                        )
                        
                        # Detect gesture
                        gesture = detect_gesture(hand_landmarks)

                if gesture:
                    text = gesture
                    color = gesture_colors.get(text, default_color)
                    font = cv2.FONT_HERSHEY_SIMPLEX

                    # Position in top left corner
                    x, y = 50, 80

                    # Background for text
                    text_size = cv2.getTextSize(text, font, 1.2, 2)[0]
                    cv2.rectangle(frame, (x-10, y-70), (x + text_size[0] + 10, y + 10), (0, 0, 0), -1)
                    
                    # Main text
                    cv2.putText(frame, text, (x, y), font, 1.2, color, 2, cv2.LINE_AA)

                    # Play audio if new gesture detected (with 2 second cooldown)
                    current_time = time.time()
                    if gesture != last_gesture and current_time - last_time > 2:
                        play_audio(text)
                        last_gesture = gesture
                        last_time = current_time
                        
                        # Display audio status
                        cv2.putText(frame, "Speaking...", (50, 120), font, 0.7, (0, 255, 0), 2)

                # Display FPS
                cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Display instruction
                cv2.putText(frame, "ESC to exit", (frame.shape[1] - 150, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Show frame
                cv2.imshow("Hand Gesture Recognition", frame)

                # Break loop on ESC key
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord(' '):  # Space bar to pause
                    cv2.waitKey(0)

    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Signal audio thread to stop
        audio_queue.put(None)
        audio_thread.join(timeout=1)
        
        pygame.mixer.quit()
        print("Program terminated cleanly")

if __name__ == "__main__":
    main()