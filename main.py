import cv2
import time
import os
import sys
import numpy as np
import pyautogui
import typing

# MediaPipe new Tasks API (0.10+) logic
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    print("Warning: mediapipe not installed. Run: pip install mediapipe opencv-python")

# Custom Module Imports
try:
    from mouse_actions import MouseActionController
    from kalman_filter import KalmanFilterStabilizer
    from gesture_engine import GestureEngine
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    print(f"Warning: Local modules missing ({e}). Ensure all .py files are in the same folder.")

# Constants and Configuration
INDEX_FINGER_TIP = 8
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"
SCREEN_W, SCREEN_H = pyautogui.size()

# Connections for manual drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (0, 9), (9, 10), (10, 11), (11, 12), # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),# Ring
    (0, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (5, 9), (9, 13), (13, 17)            # Palm
]

class HandTracker:
    def __init__(self, model_path: str = MODEL_PATH):
        self.landmarker = None
        if not MP_AVAILABLE:
            return

        if not os.path.exists(model_path):
            print(f"[HandTracker] Downloading model to {model_path}...")
            import urllib.request
            try:
                urllib.request.urlretrieve(MODEL_URL, model_path)
                print("[HandTracker] Download complete.")
            except Exception as e:
                print(f"[HandTracker] Download failed: {e}")
                return

        # Configure detection options
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)
        print("[HandTracker] HandLandmarker initialized.")

    def find_hands(self, img: np.ndarray, timestamp_ms: int, draw: bool = True) -> typing.Tuple[np.ndarray, list]:
        """Runs hand detection and optionally draws landmarks. Returns results as a list of landmark tips."""
        if self.landmarker is None:
            return img, []
        
        h, w = img.shape[:2]
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        
        hands_data = []
        if result.hand_landmarks:
            for hand_lms in result.hand_landmarks:
                hands_data.append(hand_lms)
                if draw:
                    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
                    # Draw landmark points
                    for pt in pts:
                        cv2.circle(img, pt, 4, (0, 255, 0), cv2.FILLED)
                    # Draw connectivity lines
                    for a, b in HAND_CONNECTIONS:
                        cv2.line(img, pts[a], pts[b], (200, 200, 200), 1)
        
        return img, hands_data

def main():
    if not (MP_AVAILABLE and MODULES_AVAILABLE):
        print("Missing requirements. Exit.")
        return

    print("Starting Virtual Mouse AI...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Core system components
    tracker      = HandTracker()
    mouse        = MouseActionController()
    kalman       = KalmanFilterStabilizer(process_noise=0.04, measurement_noise=0.15)
    gestures     = GestureEngine()
    
    # Persistent State
    is_left_down = False
    last_click_t = 0.0
    pTime        = 0.0
    start_t      = time.time()
    
    print("Ready. Press 'q' in webcam window to exit.")

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Current timestamp for MediaPipe Video Mode
            timestamp_ms = int((time.time() - start_t) * 1000)
            
            # 1. Detection
            frame, results = tracker.find_hands(frame, timestamp_ms)
            
            if results:
                # 2. Coordinate Extraction
                hand_lms = results[0]
                raw_cx, raw_cy = int(hand_lms[INDEX_FINGER_TIP].x * w), int(hand_lms[INDEX_FINGER_TIP].y * h)

                # 3. Gesture Classification
                gest_name, score = gestures.classify(hand_lms)

                # 4. Kalman Stabilization
                smooth_cx, smooth_cy = kalman.update(raw_cx, raw_cy)
                
                # 5. Coordinate Mapping (Resolution Scale)
                margin = 80
                scr_x = np.interp(smooth_cx, [margin, 640 - margin], [0, SCREEN_W])
                scr_y = np.interp(smooth_cy, [margin, 480 - margin], [0, SCREEN_H])

                # --- Execution Switch ---
                
                # A. MOVE MODE
                if gest_name == "MOVE":
                    if is_left_down:
                        mouse.end_drag()
                        is_left_down = False
                    mouse.move_cursor(scr_x, scr_y, duration=0)
                    cv2.circle(frame, (int(smooth_cx), int(smooth_cy)), 10, (0, 255, 0), cv2.FILLED)

                # B. LEFT CLICK / DRAG (Index Pinch)
                elif gest_name == "LEFT_CLICK":
                    if not is_left_down:
                        mouse.start_drag()
                        is_left_down = True
                    mouse.move_cursor(scr_x, scr_y, duration=0)
                    cv2.circle(frame, (int(smooth_cx), int(smooth_cy)), 15, (255, 0, 0), cv2.FILLED)

                # C. RIGHT CLICK (Middle Pinch)
                elif gest_name == "RIGHT_CLICK":
                    curr_t = time.time()
                    if curr_t - last_click_t > 0.5:
                        mouse.right_click()
                        last_click_t = curr_t
                    cv2.circle(frame, (int(smooth_cx), int(smooth_cy)), 15, (0, 0, 255), cv2.FILLED)

                # D. VOLUME (Index & Middle Both Up)
                elif gest_name == "VOLUME_CONTROL":
                    if raw_cy < 150:
                        mouse.set_volume_relative(increase=True, steps=1)
                    elif raw_cy > 330:
                        mouse.set_volume_relative(increase=False, steps=1)
                    cv2.putText(frame, "VOLUME CONTROL", (200, 450), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

                # E. BRIGHTNESS (Pinky Pinch)
                elif gest_name == "BRIGHTNESS_CONTROL":
                    bri_level = max(0, min(100, int(100 - (raw_cy / 480 * 100))))
                    mouse.set_brightness(bri_level)
                    cv2.putText(frame, f"BRIGHTNESS: {bri_level}%", (200, 450), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

                # F. SCREENSHOT (Fist)
                elif gest_name == "SCREENSHOT":
                    curr_t = time.time()
                    if curr_t - last_click_t > 2.0:
                        mouse.take_screenshot()
                        last_click_t = curr_t
                    cv2.putText(frame, "SCREENSHOT!", (200, 200), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

                cv2.putText(frame, f"MODE: {gest_name}", (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

            # Performance: FPS Display
            cTime = time.time()
            fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
            pTime = cTime
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            
            cv2.imshow("Virtual Mouse - AI Engine", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Runtime Error: {e}")
    finally:
        print("Cleaning up camera and windows...")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
