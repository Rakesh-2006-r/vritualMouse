import numpy as np
import typing

class GestureEngine:
    """
    Classifies gestures from hand landmarks to identify mouse/system actions.
    Supported: LEFT_CLICK, RIGHT_CLICK, MOVE, VOLUME_CONTROL, BRIGHTNESS_CONTROL, SCREENSHOT.
    """
    def __init__(self):
        # State tracking
        self.last_pinch = False
        self.last_gesture = "IDLE"

    def get_distance(self, p1: typing.Any, p2: typing.Any) -> float:
        """Distance between two landmarks (Normalized x, y, z)."""
        return float(np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2))

    def is_finger_up(self, landmarks: typing.List[typing.Any], finger_index: int) -> bool:
        """Heuristic to check if a finger is extended (index=1-4)."""
        # Tip index: 8 (index), 12 (middle), 16 (ring), 20 (pinky)
        # Base index: 6 (index), 10 (middle), 14 (ring), 18 (pinky)
        tip_idx = finger_index * 4 + 4
        base_idx = tip_idx - 2
        return landmarks[tip_idx].y < landmarks[base_idx].y

    def classify(self, landmarks: typing.List[typing.Any]) -> typing.Tuple[str, float]:
        """Classifies pose from MediaPipe landmarks."""
        if not landmarks:
            return "No Hand", 0.0

        # Primary tips
        thumb_tip  = landmarks[4]
        index_tip  = landmarks[8]
        middle_tip = landmarks[12]
        pinky_tip  = landmarks[20]

        # Pinch distances (Thumb + Finger)
        t_index_dist  = self.get_distance(thumb_tip, index_tip)
        t_middle_dist = self.get_distance(thumb_tip, middle_tip)
        t_pinky_dist  = self.get_distance(thumb_tip, pinky_tip)

        # Threshold to classify as a "pinch" (meeting)
        pinch_thresh = 0.05
        
        # --- Gesture Priority List ---
        
        # 1. SCREENSHOT (Fist detector)
        if not any(self.is_finger_up(landmarks, i) for i in range(1, 5)):
            return "SCREENSHOT", 1.0

        # 2. BRIGHTNESS (Pinch Pinky)
        if t_pinky_dist < pinch_thresh:
            return "BRIGHTNESS_CONTROL", t_pinky_dist

        # 3. RIGHT_CLICK (Pinch Middle)
        if t_middle_dist < pinch_thresh:
            return "RIGHT_CLICK", t_middle_dist

        # 4. LEFT_CLICK / DRAG (Pinch Index)
        if t_index_dist < pinch_thresh:
            return "LEFT_CLICK", t_index_dist
        
        # 5. VOLUME (Index & Middle Up)
        if self.is_finger_up(landmarks, 1) and self.is_finger_up(landmarks, 2):
            return "VOLUME_CONTROL", 1.0

        # 6. MOVE (Only Index Up)
        if self.is_finger_up(landmarks, 1):
            return "MOVE", 1.0

        return "IDLE", 0.0
