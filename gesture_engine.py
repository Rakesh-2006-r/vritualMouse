"""
gesture_engine.py  —  ViT-based classifier for YOLO hand crops
================================================================

This engine fulfills the "YOLO + ViT" requirement.
It takes a cropped image of the hand (found via YOLO pose wrist keypoints)
and runs it through a HuggingFace Vision Transformer (ViT) trained on hand gestures.

Model: dima806/hand_gestures_image_detection
"""

import time
import typing
import numpy as np
from PIL import Image

try:
    from transformers import pipeline
    VIT_AVAILABLE = True
except ImportError:
    VIT_AVAILABLE = False


class GestureEngine:
    STABLE_FRAMES = 3

    # ViT model labels mapped to our action triggers
    # Labels: call, dislike, fist, four, like, mute, ok, one, palm, peace,
    #         peace_inverted, rock, stop, stop_inverted, three, three2, two_up, two_up_inverted
    VIT_MAPPING = {
        "one": "INDEX_ONLY",           # Left Click
        "peace": "V_SIGN",             # Right Click / Double Click
        "peace_inverted": "V_SIGN",
        "two_up": "V_SIGN",
        "two_up_inverted": "V_SIGN",
        "fist": "FIST",                # Drag / Scroll / Select Multi
        "rock": "FIST",
        "dislike": "FIST",             # treat thumb down as fist if misclassified
        "palm": "OPEN_PALM",           # Move / Screenshot
        "stop": "OPEN_PALM",
        "stop_inverted": "OPEN_PALM",
        "four": "OPEN_PALM",
        "ok": "PINCH",                 # Volume / Brightness
        "mute": "PINCH",
        "like": "PINCH",               # sometimes thumb up looks like a pinch to model
    }

    def __init__(self):
        self._history: list[str] = []

        if VIT_AVAILABLE:
            print("Loading ViT Gesture Model (this may take a few seconds)...")
            # Run model on CPU/GPU depending on device (pipeline handles this if installed properly, 
            # or we explicitly specify device if torch is available, but pipeline defaults to CPU usually 
            # if no device=0 is provided. We'll let it use default to avoid CUDA errors).
            self.classifier = pipeline(
                "image-classification", 
                model="dima806/hand_gestures_image_detection"
            )
            print("ViT Model loaded.")
        else:
            self.classifier = None
            print("WARNING: transformers not installed. ViT will not work.")

    def classify_from_image(
        self,
        hand_crop: Image.Image,
    ) -> typing.Tuple[str, float]:
        """
        Classifies a PIL Image crop of the hand using ViT.
        Safe to call from a background thread.
        """
        if self.classifier is None or hand_crop is None:
            return "IDLE", 0.0

        # Run ViT
        # pipeline returns a list of dicts: [{'label': 'palm', 'score': 0.99}, ...]
        try:
            results = self.classifier(hand_crop)
            top_result = results[0]
            label = top_result['label']
            score = top_result['score']
        except Exception as e:
            print(f"ViT Error: {e}")
            return self._stable("IDLE", 0.0)

        # Map ViT label to our internal gesture logic
        raw = self.VIT_MAPPING.get(label, "IDLE")
        
        # We enforce a higher confidence threshold for PINCH / INDEX
        if raw in ("PINCH", "INDEX_ONLY") and score < 0.6:
            raw = "IDLE"

        return self._stable(raw, score)

    def map_to_action(
        self,
        gesture: str,
        vy: float,
        brightness_mode: bool = False,
    ) -> str:
        """Video-accurate mapping based on gesture rules."""
        if gesture == "INDEX_ONLY":
            return "LEFT_CLICK"
        elif gesture == "V_SIGN":
            return "RIGHT_CLICK"
        elif gesture == "OPEN_PALM":
            return "MOVE"
        elif gesture == "FIST":
            if abs(vy) > 80:
                return "SCROLL"
            return "DRAG"
        elif gesture == "PINCH":
            return "BRIGHTNESS" if brightness_mode else "VOLUME"
        return "IDLE"

    def _stable(self, gesture: str, conf: float) -> typing.Tuple[str, float]:
        """Buffer system to prevent flickering."""
        self._history.append(gesture)
        if len(self._history) > self.STABLE_FRAMES:
            self._history.pop(0)

        if len(self._history) == self.STABLE_FRAMES and \
           all(g == gesture for g in self._history):
            return gesture, conf

        from collections import Counter
        most_common = Counter(self._history).most_common(1)[0][0]
        return most_common, conf * 0.7


