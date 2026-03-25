import json
import time
import imageio
from PIL import Image
import threading
import queue
import sys

from yolo_detector import HandDetector
from inference import GesturePredictor
from mouse_controller import MouseController

frame_queue = queue.Queue(maxsize=1)
detection_queue = queue.Queue(maxsize=1)

def capture_thread(config):
    try:
        reader = imageio.get_reader("<video0>")
        for frame_raw in reader:
            if not frame_queue.full():
                frame = Image.fromarray(frame_raw)
                frame_queue.put(frame)
            time.sleep(1/60.0)
    except Exception as e:
        print(f"Capture error: {e}. Check if webcam is available.")

def detection_thread(config):
    detector = HandDetector(model_path=config["yolo_model"])
    predictor = GesturePredictor(config)
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        frame = frame_queue.get()
        w, h = frame.width, frame.height
        
        cropped, bbox = detector.detect_and_crop(frame)
        if cropped:
            gesture, conf = predictor.predict(cropped)
            if gesture and conf >= config["confidence_threshold"]:
                if not detection_queue.full():
                    detection_queue.put((gesture, bbox, w, h, conf))
        
        # Logging FPS
        frame_count += 1
        if time.time() - start_time >= 1.0:
            fps = frame_count / (time.time() - start_time)
            print(f"[INFO] Detection FPS: {fps:.1f}")
            frame_count = 0
            start_time = time.time()

def action_thread(config):
    controller = MouseController(config)
    while True:
        gesture, bbox, w, h, conf = detection_queue.get()
        print(f"[ACTION] Gesture: {gesture} | Confidence: {conf:.2f}")
        controller.process_gesture(gesture, bbox, w, h)

def main():
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
    except Exception as e:
        print("Could not load config.json:", e)
        sys.exit(1)
        
    print("Starting Main Threads...")
    
    t1 = threading.Thread(target=capture_thread, args=(config,), daemon=True)
    t2 = threading.Thread(target=detection_thread, args=(config,), daemon=True)
    t3 = threading.Thread(target=action_thread, args=(config,), daemon=True)
    
    t1.start()
    t2.start()
    t3.start()
    
    print("Virtual Mouse System is Running. Press Ctrl+C to stop.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting the Virtual Mouse System...")

if __name__ == "__main__":
    main()
