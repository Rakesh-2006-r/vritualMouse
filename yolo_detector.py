from ultralytics import YOLO

class HandDetector:
    def __init__(self, model_path="yolov8n.pt", conf_thresh=0.4):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh

    def detect_and_crop(self, pil_image):
        # Infer using YOLO
        results = self.model.predict(pil_image, conf=self.conf_thresh, verbose=False)
        if len(results) == 0 or len(results[0].boxes) == 0:
            return None, None
            
        # Select best detection box (assuming standard person/hand class detection)
        boxes = results[0].boxes
        best_box = boxes[0].xyxy[0].cpu().numpy()
        
        x1, y1, x2, y2 = map(int, best_box)
        margin = 15
        
        # Add margin to include more context around the hand
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(pil_image.width, x2 + margin)
        y2 = min(pil_image.height, y2 + margin)
        
        cropped = pil_image.crop((x1, y1, x2, y2))
        return cropped, (x1, y1, x2, y2)
