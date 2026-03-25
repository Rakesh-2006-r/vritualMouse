import os
import json
import av
from yolo_detector import HandDetector

def build_dataset(config_path="config.json"):
    with open(config_path, "r") as f:
        config = json.load(f)
        
    videos_dir = config["videos_dir"]
    dataset_dir = config["dataset_dir"]
    classes = config["classes"]
    frame_skip = config["frame_skip"]
    image_size = config["image_size"]
    
    # Init detector
    detector = HandDetector(model_path=config["yolo_model"])
    
    for cls in classes:
        cls_dir = os.path.join(dataset_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        
        video_path = os.path.join(videos_dir, f"{cls}.mp4")
        if not os.path.exists(video_path):
            print(f"Video not found for class: {cls} at {video_path}")
            continue
            
        print(f"Processing {video_path}...")
        container = av.open(video_path)
        
        frame_count = 0
        saved_count = 0
        
        for frame in container.decode(video=0):
            if frame_count % frame_skip == 0:
                img = frame.to_image() 
                cropped, bbox = detector.detect_and_crop(img)
                if cropped:
                    resized = cropped.resize((image_size, image_size))
                    save_path = os.path.join(cls_dir, f"{cls}_{saved_count}.jpg")
                    resized.save(save_path)
                    saved_count += 1
            frame_count += 1
        print(f"Saved {saved_count} frames for gestue '{cls}'.")

if __name__ == "__main__":
    build_dataset()
