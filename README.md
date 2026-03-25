# Virtual Mouse AI Control

A professional AI-powered virtual mouse system that uses hand gestures to control Windows mouse actions. 

## ✨ Features
- **High-Precision Cursor Control**: Uses a **Kalman Filter** to smooth out hand jitter and provide steady cursor movement.
- **Gesture Recognition**:
  - Uses state of the art **Vision Transformer (ViT)** to classify hand crops and perform commands.
  - Actions Supported: Click, Right-Click, Move, Screenshot, Volume, Brightness.
- **YOLO Detection API**: Leveraging the **YOLO (Ultralytics)** model for robust real-time object tracking and parsing bounding boxes.

## 🚀 Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Rakesh-2006-r/vritualMouse.git
    cd vritualMouse
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Download models (Automatic):
    The app will automatically download YOLO and ViT model bases on first run using PyTorch Hub and hugging face transformers.

## 🖱️ Usage
Run the main script:
```bash
py main.py
```

## 🛠️ Tech Stack
- Python 3.10+
- Ultralytics YOLO
- Transformers (ViT)
- PyTorch
- PyAutoGUI
- NumPy
