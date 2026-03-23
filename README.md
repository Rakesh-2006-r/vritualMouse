# Virtual Mouse AI Control

A professional AI-powered virtual mouse system that uses hand gestures to control Windows mouse actions. 

## ✨ Features
- **High-Precision Cursor Control**: Uses a **Kalman Filter** to smooth out hand jitter and provide steady cursor movement.
- **Gesture Recognition**:
  - **Single Finger/Pinch**: Left Click / Drag.
  - **Two-Finger Pinch**: Right Click.
  - **Fist**: Instant Screenshot.
  - **Hand Positions**: System Volume and Brightness control.
- **MediaPipe Tasks API**: Leveraging the latest Google MediaPipe 0.10+ vision models for robust hand tracking.

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

3.  Download the model:
    The app will automatically download the `hand_landmarker.task` file on first run.

## 🖱️ Usage
Run the main script:
```bash
python main.py
```

## 🛠️ Tech Stack
- Python 3.10+
- MediaPipe (HandLandmarker Tasks)
- OpenCV
- PyAutoGUI
- NumPy
