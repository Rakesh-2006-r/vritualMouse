import numpy as np

class KalmanFilterStabilizer:
    """
    Kalman Filter for smoothing 2D coordinates (x, y) 
    to prevent jitter during mouse movement.
    """
    def __init__(self, process_noise: float = 0.03, measurement_noise: float = 0.1):
        # State vector [x, y, dx, dy] (position and velocity)
        self.state = np.zeros((4, 1), dtype=np.float32)
        
        # State transition matrix F (constant velocity model)
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # Measurement matrix H (we only observe x and y)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        # Covariance matrices
        self.P = np.eye(4, dtype=np.float32) * 1000.0  # Initial uncertainty
        self.Q = np.eye(4, dtype=np.float32) * process_noise
        self.R = np.eye(2, dtype=np.float32) * measurement_noise
        
        self.initialized = False

    def update(self, x: float, y: float):
        """Update filter with new measures (x, y) and return smoothed (x, y) as floats."""
        if not self.initialized:
            self.state[:2] = [[x], [y]]
            self.initialized = True
            return float(x), float(y)

        # Prediction
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Update
        z = np.array([[x], [y]], dtype=np.float32)
        y_err = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y_err
        self.P = (np.eye(4) - K @ self.H) @ self.P

        return float(self.state[0, 0]), float(self.state[1, 0])
