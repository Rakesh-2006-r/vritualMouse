import cv2
import numpy as np

class OpticalFlowAnalyzer:
    """
    Analyzes motion between frames using Lucas-Kanade Optical Flow.
    Can be used to detect scrolling or dragging vectors.
    """
    def __init__(self):
        self.prev_gray = None
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def analyze(self, frame, points):
        """
        Calculates motion for specific points (like finger tips).
        Returns the average motion vector (dx, dy).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return 0, 0

        # Points to track (should be float32)
        p0 = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        
        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None, **self.lk_params)
        
        dx, dy = 0, 0
        if p1 is not None:
            # Filter good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            
            if len(good_new) > 0:
                # Calculate simple average displacement
                diff = good_new - good_old
                avg_diff = np.mean(diff, axis=0)
                dx, dy = avg_diff[0], avg_diff[1]

        self.prev_gray = gray
        return dx, dy
