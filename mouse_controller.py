import pyautogui
from system_control import SystemController
import numpy as np
import time

pyautogui.FAILSAFE = False

class MouseController:
    def __init__(self, config):
        self.sys_ctrl = SystemController()
        self.screen_w, self.screen_h = pyautogui.size()
        self.alpha = config["alpha_smoothing"]
        self.prev_x, self.prev_y = self.screen_w // 2, self.screen_h // 2
        
        self.cooldown_frames = config["cooldown_frames"]
        self.cooldown_counter = 0

    def process_gesture(self, gesture, bbox, frame_w, frame_h):
        # Centers
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # Invert X for mirror logic
        cx = frame_w - cx
        
        # Add center margin calibration mapping
        margin_x = frame_w * 0.2
        margin_y = frame_h * 0.2
        
        target_x = np.interp(cx, (margin_x, frame_w - margin_x), (0, self.screen_w))
        target_y = np.interp(cy, (margin_y, frame_h - margin_y), (0, self.screen_h))
        
        # Exponential smoothing
        curr_x = int(self.prev_x + self.alpha * (target_x - self.prev_x))
        curr_y = int(self.prev_y + self.alpha * (target_y - self.prev_y))
        self.prev_x, self.prev_y = curr_x, curr_y
        
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            if gesture not in ["move", "drag"]:
                return

        if gesture == "move":
            pyautogui.moveTo(curr_x, curr_y)
            
        elif gesture == "drag":
            pyautogui.dragTo(curr_x, curr_y, button='left')
            
        elif gesture == "left_click":
            pyautogui.click()
            self.cooldown_counter = self.cooldown_frames
            
        elif gesture == "right_click":
            pyautogui.rightClick()
            self.cooldown_counter = self.cooldown_frames
            
        elif gesture == "double_click":
            pyautogui.doubleClick()
            self.cooldown_counter = self.cooldown_frames
            
        elif gesture == "scroll":
            # Map up/down position
            if cy < frame_h / 2:
                pyautogui.scroll(150)
            else:
                pyautogui.scroll(-150)
                
        elif gesture == "brightness":
            level = int(np.interp(cy, (margin_y, frame_h - margin_y), (100, 0))) 
            level = max(0, min(100, level))
            self.sys_ctrl.set_brightness(level)
            
        elif gesture == "volume":
            level = np.interp(cy, (margin_y, frame_h - margin_y), (1.0, 0.0))
            level = max(0.0, min(1.0, level))
            self.sys_ctrl.set_volume(level)
            
        elif gesture == "screenshot":
            pyautogui.screenshot(f"screenshot_{int(time.time())}.png")
            self.cooldown_counter = self.cooldown_frames * 2
            
        elif gesture == "multi_select":
            pyautogui.keyDown('ctrl')
            pyautogui.click(curr_x, curr_y)
            pyautogui.keyUp('ctrl')
            self.cooldown_counter = self.cooldown_frames
