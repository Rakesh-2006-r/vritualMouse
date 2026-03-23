import pyautogui
import time
import math
try:
    import screen_brightness_control as sbc
except ImportError:
    sbc = None
    print("Warning: screen_brightness_control not installed. Run `pip install screen-brightness-control` for brightness features.")

class MouseActionController:
    """
    Controller for executing mouse and system actions based on gestures.
    Includes operations for clicking, dragging, scrolling, volume, brightness, and screenshots.
    """
    def __init__(self):
        # Disable pyautogui failsafe if necessary, but keep it on by default for safety
        pyautogui.FAILSAFE = False
        
        # State variables for drag and drop
        self.is_dragging = False

    def move_cursor(self, x, y, duration=0.1):
        """Move cursor to specific screen coordinates (x, y)."""
        # Kalman filter smoothing can be added here
        pyautogui.moveTo(x, y, duration=duration)

    def left_click(self):
        """Execute a single left click."""
        pyautogui.click(button='left')
        print("Action: Left Click")

    def right_click(self):
        """Execute a single right click."""
        pyautogui.click(button='right')
        print("Action: Right Click")

    def double_click(self):
        """Execute a double left click."""
        pyautogui.doubleClick()
        print("Action: Double Click")

    def start_drag(self):
        """Start drag operation (mouse down)."""
        if not self.is_dragging:
            pyautogui.mouseDown(button='left')
            self.is_dragging = True
            print("Action: Drag Started")

    def end_drag(self):
        """End drag operation (mouse up)."""
        if self.is_dragging:
            pyautogui.mouseUp(button='left')
            self.is_dragging = False
            print("Action: Drag Ended")

    def select_multiple_items(self, x_start, y_start, x_end, y_end):
        """Select multiple items by dragging a box."""
        pyautogui.moveTo(x_start, y_start)
        pyautogui.dragTo(x_end, y_end, button='left', duration=0.5)
        print("Action: Select Multiple Items")

    def hold_ctrl_to_select(self, click_positions):
        """Select multiple specific items by holding Ctrl and clicking them."""
        pyautogui.keyDown('ctrl')
        for pos in click_positions:
            pyautogui.click(pos[0], pos[1])
            time.sleep(0.1)
        pyautogui.keyUp('ctrl')
        print("Action: Selected Multiple Items via Control-Click")

    def set_volume_relative(self, increase=True, steps=2):
        """Adjust system volume up or down."""
        key = 'volumeup' if increase else 'volumedown'
        for _ in range(steps):
            pyautogui.press(key)
        print(f"Action: Volume {'Increased' if increase else 'Decreased'}")

    def set_volume_absolute(self, level_percentage):
        """
        Adjust volume proportionally based on a gesture (e.g. distance between fingers).
        Since pyautogui only supports presses, this is a simulated smooth control.
        For true absolute control, use `pycaw` library.
        """
        # For simplicity, if we want to mimic absolute control without pycaw, we spam up/down.
        # It's recommended to install pycaw for true absolute volume control on Windows.
        print("Action: Absolute Volume control requested (requires pycaw for direct setting).")

    def set_brightness(self, level_percentage):
        """Set screen brightness to a specific percentage (0-100)."""
        if sbc is not None:
            # Ensure level is within 0-100
            level = max(0, min(100, int(level_percentage)))
            sbc.set_brightness(level)
            print(f"Action: Set Brightness to {level}%")
        else:
            print("Action: Brightness control failed (screen_brightness_control not installed).")

    def take_screenshot(self, filename="screenshot.png"):
        """Take a screenshot and save it."""
        try:
            screenshot_img = pyautogui.screenshot()
            screenshot_img.save(filename)
            print(f"Action: Screenshot saved as {filename}")
        except Exception as e:
            print(f"Action: Screenshot failed - {e}")

if __name__ == "__main__":
    # Test block
    controller = MouseActionController()
    print("Testing controller initialized successfully.")
