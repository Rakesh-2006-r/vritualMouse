import pyautogui
import time
import math
try:
    import screen_brightness_control as sbc
except ImportError:
    sbc = None
    print("Warning: screen_brightness_control not installed.")

# Optional – pycaw for true absolute volume control on Windows
try:
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    _devices = AudioUtilities.GetSpeakers()
    _interface = _devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    _volume = cast(_interface, POINTER(IAudioEndpointVolume))
    PYCAW_AVAILABLE = True
except Exception:
    _volume = None
    PYCAW_AVAILABLE = False


class MouseActionController:
    """
    Executes mouse / system actions from gesture commands.
    Supports: move, left-click, right-click, double-click,
              drag, scroll, volume, brightness, screenshot, select-multiple.
    """
    def __init__(self):
        pyautogui.FAILSAFE = False
        self.is_dragging = False
        self._last_scroll_y = None
        self._last_vol_spread = None
        self._last_bri_spread = None

    # ------------------------------------------------------------------ cursor
    def move_cursor(self, x, y, duration=0.0):
        pyautogui.moveTo(int(x), int(y), duration=duration)

    # ------------------------------------------------------------------ clicks
    def left_click(self):
        pyautogui.click(button='left')
        print("Action: Left Click")

    def right_click(self):
        pyautogui.click(button='right')
        print("Action: Right Click")

    def double_click(self):
        pyautogui.doubleClick()
        print("Action: Double Click")

    # ------------------------------------------------------------------ drag
    def start_drag(self):
        if not self.is_dragging:
            pyautogui.mouseDown(button='left')
            self.is_dragging = True
            print("Action: Drag Started")

    def end_drag(self):
        if self.is_dragging:
            pyautogui.mouseUp(button='left')
            self.is_dragging = False
            print("Action: Drag Ended")

    # ------------------------------------------------------------------ scroll
    def scroll(self, amount: int):
        """
        Vertical scrolling driven directly by velocity/amount.
        Positive amount = scroll up, negative = scroll down.
        """
        pyautogui.scroll(amount)

    # ------------------------------------------------------------------ select multiple
    def select_multiple_start(self, x, y):
        """Start a lasso-selection drag."""
        self._sel_start = (x, y)
        pyautogui.moveTo(x, y)
        pyautogui.mouseDown(button='left')
        print("Action: Select Multiple — Start")

    def select_multiple_update(self, x, y):
        pyautogui.moveTo(x, y)

    def select_multiple_end(self):
        pyautogui.mouseUp(button='left')
        print("Action: Select Multiple — End")

    # ------------------------------------------------------------------ volume
    def set_volume_from_spread(self, spread: float):
        """
        spread: normalised 0-1 distance between index tip and thumb tip.
        Maps linearly to 0-100% volume.
        """
        level = max(0.0, min(1.0, spread))
        if PYCAW_AVAILABLE and _volume is not None:
            # SetMasterVolumeLevelScalar maps 0.0-1.0 linearly to the correct dB
            _volume.SetMasterVolumeLevelScalar(level, None)
        else:
            # Fallback: keypress simulation
            steps = int(level * 50)
            for _ in range(2):
                pyautogui.press('volumeup' if level > 0.5 else 'volumedown')

    def set_volume_relative(self, increase=True, steps=2):
        key = 'volumeup' if increase else 'volumedown'
        for _ in range(steps):
            pyautogui.press(key)
        print(f"Action: Volume {'Up' if increase else 'Down'}")

    # ------------------------------------------------------------------ brightness
    def set_brightness_from_spread(self, spread: float):
        """spread: normalised 0-1."""
        level = int(max(0, min(100, spread * 100)))
        if sbc is not None:
            sbc.set_brightness(level)
            print(f"Action: Brightness → {level}%")
        else:
            print("Action: Brightness skipped (screen_brightness_control missing)")

    def set_brightness(self, level_percentage: float):
        if sbc is not None:
            level = max(0, min(100, int(level_percentage)))
            sbc.set_brightness(level)
            print(f"Action: Brightness → {level}%")

    # ------------------------------------------------------------------ screenshot
    def take_screenshot(self, filename="screenshot.png"):
        try:
            pyautogui.screenshot().save(filename)
            print(f"Action: Screenshot saved → {filename}")
        except Exception as e:
            print(f"Action: Screenshot failed — {e}")


if __name__ == "__main__":
    controller = MouseActionController()
    print("MouseActionController initialised OK.")
