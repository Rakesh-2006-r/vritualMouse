import screen_brightness_control as sbc
from ctypes import cast, POINTER
from comtypes import CoInitialize, CoUninitialize
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

class SystemController:
    def __init__(self):
        # Initialize COM in the caller's thread
        CoInitialize()
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, 1, None)
        self.volume = cast(interface, POINTER(IAudioEndpointVolume))
        self.vol_range = self.volume.GetVolumeRange()
        
    def __del__(self):
        CoUninitialize()

    def set_brightness(self, level):
        try:
            sbc.set_brightness(level)
        except Exception as e:
            print(f"Brightness Error: {e}")

    def set_volume(self, level):
        min_vol = self.vol_range[0]
        max_vol = self.vol_range[1]
        target_vol = min_vol + (max_vol - min_vol) * level
        self.volume.SetMasterVolumeLevel(target_vol, None)
