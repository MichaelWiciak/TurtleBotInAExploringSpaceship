"""
Represents detections that the computer vision shows
"""

from enum import Enum
from .cv_planets import Planet


class DetectionSwitch:
    def __init__(self) -> None:
        self.detect_potential_windows = True
        self.detect_status_light = True

    def set_detect_potential_windows(self, value: bool):
        self.detect_potential_windows = value
        return self

    def set_detect_status_light(self, value: bool):
        self.detect_status_light = value
        return self


class PlanetEnum(Enum):
    EARTH = "earth"
    MOON = "moon"
    OTHER = "other"
    NOT_DETECTED = "no planets"

    def __str__(self) -> str:
        return self.value


class StatusEnum(Enum):
    GREEN = "green"
    RED = "red"

    def __str__(self) -> str:
        return self.value


class Detection:

    def __init__(self, x: float, y: float, size: float) -> None:
        self.x = x
        self.y = y
        self.size = size

    def __str__(self) -> str:
        # return f"Detection @ x: {self.x}, y: {self.y}, size: {self.size}"
        return f"Something detected"


class StatusDetection(Detection):

    def __init__(self, x: float, y: float, size: float, status: StatusEnum) -> None:
        super().__init__(x, y, size)
        self.status = status

    def __str__(self) -> str:
        # return f"StatusDetection @ x: {self.x}, y: {self.y}, size: {self.size}, status: {self.status}"
        return f"Status detected ({self.status})"


class WindowDetection(Detection):

    def __init__(
        self,
        x: float,
        y: float,
        size: float,
        height: float,
        planet: PlanetEnum,
        planet_data: Planet,
        capture,
        capture_drawing,
    ) -> None:
        super().__init__(x, y, size)
        self.height = height
        self.planet = planet
        self.planet_data = planet_data
        self.capture = capture
        self.capture_drawing = capture_drawing

    def __str__(self) -> str:
        # return f"WindowDetection @ x: {self.x}, y: {self.y}, size: {self.size}, planet: {self.planet}"
        return f"Window detected ({self.planet})"


class PosterDetection(Detection):

    def __init__(
        self,
        x: float,
        y: float,
        size: float,
        capture,
        capture_drawing,
    ) -> None:
        super().__init__(x, y, size)
        self.capture = capture
        self.capture_drawing = capture_drawing

    def __str__(self) -> str:
        # return f"PosterDetection @ x: {self.x}, y: {self.y}, size: {self.size}"
        return f"Poster detected"


class DetectMetadata:
    def __init__(self, window_s: float, status_s: float):
        self.window_s = window_s
        self.status_s = status_s
